import copy
import logging
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from yarr.agents.agent import Agent, Summary, ActResult, \
    ScalarSummary, HistogramSummary

from arm import utils
from arm.utils import stack_on_channel

NAME = 'BCAgent'
REPLAY_ALPHA = 0.7
REPLAY_BETA = 1.0


class Actor(nn.Module):

    def __init__(self, actor_network: nn.Module):
        super(Actor, self).__init__()
        self._actor_network = copy.deepcopy(actor_network)
        self._actor_network.build()

    def forward(self, observations, robot_state):
        mu = self._actor_network(observations, robot_state)
        return mu


class BCAgent(Agent):

    def __init__(self,
                 nbv_actor_network: nn.Module,
                 nbp_actor_network: nn.Module,
                 camera_name: str,
                 lr: float = 0.01,
                 weight_decay: float = 1e-5,
                 grad_clip: float = 20.0):
        self._camera_name = camera_name
        self._nbv_actor_network = nbv_actor_network
        self._nbp_actor_network = nbp_actor_network
        
        self._lr = lr
        self._weight_decay = weight_decay
        self._grad_clip = grad_clip

    def build(self, training: bool, device: torch.device = None):
        if device is None:
            device = torch.device('cpu')
        self._nbv_actor = Actor(self._nbv_actor_network).to(device).train(training)
        self._nbp_actor = Actor(self._nbp_actor_network).to(device).train(training)

        if training:
            self._nbv_actor_optimizer = torch.optim.Adam(
                self._nbv_actor.parameters(), lr=self._lr,
                weight_decay=self._weight_decay)
            self._nbp_actor_optimizer = torch.optim.Adam(
                self._nbp_actor.parameters(), lr=self._lr,
                weight_decay=self._weight_decay)
            
            logging.info('#NBV Actor Params: %d' % sum(
                p.numel() for p in self._nbv_actor.parameters() if p.requires_grad))
            logging.info('#NBP Actor Params: %d' % sum(
                p.numel() for p in self._nbp_actor.parameters() if p.requires_grad))
        else:
            for p in self._nbv_actor.parameters():
                p.requires_grad = False
            for p in self._nbp_actor.parameters():
                p.requires_grad = False
        self._device = device

    def _grad_step(self, loss, opt, model_params=None, clip=None):
        opt.zero_grad()
        loss.backward()
        if clip is not None and model_params is not None:
            nn.utils.clip_grad_value_(model_params, clip)
        opt.step()

    def update(self, step: int, replay_sample: dict) -> dict:

        robot_state_tp0 = stack_on_channel(replay_sample['low_dim_state_layer_0'][:, -1:])
        robot_state_tp0_1 = stack_on_channel(replay_sample['low_dim_state_layer_1'][:, -1:])
        
        action_tp0 = stack_on_channel(replay_sample['action_layer_0'][:, -1:])
        action_tp0_1 = stack_on_channel(replay_sample['action_layer_1'][:, -1:])

        observations_tp0 = [
            stack_on_channel(replay_sample['%s_rgb_layer_0' % self._camera_name]),
            stack_on_channel(replay_sample['%s_point_cloud_layer_0' % self._camera_name])
        ]
        observations_tp0_1 = [
            stack_on_channel(replay_sample['%s_rgb_layer_1' % self._camera_name]),
            stack_on_channel(replay_sample['%s_point_cloud_layer_1' % self._camera_name])
        ]

        nbv_mu = self._nbv_actor(observations_tp0, robot_state_tp0)
        nbp_mu = self._nbp_actor(observations_tp0_1, robot_state_tp0_1)
        loss_weights = utils.loss_weights(replay_sample, REPLAY_BETA)
        nbv_delta = F.mse_loss(
            nbv_mu, action_tp0, reduction='none').mean(1)
        nbp_delta = F.mse_loss(
            nbp_mu, action_tp0_1, reduction='none').mean(1)
        
        nbv_loss = (nbv_delta * loss_weights).mean()
        nbp_loss = (nbp_delta * loss_weights).mean()
        self._grad_step(nbv_loss, self._nbv_actor_optimizer,
                        self._nbv_actor.parameters(), self._grad_clip)
        self._grad_step(nbp_loss, self._nbp_actor_optimizer,
                        self._nbp_actor.parameters(), self._grad_clip)

        self._summaries = {
            'nbv_pi/loss': nbv_loss,
            'nbv_pi/mu': nbv_mu.mean(),
            'nbp_pi/loss': nbp_loss,
            'nbp_pi/mu': nbp_mu.mean(),}
        delta = (nbv_delta+nbp_delta)/2
        return {'priority': delta ** REPLAY_ALPHA}

    def _normalize_quat(self, x):
        return x / x.square().sum(dim=1).sqrt().unsqueeze(-1)

    def act(self, step: int, observation: dict, layer:int,
            deterministic=False) -> ActResult:
        with torch.no_grad():
            observations = [
                stack_on_channel(observation['%s_rgb' % self._camera_name]),
                stack_on_channel(observation['%s_point_cloud' % self._camera_name])
            ]
            robot_state = stack_on_channel(observation['low_dim_state'][:, -1:])
            if layer==0:
                mu = self._nbv_actor(observations, robot_state)
                mu = mu[:, :2]
                return ActResult(mu[0])
            else:
                mu = self._nbp_actor(observations, robot_state)
                mu = torch.cat(
                    [mu[:, :3], self._normalize_quat(mu[:, 3:7]), mu[:, 7:]], dim=-1)
                return ActResult(mu[0])


    def update_summaries(self) -> List[Summary]:
        summaries = []
        for n, v in self._summaries.items():
            summaries.append(ScalarSummary('%s/%s' % (NAME, n), v))

        for tag, param in self._nbv_actor.named_parameters():
            summaries.append(
                HistogramSummary('nbv/%s/gradient/%s' % (NAME, tag), param.grad))
            summaries.append(
                HistogramSummary('nbv/%s/weight/%s' % (NAME, tag), param.data))
            
        for tag, param in self._nbp_actor.named_parameters():
            summaries.append(
                HistogramSummary('nbp/%s/gradient/%s' % (NAME, tag), param.grad))
            summaries.append(
                HistogramSummary('nbp/%s/weight/%s' % (NAME, tag), param.data))

        return summaries

    def act_summaries(self) -> List[Summary]:
        return []

    def load_weights(self, savedir: str):
        self._nbv_actor.load_state_dict(
            torch.load(os.path.join(savedir, 'bc_nbv_actor.pt'),
                       map_location=torch.device('cpu')))
        self._nbp_actor.load_state_dict(
            torch.load(os.path.join(savedir, 'bc_nbp_actor.pt'),
                       map_location=torch.device('cpu')))

    def save_weights(self, savedir: str):
        torch.save(self._nbv_actor.state_dict(),
                   os.path.join(savedir, 'bc_nbv_actor.pt'))
        torch.save(self._nbp_actor.state_dict(),
                   os.path.join(savedir, 'bc_nbp_actor.pt'))
