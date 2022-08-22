import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import custom.utils as utils
from omegaconf import DictConfig

import modules as m
from sac import SAC
import augmentations


class SODA(SAC):
	def __init__(self, obs_shape, action_shape, cfg: DictConfig):
		super().__init__(obs_shape, action_shape, cfg)
		self.cfg = cfg
		self.aux_update_freq = self.cfg.aux_task.aux_update_freq
		self.soda_batch_size = self.cfg.batch_size
		self.soda_tau = self.cfg.tau

		self.aug_dict = {
			"random_crop": augmentations.random_crop,
			"random_overlay": augmentations.random_overlay,
			"random_conv": augmentations.random_conv,
			"random_shift": augmentations.random_shift,
		}
		self.aug_list = []
		assert self.cfg.augmentation.aug_num > 0 and self.cfg.augmentation.aug_num < 4, "aug_num has to be integer between 1 to 3"
		self.aug_list.append(self.aug_dict[self.cfg.augmentation.first_aug])
		if self.cfg.augmentation.second_aug != None:
			self.aug_list.append(self.aug_dict[self.cfg.augmentation.second_aug])
			if self.cfg.augmentation.third_aug != None:
				self.aug_list.append(self.aug_dict[self.cfg.augmentation.third_aug])
		assert len(self.aug_list) == self.cfg.augmentation.aug_num, "the number of augmentation methods is different with aug_num"

		shared_cnn = self.critic.encoder.shared_cnn
		aux_cnn = self.critic.encoder.head_cnn
		soda_encoder = m.Encoder(
			shared_cnn,
			aux_cnn,
			m.SODAMLP(aux_cnn.out_shape[0], self.cfg.architecture.projection_dim, self.cfg.architecture.projection_dim)
		)

		self.predictor = m.SODAPredictor(soda_encoder, self.cfg.architecture.projection_dim).cuda()
		self.predictor_target = deepcopy(self.predictor)

		self.soda_optimizer = torch.optim.Adam(
			self.predictor.parameters(), lr=self.cfg.aux_task.aux_lr, betas=(self.cfg.aux_task.aux_beta, 0.999)
		)
		self.train()

	def train(self, training=True):
		super().train(training)
		if hasattr(self, 'soda_predictor'):
			self.soda_predictor.train(training)

	def compute_soda_loss(self, x0, x1):
		h0 = self.predictor(x0)
		with torch.no_grad():
			h1 = self.predictor_target.encoder(x1)
		h0 = F.normalize(h0, p=2, dim=1)
		h1 = F.normalize(h1, p=2, dim=1)

		return F.mse_loss(h0, h1)

	def update_soda(self, replay_buffer, step=None):
		x = replay_buffer.sample_soda(self.soda_batch_size)
		assert x.size(-1) == 100

		aug_x = x.clone()

		'''x = augmentations.random_crop(x)
		aug_x = augmentations.random_crop(aug_x)
		aug_x = augmentations.random_overlay(aug_x)'''
		x = self.aug_dict[self.cfg.anchor_augmentation](x)
		for aug_idx in range(self.cfg.augmentation.aug_num):
			aug_x = self.aug_list[aug_idx](aug_x)

		soda_loss = self.compute_soda_loss(aug_x, x)

		soda_loss_value = soda_loss.clone().detach()
		
		if self.cfg.aux_thred:
			if soda_loss_value >= self.cfg.aux_thred_value:
				self.soda_optimizer.zero_grad()
				soda_loss.backward()
				self.soda_optimizer.step()

				utils.soft_update_params(
					self.predictor, self.predictor_target,
					self.soda_tau
				)
			else: return soda_loss_value
		else:
			self.soda_optimizer.zero_grad()
			soda_loss.backward()
			self.soda_optimizer.step()

			utils.soft_update_params(
				self.predictor, self.predictor_target,
				self.soda_tau
			)

		return soda_loss_value

	def update(self, replay_buffer, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample()

		self.update_critic(obs, action, reward, next_obs, not_done, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()

		if step % self.aux_update_freq == 0:
			soda_loss_value = self.update_soda(replay_buffer, step)
			return soda_loss_value

		return None

	def advantage_diff(self, obs):
		with torch.no_grad():
			obs = torch.unsqueeze(obs, dim=0)
			_, action, log_pi, _ = self.actor(obs)
			aug_obs = obs.clone()
			for aug_idx in range(self.cfg.augmentation.aug_num):
				aug_obs = self.aug_list[aug_idx](aug_obs)

			Q1, Q2 = self.critic(obs, action)

			aug_obs = torch.mean(aug_obs, 0, True)
			aug_Q1, aug_Q2 = self.critic(aug_obs, action)

			_, _, aug_log_pi, _ = self.actor(aug_obs)

			V = torch.min(Q1, Q2) - self.alpha.detach() * log_pi
			aug_V = torch.min(aug_Q1, aug_Q2) - self.alpha.detach() * aug_log_pi

			return torch.abs((torch.min(Q1, Q2) - V) - (torch.min(aug_Q1, aug_Q2) - aug_V))