from turtle import update
import yaml
import argparse
import sys
import random
import torch
import wandb
import os
import numpy as np

from datetime import datetime
import torch.nn.functional as F

from omegaconf import OmegaConf
from omegaconf import DictConfig

from utils import generate_expt_id
from environments import make_environment

sys.path.append("./custom")
import custom.utils as utils

_soda_cfg_dict: dict = {
    "algorithm": "soda",
    "train": False,
    "eval": True,
    "eval_freq": 1,
    "pre_trained": True,
    "pre_trained_dir": "/home/guest-cch/apple-rl-core/param/soda , _crop_overlay , walker , None , static , 2022-08-15T08:09",
    # "batch_size": 256,
    "batch_size": 128, # https://bskyvision.com/entry/python-MemoryError-Unable-to-allocate-array-with-shape-and-data-type-%ED%95%B4%EA%B2%B0%EB%B2%95
    "tau": 0.005,
    "train_steps": 500000,
    "discount": 0.99,
    "init_steps": 1000,
    #"init_steps": 250,
    "hidden_dim": 1024,
    "actor": {
        # "lr": 1e-3,
        "lr": 5e-4,
        "beta": 0.9,
        "log_std_min": -10,
        "log_std_max": 2,
        "update_freq": 2,
    },
    "critic": {
        # "lr": 1e-3,
        "lr": 5e-4,
        "beta": 0.9,
        "tau": 0.01,
        "target_update_freq": 2,
    },
    "architecture": {
        "num_shared_layers": 11,
        "num_head_layers": 0,
        "num_filters": 32,
        "projection_dim": 100,
        "encoder_tau": 0.05,
    },
    "entropy_maximization": {
        "init_temperature": 0.1,
        # "alpha_lr": 1e-4,
        "alpha_lr": 5e-5,
        "alpha_beta": 0.5,
    },
    "aux_task": {
        # "aux_lr": 1e-3,
        "aux_lr": 5e-4,
        "aux_beta": 0.9,
        "aux_update_freq": 2
    },
    "anchor_augmentation": "random_crop",
    "augmentation":{
        "aug_num": 2,
        "first_aug": "random_crop",
        "second_aug": "random_overlay",
        "third_aug": None,
    }, # random_crop, random_overlay, random_conv, random_shift
}

_soda_cfg = OmegaConf.create(_soda_cfg_dict)

wandb_log = True


class Trainer(object):
    def __init__(self, config, agent_cfg: DictConfig):
        self.config = config
        self.agent_cfg = agent_cfg

        self.num_envs = self.config['num_envs']
        self.num_val_envs = self.config['num_val_envs']
        seed = self.config['seed']
        self.train_env_containers = [make_environment(self.config['env'], train=True, seed=seed+i) for i in range(self.num_envs)]
        seed += self.num_envs
        self.val_env_containers = [make_environment(self.config['env'], train=False, seed=seed+i) for i in range(self.num_val_envs)]
        self.env = self.train_env_containers[0]
        self.eval_env = self.val_env_containers[0]
        self.action_repeat = self.env.get_action_repeat()
        action_dims = self.env.get_action_dims()
        self.obs_channels, self.obs_height, self.obs_width = self.env.get_obs_chw()
        self.obs_other_dims = self.env.get_obs_other_dims()

        self.best_after_interval = 100
        self.best_update_step = 0

        self.train_flag = self.agent_cfg.train
        self.eval_flag = self.agent_cfg.eval
        self.eval_freq = self.agent_cfg.eval_freq

        obs_shape=(self.obs_channels, self.obs_height, self.obs_width)
        action_shape=(action_dims, )
        print(f"action_shape: {action_shape}")
        capacity=self.agent_cfg.train_steps
        batch_size=self.agent_cfg.batch_size

        self.replay_buffer = utils.ReplayBuffer(
            obs_shape=obs_shape,
            action_shape=action_shape,
            capacity=capacity,
            batch_size=batch_size
        )
        cropped_obs_shape = (3*config['env']['num_frames_to_stack'], 
                             config['env']['image_crop_size'], 
                             config['env']['image_crop_size'])
        print('Observations:', obs_shape)
        print('Cropped observations:', cropped_obs_shape)
        self.agent = utils.make_agent(
            obs_shape=cropped_obs_shape,
            action_shape=action_shape,
            cfg=self.agent_cfg
        )
        if self.agent_cfg.pre_trained:
            self.agent = torch.load(f"{self.agent_cfg.pre_trained_dir}/model.pth")
            print("load completed")

    def train(self, save_name):
        max_step_per_episode = int(self.config['episode_steps'] / self.action_repeat)
        action_low, action_high = self.env.get_action_limits()
        action_dims = self.env.get_action_dims()
        start_step, episode, episode_reward, done = 0, 0, 0, True
        update_step = 0
        best_epi = 0
        
        for step in range(start_step, self.agent_cfg.train_steps+1):
            if self.eval_flag and step % self.eval_freq == 0:
                eval_obs = self.eval_env.reset()
                eval_obs = eval_obs['image']
                eval_episode_reward = 0
                for eval_episode_step in range(max_step_per_episode):
                    with utils.eval_mode(self.agent):
                        eval_action = self.agent.select_action(eval_obs)
                    eval_obs, eval_reward, eval_done, _ = self.eval_env.step(eval_action)
                    eval_obs = eval_obs['image']
                    eval_episode_reward += eval_reward
                if wandb_log:
                    wandb.log({"eval_episode_reward": eval_episode_reward}, 
                              step=step)
            
            if self.train_flag:
                if done:
                    obs = self.env.reset()
                    obs = obs['image']
                    done = False
                    episode_reward = 0
                    episode_step = 0
                    episode += 1

                # Sample action for data collection
                if step < self.agent_cfg.init_steps and not self.agent_cfg.pre_trained:
                    action = np.random.uniform(action_low, action_high, action_dims)
                else:
                    with utils.eval_mode(self.agent):
                        action = self.agent.sample_action(obs)

                # Run training update
                if step >= self.agent_cfg.init_steps:
                    num_updates = self.agent_cfg.init_steps if step == self.agent_cfg.init_steps else 1
                    for _ in range(num_updates):
                        self.agent.update(self.replay_buffer, step)

                # Take step
                next_obs, reward, done, _ = self.env.step(action)
                done_bool = 0 if episode_step + 1 == self.config['episode_steps'] else float(done)
                self.replay_buffer.add(obs, action, reward, next_obs['image'], done_bool)
                episode_reward += reward
                obs = next_obs['image']

                if episode_step >= max_step_per_episode - 1:
                    done = True
                    if wandb_log:
                        wandb.log({
                            "episode_reward": episode_reward
                        }, step=update_step)
                    update_step += 1
                    if best_epi <= episode_reward:
                        best_epi = episode_reward
                        torch.save(self.agent, f'param/{save_name}/model.pth')
                        print(f"best record updated: {episode_reward}")

                    if update_step - self.best_update_step >= self.best_after_interval:
                        self.best_update_step = update_step
                        torch.save(self.agent, f'param/{save_name}/best_model_after_{self.best_after_interval}_step.pth')
                
                episode_step += 1
            
        torch.save(self.agent, f'param/{save_name}/final_model.pth')

def argument_parser(argument):
    """ Argument parser """
    parser = argparse.ArgumentParser(description='Binder Network.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('-c', '--config', default='', type=str, help='Training config')
    parser.add_argument('--debug', action='store_true', help='Debug mode. Disable logging.')
    args = parser.parse_args(argument)
    return args

def main():
    args = argument_parser(None)
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print('Running on GPU {}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device('cpu')
        print('Running on CPU')

    try:
       with open(args.config) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error opening specified config yaml at: {}. "
              "Please check filepath and try again.".format(args.config))
        sys.exit(1)

    config = config['parameters']
    config['expt_id'] = generate_expt_id()
    
    seed = config['seed']
    '''random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)'''
    utils.set_seed_everywhere(seed)

    # select algorithm & GPU
    _agent_cfg_dict = _soda_cfg_dict
    _agent_cfg = _soda_cfg
    trainer = Trainer(config, _agent_cfg)

    # select domain & difficulty & dynamic
    domain = config['env']['domain']
    if config['env']['difficulty'] == "None":
        difficulty = "None"
    else:
        distraction_list = [config['env']['allow_background_distraction'],
                            config['env']['allow_camera_distraction'],
                            config['env']['allow_color_distraction']]
        distraction_method_list = ['bg', 'cam', 'color']
        difficulty = f"{config['env']['difficulty']}"
        false_method_num = 0

        for idx, distraction_method in enumerate(distraction_list):
            if distraction_method is False:
                difficulty += f' + no {distraction_method_list[idx]}'
                false_method_num += 1
                continue
            true_method_idx = idx
        if false_method_num == 2:
            difficulty = f"{config['env']['difficulty']} + only {distraction_method_list[true_method_idx]}"
    dynamic = "dynamic" if config['env']['dynamic'] is True else "static"


    # wandb logging
    algorithm = _agent_cfg.algorithm
    if _agent_cfg.pre_trained:
        algorithm = "pre_trained " + algorithm
    aug_name = _agent_cfg.augmentation.first_aug[6:]
    if _agent_cfg.augmentation.second_aug is not None:
        aug_name += _agent_cfg.augmentation.second_aug[6:]
        if _agent_cfg.augmentation.third_aug is not None:
            aug_name += _agent_cfg.augmentation.third_aug[6:]
    if _agent_cfg.anchor_augmentation != "random_crop":
        aug_name += f' + anchor{_agent_cfg.anchor_augmentation[6:]}'
    run_name = f"{algorithm} / {aug_name} / {domain} / {difficulty} / {dynamic} / {datetime.now().isoformat(timespec='minutes')}"
    project_name = "distracting_cs_augmentation"
    run_tags = [config['env']['domain']]

    if wandb_log:
        wandb.init(
            project=project_name,
            name=run_name,
            tags=run_tags,
            config=_agent_cfg_dict
        )

    run_name = run_name.replace('/', ',')
    os.mkdir(f'param/{run_name}')

    print("start training")
    trainer.train(save_name=run_name)

if __name__ == '__main__':
    main()