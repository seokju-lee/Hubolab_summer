from ctypes import sizeof
from random import random
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin.manipulator import RaisimGymEnv
from raisimGymTorch.env.bin.manipulator import NormalSampler
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse
import random

# task specification
task_name = "manipulation"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
env = VecEnv(RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)))

# shortcuts
ob_dim = env.num_obs + 6
act_dim = env.num_acts
num_threads = cfg['environment']['num_threads']

# random_sampling
rannum = 6

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

avg_rewards = []

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                           env.num_envs,
                                                                           1.0,
                                                                           NormalSampler(act_dim),
                                                                           cfg['seed']),
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])
tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.996,
              lam=0.95,
              num_mini_batches=4,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              )

if mode == 'retrain':
    load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)

for update in range(100010):
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.

    if update % 25 == 0:
        random_sample = np.zeros([env.num_envs, rannum], dtype=np.float32)
        for i in range(env.num_envs):
            random_sample[i][0] = round(random.uniform(0.0, 1.0), 1)
            random_sample[i][1] = round(random.uniform(-1.0, 1.0), 1)
            random_sample[i][2] = round(random.uniform(0.5, 2.5), 1)
            random_sample[i][3] = round(random.uniform(0.0, 2.0), 1)
            random_sample[i][4] = round(random.uniform(-2.0, 2.0), 1)
            random_sample[i][5] = round(random.uniform(-2.0, 2.0), 1)
            
    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
        # we create another graph just to demonstrate the save/load method
        loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
        loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

        env.turn_on_visualization()
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        for step in range(n_steps*2):
            with torch.no_grad():
                frame_start = time.time()
                obs = env.observe(False)
                obs_new = np.concatenate([obs, random_sample], axis=1)
                action_ll = loaded_graph.architecture(torch.from_numpy(obs_new).cpu())
                reward_ll, dones = env.stepran(action_ll.cpu().detach().numpy(), random_sample)
                frame_end = time.time()
                wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
                if wait_time > 0.:
                    time.sleep(wait_time)

        env.stop_video_recording()
        env.turn_off_visualization()

        env.reset()
        env.save_scaling(saver.data_dir, str(update))

    # actual training
    for step in range(n_steps):
        obs = env.observe()
        obs_new = np.concatenate([obs, random_sample], axis=1)
        action = ppo.act(obs_new)
        reward, dones = env.stepran(action, random_sample)
        ppo.step(value_obs=obs_new, rews=reward, dones=dones)
        done_sum = done_sum + np.sum(dones)
        reward_ll_sum = reward_ll_sum + np.sum(reward)

    # take st step to get value obs
    obs = env.observe()
    obs_new = np.concatenate([obs, random_sample], axis=1)
    ppo.update(actor_obs=obs_new, value_obs=obs_new, log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)
    
    x = 0
    y = 0
    z = 0
    vx = 0
    vy = 0
    vz = 0

    for i in range(env.num_envs):
        if obs_new[i][52] == 0:
            x = 0
        else:
            x += np.abs(obs_new[i][52] - obs[i][46])/obs_new[i][52]
        if obs_new[i][53] == 0:
            x = 0
        else:
            y += np.abs(obs_new[i][53] - obs[i][47])/obs_new[i][53]
        if obs_new[i][54] == 0:
            x = 0
        else:
            z += np.abs(obs_new[i][54] - obs[i][48])/obs_new[i][54]
        if obs_new[i][55] == 0:
            x = 0
        else:
            vx += np.abs(obs_new[i][55] - obs[i][49])/obs_new[i][55]
        if obs_new[i][56] == 0:
            x = 0
        else:
            vy += np.abs(obs_new[i][56] - obs[i][50])/obs_new[i][56]
        if obs_new[i][57] == 0:
            x = 0
        else:
            vz += np.abs(obs_new[i][57] - obs[i][51])/obs_new[i][57]

    performance = (x + y + z + vx + vy + vz)/6
    ppo.rewt_log(actor_obs=obs_new, value_obs=obs_new, log_this_iteration=update % 10, rews=performance, update=update)

    actor.update()
    actor.distribution.enforce_minimum_std((torch.ones(18)*0.2).to(device))

    # curriculum update. Implement it in Environment.hpp
    env.curriculum_callback()

    end = time.time()

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('----------------------------------------------------\n')
