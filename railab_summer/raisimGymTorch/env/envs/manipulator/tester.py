from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import manipulator
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymTorch.algo.ppo.module as ppo_module
import os
import math
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
cfg['environment']['num_envs'] = 1

env = VecEnv(manipulator.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs + 6
act_dim = env.num_acts

weight_path = args.weight
iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'

if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    print("Visualizing and evaluating the policy: ", weight_path)
    loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim, act_dim)
    loaded_graph.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()

    # max_steps = 1000000
    max_steps = 1000 ## 10 secs
    ran_pose = [0, 0, 0]
    ran_pose = list(map(float, input("Enter the end_effector pose (x, y, z): ").split()))
    ran_vel = [0, 0, 0]
    ran_vel = list(map(float, input("Enter the end_effector velocity (x, y, z): ").split()))
    
    # random_sampling
    rannum = 6
    input_sample = np.zeros([env.num_envs, rannum], dtype=np.float32)

    for i in range(env.num_envs):
        for j in range(3):
            input_sample[i][j] = ran_pose[j]
        for k in range(3):
            input_sample[i][3+k] = ran_vel[k]

    ball_err = []
    t = []
    err1 = 0
    err2 = 0
    for step in range(max_steps):
        time.sleep(0.01)
        obs = env.observe(False)
        err1 = (obs[0][46] - ran_pose[0])**2 + (obs[0][47] - ran_pose[1])**2 + (obs[0][48] - ran_pose[2])**2
        err2 = (obs[0][49] - ran_vel[0])**2 + (obs[0][50] - ran_vel[1])**2 + (obs[0][51] - ran_vel[2])**2 
        ball_err.append(np.sqrt((err1 + err2)/6))
        t.append(step)
        obs_new = np.concatenate([obs, input_sample], axis=1)
        print(obs_new)
        action_ll = loaded_graph.architecture(torch.from_numpy(obs_new).cpu())
        reward_ll, dones = env.stepran(action_ll.cpu().detach().numpy(), input_sample)
        reward_ll_sum = reward_ll_sum + reward_ll[0]
        if dones or step == max_steps - 1:
            print('----------------------------------------------------')
            print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_ll_sum / (step + 1 - start_step_id))))
            print('{:<40} {:>6}'.format("time elapsed [sec]: ", '{:6.4f}'.format((step + 1 - start_step_id) * 0.01)))
            print('----------------------------------------------------\n')
            start_step_id = step + 1
            reward_ll_sum = 0.0

    plt.plot(t, ball_err, 'r')
    plt.show()
    env.turn_off_visualization()
    env.reset()
    print("Finished at the maximum visualization steps")
