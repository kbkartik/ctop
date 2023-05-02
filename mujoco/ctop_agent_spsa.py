import random
from argparse import ArgumentParser
from collections import deque

import gym
from gym.wrappers import RescaleAction
import numpy as np
import torch
import pdb
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Callable

from top import TOP_Agent
from utils import MeanStdevFilter, Transition, make_gif, make_checkpoint
import datetime
import pandas as pd
from spsa import SPSA1

GYM_ENV = gym.wrappers.time_limit.TimeLimit

def train_agent_model_free(agent: TOP_Agent, env: GYM_ENV, params: Dict, spsa) -> None:
    
    update_timestep = params['update_every_n_steps']
    seed = params['seed']
    log_interval = 1000
    gif_interval = 1000000
    n_random_actions = params['n_random_actions']
    n_evals = params['n_evals']
    n_collect_steps = params['n_collect_steps']
    use_statefilter = params['obs_filter']
    save_model = params['save_model']
    filename = params['filename'] + '_' + params['fb_type']

    assert n_collect_steps > agent.batchsize, "We must initially collect as many steps as the batch size!"

    time_step = 0
    cumulative_timestep = 0
    n_updates = 0
    samples_number = 0
    episode_rewards = []
    train_episode_return = []
    eval_episode_return = []
    optimisms = []
    evaluations = []
    ep_num = 0

    if use_statefilter:
        state_filter = MeanStdevFilter(env.env.observation_space.shape[0])
    else:
        state_filter = None

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.np_random.seed(seed)

    max_steps = env.spec.max_episode_steps
    writer = SummaryWriter(log_dir='logs/' + filename)

    print('cTOP training begins....')

    prev_episode_reward = 0
    while samples_number < 1e6:
        time_step = 0
        episode_reward = 0
        state = env.reset()
        if state_filter:
            state_filter.update(state)
        done = False

        # sample an optimism setting for this episode
        # SPSA2
        #if ep_num == 0 or ep_num % 2 == 0:
        #    optimism_plus, optimism_minus = spsa.sample()
        #"""
        #Delayed updates
        if ep_num == 0:
            optimism = spsa.sample()
            optimisms.append(spsa.optimism)
            writer.add_scalar('Metrics/Optimism', spsa.optimism, ep_num)
        #"""

        while (not done):
            cumulative_timestep += 1
            time_step += 1
            samples_number += 1
            if samples_number < n_random_actions:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state, state_filter=state_filter)
            
            nextstate, reward, done, _ = env.step(action)
            # if we hit the time-limit, it's not a 'real' done; we don't want to assign low value to those states
            real_done = False if time_step == max_steps else done
            agent.replay_pool.push(Transition(state, action, reward, nextstate, real_done))
            state = nextstate
            if state_filter:
                state_filter.update(state)
            episode_reward += reward

            # SPSA2
            #if ep_num == 0 or ep_num % 2 == 0:
            #    optimism = optimism_plus
            #else:
            #    optimism = optimism_minus

            # update if it's time
            if cumulative_timestep % update_timestep == 0 and cumulative_timestep >= n_collect_steps:
                q1_loss, q2_loss, pi_loss, avg_wd, q1, q2 = agent.optimize(update_timestep, optimism, state_filter=state_filter)
                n_updates += 1
            # logging
            if cumulative_timestep % log_interval == 0 and cumulative_timestep >= n_collect_steps:
                writer.add_scalar('Loss/Q-func_1', q1_loss, n_updates)
                writer.add_scalar('Loss/Q-func_2', q2_loss, n_updates)
                writer.add_scalar('Loss/WD', avg_wd, n_updates)
                writer.add_scalar('Distributions/Mean_1', torch.mean(q1), n_updates)
                writer.add_scalar('Distributions/Median_1', torch.median(q1), n_updates)
                writer.add_scalar('Distributions/Mean_2', torch.mean(q2), n_updates)
                writer.add_scalar('Distributions/Median_2', torch.median(q2), n_updates)

                if pi_loss:
                    writer.add_scalar('Loss/policy', pi_loss, n_updates)
                running_reward = np.mean(episode_rewards)
                evaluations.append(running_reward)
                print('Timesteps: ', cumulative_timestep)
                eval_reward = evaluate_agent(env, agent, state_filter, n_starts=n_evals)
                eval_episode_return.append(eval_reward)
                writer.add_scalar('Metrics/Train_running_reward', running_reward, cumulative_timestep)
                writer.add_scalar('Metrics/Eval_return', eval_reward, cumulative_timestep)
                episode_rewards = []

        episode_rewards.append(episode_reward)
        writer.add_scalar('Metrics/Train_return', episode_reward, cumulative_timestep)
        train_episode_return.append(episode_reward)
        ep_num += 1

        # update bandit parameters
        feedback = episode_reward - prev_episode_reward
        spsa.append_fb(episode_reward, prev_episode_reward)
        writer.add_scalar('Metrics/Episode', ep_num, ep_num)
        
        #Delayed updates
        #if ((ep_num % 50) == 0) or (samples_number == int(1e6)):
        spsa.update()
        optimisms.append(spsa.optimism)
        writer.add_scalar('Metrics/Optimism', spsa.optimism, cumulative_timestep)
        optimism = spsa.sample()
        
        prev_episode_reward = episode_reward

    s1 = pd.Series(eval_episode_return, name='Eval_ep_return')
    s2 = pd.Series(train_episode_return, name='Train_ep_return')
    s3 = pd.Series(optimisms, name='Optimism')
    s4 = pd.Series(evaluations, name='Train_running_reward')
    valdict = pd.concat([s1, s2, s3, s4], axis=1)
    df = pd.DataFrame(valdict)
    df.to_csv('./results/'+filename+'.csv', index=False)

def evaluate_agent(
    env: GYM_ENV,
    agent: TOP_Agent,
    state_filter: Callable,
    n_starts: int = 1) -> float:
    
    reward_sum = 0
    for _ in range(n_starts):
        done = False
        state = env.reset()
        while (not done):
            action = agent.get_action(state, state_filter=state_filter, deterministic=True)
            nextstate, reward, done, _ = env.step(action)
            reward_sum += reward
            state = nextstate

    print("---------------------------------------")
    print(f"Eval Return: {reward_sum / n_starts:.3f} over {n_starts} episodes")
    print("---------------------------------------")
    return reward_sum / n_starts


def main():
    
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--use_obs_filter', dest='obs_filter', action='store_true')
    parser.add_argument('--update_every_n_steps', type=int, default=1)
    parser.add_argument('--n_random_actions', type=int, default=25000)
    parser.add_argument('--n_collect_steps', type=int, default=1000)
    parser.add_argument('--n_evals', type=int, default=10)
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--n_quantiles', type=int, default=50)
    parser.add_argument('--bandit_lr', type=float, default=0.1)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--fb_type', type=str)
    parser.set_defaults(obs_filter=False)
    parser.set_defaults(save_model=False)

    args = parser.parse_args()
    params = vars(args)

    seed = params['seed']
    env = gym.make(params['env'])
    env = RescaleAction(env, -1, 1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    start_time = datetime.datetime.now()
    # initialize agent
    agent = TOP_Agent(seed, state_dim, action_dim, \
        n_quantiles=params['n_quantiles'], bandit_lr=params['bandit_lr'])
    spsa = SPSA1(seed, params['bandit_lr'], params['fb_type'])

    # train agent 
    train_agent_model_free(agent=agent, env=env, params=params, spsa=spsa)
    print('Total training time: ', datetime.datetime.now() - start_time)


if __name__ == '__main__':
    main()
