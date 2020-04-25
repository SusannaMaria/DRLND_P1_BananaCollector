#!/usr/bin/env python
# coding: utf-8

from unityagents import UnityEnvironment
import os.path
import numpy as np
import pandas as pd
from collections import deque
import torch
from dqn_agent import Agent
from classes.utils import helper
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from secondcounter import SecondCounter

env = UnityEnvironment(
    file_name="Banana_Windows_x86_64/Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

def env_show_info():
    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))
    # number of actions
    print('Number of actions:', action_size)
    # examine the state space
    print('States look like:', state)
    print('States have length:', state_size)    


def dqn_train(state_size: int,
              action_size: int,
              n_episodes: int = 2000,
              max_t: int = 400,
              eps_start: float = 1.0,
              eps_end: float = 0.001,
              eps_decay: float = 0.995,
              seed: int = 0):
    """Deep Q-Learning train function.

    Params
    ======
        state_size (int):   size of state space
        action_size (int):  action numbers    
        n_episodes (int):   maximum number of training episodes
        max_t (int):        maximum number of timesteps per episode
        eps_start (float):  starting value of epsilon, for epsilon-greedy action selection
        eps_end (float):    minimum value of epsilon
        eps_decay (float):  multiplicative factor (per episode) for decreasing epsilon
    """
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    # list containing scores from each episode
    scores = []                                                                 
    # last 100 scores
    scores_window = deque(maxlen=10)                                           
    # initialize epsilon
    eps = eps_start
    
    # create the counter instance
    count = SecondCounter()
    count.start()
    for i_episode in range(1, n_episodes+1):
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]                       
        # get the current state
        state = env_info.vector_observations[0]                                 
        # initialize score            
        score = 0                                                               
        for t in range(max_t):
            # select an action
            action = agent.act(np.array(state),eps)                             
            # cast action to int
            action = action.astype(int)
            # send the action to the environment
            env_info = env.step(action)[brain_name]                             
            # get the next state
            next_state = env_info.vector_observations[0]                        
            # get the reward
            reward = env_info.rewards[0]                                        
            # see if episode has finished
            done = env_info.local_done[0]                                       
            # send update step to agent
            agent.step(state,action,reward,next_state,done)                     
            # update the score
            score += reward                                                     
            # roll over the state to next time step
            state = next_state                                                  
            # exit loop if episode finished
            if done:                                                            
                break            
            
        # save score for 100 most recent scores
        scores_window.append(score)                                             
        # save score for episodes
        scores.append(score)                                                    
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        # print score every 10 episodes
        if i_episode % 10== 0:                                                  
            print('\rEpisode {}\tAverage Score of last 10 episodes: {:.2f}'.format(i_episode, np.mean(scores_window)), end = '')
        # save network every 100 episodes
        if i_episode % 100 == 0:                                                
             torch.save(agent.qnetwork_local.state_dict(), 'dqn_checkpoints/dqn_checkpoint_{:06d}.pth'.format(i_episode))       
    
    seconds = count.finish()
    print('\nTraining finished with {} episodes in {:.2f} seconds'.format(n_episodes, seconds))
    torch.save(agent.qnetwork_local.state_dict(), 'dqn_checkpoints/dqn_checkpoint_{:06d}.pth'.format(n_episodes))
    return scores


def dqn_test(agent):
    """Deep Q-Learning test function.

    Params
    ======
        agent:   DQN Agent with loaded network for testing
    """    
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]                            
    # get the current state
    state = env_info.vector_observations[0]                                      
    # initialize score
    score = 0                                                                     
    try:
        # run until done
        while (1):                                                               
            # select an action
            action = agent.act(np.array(state), 0)                               
            # cast action to int
            action = action.astype(int)
            # send the action to the environment
            env_info = env.step(action)[brain_name]
            # get the next state
            state = env_info.vector_observations[0] 
            # get the reward
            reward = env_info.rewards[0]   
            # see if episode has finished
            done = env_info.local_done[0]
            # update the score
            score += reward                                                      
            if done:
                return score
    except Exception as e:
        print("exception:", e)
        return score

def dqn_analytic_of_scores(state_size: int,
                           action_size: int,
                           checkpoint_min: int = 100,
                           checkpoint_max: int = 2100,
                           checkpoint_step: int = 100,
                           n_episode_run: int = 100,
                           goal_score: float = 13.0,
                           seed: int = 0):
    """Analytic for Deep Q-Learning agent.

    Params
    ======
        state_size (int):       size of state space
        action_size (int):      action numbers
        checkpoint_min (int):   start of checkpoint for analytics
        checkpoint_max (int):   end of checkpoint for analytics
        checkpoint_step (int):  steps between checkpoint        
        n_episode_run (int):    how many episodes for each checkpoints are executed
        goal_score (float):     what score has to be achieved (only used for ploting)
        seed (int):             seed for randomizing the agent
    """
    # initialize DQN Agent
    agent = Agent(state_size=state_size, action_size=action_size, seed=seed)    
    # range of checkpoints
    checkpoints = np.arange(checkpoint_min, 
                            checkpoint_max+checkpoint_step,
                            checkpoint_step)                                    
    # define X-Axis 
    X = np.arange(0, n_episode_run, 1)
    # create mesh grid for 3d plot
    X, Y = np.meshgrid(X, checkpoints)
    # create 2D-array for scores
    score_array = np.zeros(X.shape)                                             

    n_checkpoint_count = 0

    # create the counter instance
    count = SecondCounter()
    count.start()
    
    print("Starting analysing of dqn network")
    
    # iterate over checkpoints
    for n_checkpoint in checkpoints:                                            
        checkpoint_file = 'dqn_checkpoints/dqn_checkpoint_{:06d}.pth'.format(n_checkpoint)
        # is pretrained model for checkpoint available ?
        if not os.path.isfile(checkpoint_file):                             
            print("Checkpoint file not found: {}".format(checkpoint_file))
            continue
        # load pretrained model
        agent.qnetwork_local.load_state_dict(torch.load(checkpoint_file))       
        
        # prepare scores array for all episodes
        scores = np.arange(0, n_episode_run, 1)                                 
        for n_episode in range(0, n_episode_run):
            # test agent for one episode
            score = dqn_test(agent)
            # store result in array
            scores[n_episode] = score                                           
        
        # sort array in descending order
        scores = np.sort(scores)[::-1]                                          
        # maybe it's allowed to remove the best and worst 2 elements
        for n_episode in range(0, n_episode_run):
            score_array[n_checkpoint_count][n_episode] = scores[n_episode]      # Update 2D-array 

        n_checkpoint_count += 1
        print("Checkpoint: {} - mean score over {} episodes: {}".format(
            n_checkpoint, n_episode_run, np.mean(scores)))
    
    seconds = count.finish()
    print('Analytics finished with {} episodes in {:.2f} seconds'.format(n_episode_run, seconds))

    plot_3dsurface(X, Y, score_array)
    plot_minmax(checkpoints, score_array, checkpoint_min, checkpoint_max, goal_score)


def plot_3dsurface(X, Y, score_array):
    """Print 3D surface plot of DQN Agent analytics

    Params
    ======
        X (np.array):           Meshgrid X Axis
        Y (np.array):           Meshgrid X Axis
        score_array (np.array)  scores of all episodes of all checkpoints
    """     
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(X, Y, score_array, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-2, 22)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plot_minmax(checkpoints, 
                score_array, 
                checkpoint_min: int = 100, 
                checkpoint_max: int = 2100, 
                goal_score: float = 13):
    """Print min max plot of DQN Agent analytics

    Params
    ======
        checkpoints (np.array):           Array of checkpoints
        checkpoint_min (int):             Minimum Checkpoint for plot
        checkpoint_max (int):             Maximum Checkpoint for plot
        goal_score (float):               Goal of score for DQN Agent
    """   
    
    df = pd.DataFrame(columns=['checkpoint', 'min', 'max', 'mean'])

    row = 0
    for i in checkpoints:
        for j in score_array[row]:
            df.loc[row] = [i] + list([np.min(score_array[row]),
                                                       np.max(score_array[row]),
                                                       np.mean(score_array[row])])
        row += 1
    ax  = df.plot(x='checkpoint', y='mean', c='white')
    plt.fill_between(x='checkpoint',y1='min',y2='max', data=df)
    x_coordinates = [checkpoint_min, checkpoint_max]
    y_coordinates = [13, 13]
    plt.plot(x_coordinates, y_coordinates, color='red')    
    plt.show()


# In[24]:


# # Test of notebook
# # Train 200 episodes
# train_scores = dqn_train(state_size, action_size,200)


# # In[29]:


# # Analysis until 200 Checkpoint
# dqn_analytic_of_scores(state_size, action_size, 100, 200, 100, 10, 13, 333)


# # In[ ]:


# # Train of DQN for 2100 episodes
# train_scores = dqn_train(state_size, action_size,2100)


# # In[ ]:


# # Analyse until 2100 Checkpoint
# dqn_analytic_of_scores(state_size, action_size, 100, 2100, 100, 10, 13, 333)

