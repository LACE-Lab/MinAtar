################################################################################################################
# Authors:                                                                                                     #
# Zoe
#
# This file contains DQN alg training with Cart Pole.
# Incorpeted MVE, NN for Env model
# The Env model predicts only the state
# The reward and termination rule are hardcoded
# Env Model is with range input, range output
# This one is using model rollout to train env model
################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
import time
import gymnasium as gym
from scipy.special import softmax

import random, numpy, argparse, logging, os

import numpy as np

from collections import namedtuple
from customAcrobot import CustomAcrobot
from scipy.stats import pearsonr

################################################################################################################
# Constants
################################################################################################################
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000
TARGET_NETWORK_UPDATE_FREQ = 500
TRAINING_FREQ = 1
NUM_FRAMES = 100000
FIRST_N_FRAMES = 100
REPLAY_START_SIZE = 64
END_EPSILON = 0.1
STEP_SIZE = 0.003
WEIGHT_DECAY = 0.0001
GRAD_MOMENTUM = 0.95
SQUARED_GRAD_MOMENTUM = 0.95
MIN_SQUARED_GRAD = 0.01
GAMMA = 0.99
EPSILON = 1
H = 1 # rollout constant
ENV_H = 5  # rollout constant for env training
SEED = 42
ENV_HIDDEN_SIZE = 128
QUANTILES = [0.01, 0.99]  # The target quantiles
TEMPERATURE = 1
DECAY = 1

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

################################################################################################################
# class QNetwork
#
# One hidden 2D conv with variable number of input channels.  We use 16 filters, a quarter of the original DQN
# paper of 64.  One hidden fully connected linear layer with a quarter of the original DQN paper of 512
# rectified units.  Finally, the output layer is a fully connected linear layer with a single output for each
# valid action.
#
################################################################################################################
class QNetwork(pl.LightningModule, nn.Module):
    def __init__(self, in_channels, num_actions, hidden_dim = 100):

        super().__init__()
        super(QNetwork, self).__init__()
        self.save_hyperparameters()
        self.dropout = nn.Dropout(p=0.2)

        # self.fc_hidden = nn.Linear(in_features=in_channels, out_features=hidden_dim)
        
        self.fc_hidden = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=hidden_dim),
            nn.ReLU()
        )

        # Output layer:
        self.output =nn.Linear(int(hidden_dim), num_actions)

    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x):
        # Rectified output from the final hidden layer
        x = self.fc_hidden(x)

        # Returns the output from the fully-connected linear layer
        return self.output(x)

class QuantileEnvModel(nn.Module):
    def __init__(self, state_size, action_size, quantiles, hidden_size=ENV_HIDDEN_SIZE):
        super(QuantileEnvModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        self.fc1 = nn.Linear((state_size * 2) + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, len(quantiles) * state_size)
        self.fc3 = nn.Linear(hidden_size, state_size)

    def load_state(self, max_state, min_state):
        self.max_state = max_state.clone().detach()
        self.min_state = min_state.clone().detach()

    def save_state(self):
        return self.max_state.clone().detach(), self.min_state.clone().detach()

    def forward(self, state_action_pair):
        x = self.fc1(state_action_pair)
        x = F.relu(x)
        quantile_outputs = self.fc2(x).view(x.shape[0], self.state_size, self.num_quantiles).squeeze()
        mean_outputs = self.fc3(x).squeeze(0)
        return quantile_outputs, mean_outputs

    def step(self, action):
        one_hot_action = torch.eye(self.action_size)[action.squeeze().long()]
        range_state = torch.cat((self.min_state, self.max_state), dim=-1).squeeze()
        state_action_pair = torch.cat((range_state, one_hot_action), dim=-1).unsqueeze(0)

        quantile_outputs, mean_outputs = self.forward(state_action_pair)

        predicted_next_state_means = mean_outputs
        predicted_next_state_min = quantile_outputs[:, 0]
        predicted_next_state_max = quantile_outputs[:, -1]
        
        return predicted_next_state_means, predicted_next_state_min, predicted_next_state_max


###########################################################################################################
# class replay_buffer
#
# A cyclic buffer of a fixed size containing the last N number of recent transitions.  A transition is a
# tuple of state, next_state, action, reward, is_terminal.  The boolean is_terminal is used to indicate
# whether if the next state is a terminal state or not.
#
###########################################################################################################
transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')
class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, *args):
        # Append when the buffer is not full but overwrite when the buffer is full
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
class rollout_replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, sequence):
        # sequence is a list of transitions
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(sequence)
        else:
            self.buffer[self.location] = sequence
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

################################################################################################################
# train
#
# This is where learning happens. More specifically, this function learns the weights of the policy network
# using huber loss.
#
# Inputs:
#   sample: a batch of size 1 or 32 transitions
#   policy_net: an instance of QNetwork
#   target_net: an instance of QNetwork
#   optimizer: centered RMSProp
#
################################################################################################################
def quantile_loss(preds, targets, quantiles):
    assert not targets.requires_grad
    assert preds.size(0) == targets.size(0)
    losses = []
    for i, q in enumerate(quantiles):
        errors = targets - preds[:, i]
        losses.append(torch.max((q-1)*errors, q*errors).unsqueeze(1))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss

rangeTransition = namedtuple('rangeTransition', 'true_state, state_max, state_min, true_next_state, next_state_max, next_state_min, next_state_mean, action, reward, is_terminal')

def process_batch(batch, env_model):
    new_batch = []
    for rollout in batch:
        state_max = rollout[0].state
        state_min = rollout[0].state
        
        rollout_transitions = []

        for transition in rollout:
            env_model.load_state(max_state=state_max, min_state=state_min)
            predicted_next_state_mean, predicted_next_state_min, predicted_next_state_max = env_model.step(transition.action)

            new_transition = rangeTransition(transition.state, state_max, state_min, 
                                             transition.next_state, predicted_next_state_max, predicted_next_state_min, predicted_next_state_mean, 
                                             transition.action, transition.reward, transition.is_terminal)

            rollout_transitions.append(new_transition)
            state_max, state_min = predicted_next_state_max, predicted_next_state_min
            
        random_transition = random.choice(rollout_transitions)
        new_batch.append(random_transition)

    return new_batch

def train_env_model(sample, env_model, optimizer, device, scheduler=None, clip_grad=None, weight_decay=0):
    # Process the batch
    processed_sample = process_batch(sample, env_model)
    
    # Unpack the processed batch
    batch_samples = rangeTransition(*zip(*processed_sample))

    states_max = torch.vstack((batch_samples.state_max)).to(device)
    states_min = torch.vstack((batch_samples.state_min)).to(device)
    
    actions = torch.cat(batch_samples.action, 0)
    actions = actions.reshape(BATCH_SIZE, 1).type(torch.int64)

    next_states_max = torch.vstack((batch_samples.next_state_max)).to(device)
    next_states_min = torch.vstack((batch_samples.next_state_min)).to(device)
    predicted_next_states_means = torch.vstack((batch_samples.next_state_mean)).to(device)
    true_next_states = torch.vstack((batch_samples.true_next_state)).to(device)
    
    predicted_next_states_quantiles = torch.zeros((BATCH_SIZE, env_model.state_size, len(QUANTILES)))
    for i in range(env_model.state_size):
        predicted_next_states_quantiles[:, i, 0] = next_states_min[:, i]
        predicted_next_states_quantiles[:, i, 1] = next_states_max[:, i]

    loss = 0.0
    quantile = 0.0
    for i in range(env_model.state_size):
        quantile_single = quantile_loss(predicted_next_states_quantiles[:, i, :], true_next_states[:, i].detach(), QUANTILES)
        quantile += quantile_single
        loss += quantile_single
    
    mean = F.mse_loss(predicted_next_states_means, true_next_states)
    loss += mean

    l2_reg = torch.tensor(0., device=device)
    for param in env_model.parameters():
        l2_reg += torch.norm(param, p=2)
    loss += weight_decay * l2_reg

    optimizer.zero_grad()
    loss.backward()

    if clip_grad is not None:
        torch.nn.utils.clip_grad_norm_(env_model.parameters(), clip_grad)

    optimizer.step()

    return quantile, mean, loss
    
def choose_action(t, replay_start_size, state, policy_net, n_actions):
    # print(state)
    x = state.to(device).clone().detach().unsqueeze(0)
    epsilon = END_EPSILON if t - replay_start_size >= FIRST_N_FRAMES \
        else ((END_EPSILON - EPSILON) / FIRST_N_FRAMES) * (t - replay_start_size) + EPSILON
    
    # epsilon-greedy
    if np.random.uniform() < epsilon: # random
        action = torch.tensor([random.randint(0, n_actions-1)]).to(device)

    else: # greedy
        actions_value = policy_net(x) # score of actions
        action = torch.max(actions_value, 1)[1].cpu().data.numpy() # pick the highest one
        action = torch.tensor(action).to(device)

    return action

def choose_greedy_action(state, policy_net):
    # print(state)
    x = state.to(device).clone().detach().unsqueeze(0)
    
    actions_value = policy_net(x) # score of actions
    action = torch.max(actions_value, 1)[1].data.numpy() # pick the highest one
    action = torch.tensor(action).to(device)

    return action

def extend_list(lst, n, elem):
    return lst + [elem]*(n - len(lst))

def softmax_with_temperature(x, temperature=1):
    e_x = np.exp((np.array(x) - np.max(x)) / temperature)
    return e_x / e_x.sum()

def trainWithRollout(sample, policy_net, target_net, optimizer, H, env_model, temp, decay, errors_per_step, overall_errors, weights_per_step, effective_planning_horizons, td_errors_diff, td_errors_direction_diff, full_target_errors, uncertainty_measures):
    # unzip the batch samples and turn components into tensors
    env = CustomAcrobot()
    env.reset()
    
    full_env = CustomAcrobot()
    full_env.reset()

    batch_samples = transition(*zip(*sample))

    states = torch.vstack((batch_samples.state)).to(device)
    next_states = torch.vstack((batch_samples.next_state)).to(device)

    actions = torch.cat(batch_samples.action, 0)
    actions = actions.reshape(BATCH_SIZE, 1).type(torch.int64)

    rewards = torch.tensor(batch_samples.reward).to(device).reshape(BATCH_SIZE, 1)
    is_terminal = torch.tensor(batch_samples.is_terminal).to(device).reshape(BATCH_SIZE, 1)

    Q_s_a = policy_net(states).gather(1, actions)

    if H == 1:
        none_terminal_next_state_index = torch.tensor([i for i, is_term in enumerate(is_terminal) if is_term == 0], dtype=torch.int64, device=device)
        none_terminal_next_states = next_states.index_select(0, none_terminal_next_state_index)
        Q_s_prime_a_prime = torch.zeros(BATCH_SIZE, 1, device=device)
        if len(none_terminal_next_states) != 0:
            Q_s_prime_a_prime[none_terminal_next_state_index] = target_net(none_terminal_next_states).detach().max(1)[0].unsqueeze(1)
        target = rewards + GAMMA * Q_s_prime_a_prime
        
    else:
        target = torch.empty((0))

        for i in range(BATCH_SIZE):
            initial_state = torch.tensor(batch_samples.state[i], dtype=torch.float32).to(device)
            env.set_state_from_observation(initial_state)
            env_model.load_state(initial_state, initial_state)
            
            state = states[i]
            next_state = next_states[i]
            done = is_terminal[i]
            
            full_reward_list = torch.zeros(H).to(device)
            full_value_list = torch.zeros(H).to(device)
            full_reward_list[0] = rewards[i]
            full_value_list[0] = 0 if done else target_net(next_state).max(0)[0].item()
            
            reward_list = torch.zeros(H).to(device)
            value_list = torch.zeros(H).to(device)
            reward_list[0] = rewards[i]
            value_list[0] = 0 if done else target_net(next_state).detach().max(0)[0].item()
            
            next_state = torch.tensor(batch_samples.next_state[i], dtype=torch.float32).to(device)
            env_model.load_state(next_state, next_state)
            env.set_state_from_observation(batch_samples.next_state[i].numpy())
            
            uncertainty_sample = [0]
            error_sample = [0]
            full_next_state = next_state
            full_done = done

            for h in range(1, H):
                if not done:
                    action = choose_greedy_action(state, policy_net)
                    full_action = choose_greedy_action(full_next_state, policy_net)
                    predicted_next_state_means, predicted_next_state_min, predicted_next_state_max = env_model.step(action)
                    uncertainty = torch.abs(predicted_next_state_max - predicted_next_state_min).sum().item()

                    uncertainty_sample.append(uncertainty)

                    cosTheta1 = next_state[0]
                    sinTheta1 = next_state[1]
                    cosTheta2 = next_state[2]
                    sinTheta2 = next_state[3]
                    dTheta1 = next_state[4]
                    dTheta2 = next_state[5]

                    done = False
                    reward = -1
                    if -cosTheta1 - (cosTheta2 * cosTheta1 - sinTheta2 * sinTheta1) > 1.0:
                        done = True
                        reward = 0
                        
                    # One step state
                    real_next_state, _, _, _, _ = env.step(action.item())
                    real_next_state = torch.tensor(real_next_state)
                    noise = (torch.randn(1) * 50).to(device)
                    real_next_state = torch.cat((real_next_state, noise), 0)
                    
                    error = torch.abs(real_next_state - predicted_next_state_means).sum().item()
                    error_sample.append(error)
                    
                    env.set_state_from_observation(predicted_next_state_means)
                    predicted_next_state_means = torch.Tensor(predicted_next_state_means).to(device)
                    env_model.load_state(predicted_next_state_max, predicted_next_state_min)

                    value_list[h] = 0 if done else target_net(predicted_next_state_means).max(0)[0].item()
                    reward_list[h] = reward
                    state = predicted_next_state_means
                    
                    if not full_done:
                        full_next_state, full_reward, full_terminated, full_truncated, _ = full_env.step(full_action.item())
                        full_done = full_terminated or full_truncated
                        full_next_state = torch.tensor(full_next_state)
                        noise = (torch.randn(1) * 50).to(device)
                        full_next_state = torch.cat((full_next_state, noise), 0)
                    
                    
                    full_value_list[h] = 0 if full_done else target_net(full_next_state).max(0)[0].item()
                    full_reward_list[h] = full_reward
                else:
                    break

            uncertainty_sample = extend_list(uncertainty_sample, n=H, elem=uncertainty_sample[-1])
            error_list = uncertainty_sample
            uncertainty_list = uncertainty_sample
            
            for index, val in enumerate(error_list):
                errors_per_step[index].append(val)
                
            overall_errors += error_list[1:]
            
            for index, val in enumerate(uncertainty_list):
                uncertainty_measures[index % H].append(val)
            
            negative_uncertainty_sample = [-1 * x for x in uncertainty_sample]
            weights = softmax_with_temperature(negative_uncertainty_sample, temp)
            weights = torch.Tensor(weights).to(device).unsqueeze(0)
            
            decays = torch.tensor([decay**i for i in range(len(uncertainty_sample))])   
            weights = weights * decays
            
            weights_list = weights.squeeze().tolist()
            for index, val in enumerate(weights_list):
                weights_per_step[index % H].append(val)
            
            effective_planning_horizon = sum(i * weight for i, weight in enumerate(weights_list, 1))
            effective_planning_horizons.append(effective_planning_horizon)
            
            indices = torch.arange(0, len(reward_list)).unsqueeze(0).float()
            indices_val = torch.arange(1, len(reward_list) + 1).unsqueeze(0).float()
            
            gamma_powers = GAMMA ** indices
            gamma_powers_val = GAMMA ** indices_val
            
            running_reward = (gamma_powers * reward_list).cumsum(dim=1)
            discounted_rewards = running_reward + gamma_powers_val * value_list
            
            # for full env (oracles)
            true_indices = torch.arange(0, len(full_reward_list)).unsqueeze(0).float()
            true_indices_val = torch.arange(1, len(full_reward_list) + 1).unsqueeze(0).float()
            
            true_gamma_powers = GAMMA ** true_indices
            true_gamma_powers_val = GAMMA ** true_indices_val
            
            true_running_reward = (true_gamma_powers * full_reward_list).cumsum(dim=1)
            true_discounted_rewards = true_running_reward + true_gamma_powers_val * full_value_list
            
            full_target_error = torch.abs(discounted_rewards - true_discounted_rewards).squeeze()
            full_target_error_list = full_target_error.tolist()
            for index, val in enumerate(full_target_error_list):
                full_target_errors[index % H].append(val)

            # Calculate the weighted average using weights
            weighted_avg = (weights * discounted_rewards).sum()
            avg = torch.Tensor([weighted_avg.item()]).detach()
            target = torch.cat((target, avg)).detach()

        target.requires_grad = True
        target = target.reshape(BATCH_SIZE, 1)
        TD_error = target - Q_s_a

    # Perfect Model    
    if H == 1:
        # If H=1, we are not performing rollouts, so we just use the standard DQN target computation
        none_terminal_next_state_index = torch.tensor([i for i, is_term in enumerate(is_terminal) if is_term == 0], dtype=torch.int64, device=device)
        none_terminal_next_states = next_states.index_select(0, none_terminal_next_state_index)
        Q_s_prime_a_prime = torch.zeros(BATCH_SIZE, 1, device=device)
        if len(none_terminal_next_states) != 0:
            Q_s_prime_a_prime[none_terminal_next_state_index] = target_net(none_terminal_next_states).detach().max(1)[0].unsqueeze(1)
        perfect_target = rewards + GAMMA * Q_s_prime_a_prime
   
    else:
        perfect_target = torch.empty((0))

        for i in range(BATCH_SIZE):

            initial_state = batch_samples.state[i].numpy()
            env.set_state_from_observation(initial_state)
            
            state = states[i]
            next_state = next_states[i]
            done = is_terminal[i]

            reward_list = torch.zeros(H).to(device)
            value_list = torch.zeros(H).to(device)
            
            reward_list[0] = rewards[i]
            value_list[0] = 0 if done else target_net(next_state).max(0)[0].item()
            
            env.set_state_from_observation(batch_samples.next_state[i].numpy())

            for h in range(1, H):
                if not done:
                    action = choose_greedy_action(state, policy_net)
                    
                    next_state, reward, terminated, truncated, _ = env.step(action.item())
                    done = terminated or truncated
                    
                    env.set_state_from_observation(next_state)
                    next_state = torch.Tensor(next_state).to(device)
                    noise = (torch.randn(1) * 50).to(device)
                    next_state = torch.cat((next_state, noise), 0)

                    value_list[h] = 0 if done else target_net(next_state).max(0)[0].item()
                    reward_list[h] = reward

                    state = next_state
                else:
                    break

            # Create a tensor with indices for the power operation
            indices = torch.arange(0, len(reward_list)).unsqueeze(0).float()
            indices_val = torch.arange(1, len(reward_list) + 1).unsqueeze(0).float()

            # Compute the gamma powers
            gamma_powers = GAMMA ** indices
            gamma_powers_val = GAMMA ** indices_val

            # Calculate the discounted rewards
            running_reward = (gamma_powers * reward_list).cumsum(dim=1)
            discounted_rewards = running_reward + gamma_powers_val * value_list

            weights = torch.ones(H)
            weights = torch.Tensor(weights).to(device).unsqueeze(0)

            decays = torch.tensor([decay**i for i in range(H)])
            weights = weights * decays

            # Calculate the average
            weighted_avg = (weights * discounted_rewards).sum() / weights.sum()
            avg = torch.Tensor([weighted_avg.item()]).detach()

            perfect_target = torch.cat((perfect_target, avg)).detach()

        perfect_target.requires_grad = True
        perfect_target = perfect_target.reshape(BATCH_SIZE, 1)
        
        perfect_TD_error = perfect_target - Q_s_a
        
    
    loss = F.mse_loss(Q_s_a, target)
    
    td_diff = torch.abs(TD_error - perfect_TD_error).squeeze()
    td_diff_list = td_diff.tolist()
    td_errors_diff += td_diff_list
    
    sign_TD_error = torch.sign(TD_error)
    sign_perfect_TD_error = torch.sign(perfect_TD_error)
    product = sign_TD_error * sign_perfect_TD_error
    count = torch.sum(product == -1)
    td_errors_direction_diff += count.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return errors_per_step, overall_errors, weights_per_step, effective_planning_horizons, td_errors_diff, td_errors_direction_diff, uncertainty_measures, full_target_errors

################################################################################################################
# dqn
#
# DQN algorithm with the option to disable replay and/or target network, and the function saves the training data.
#
# Inputs:
#   env: environment of the game
#   replay_off: disable the replay buffer and train on each state transition
#   target_off: disable target network
#   output_file_name: directory and file name prefix to output data and network weights, file saved as 
#       <output_file_name>_data_and_weights
#   store_intermediate_result: a boolean, if set to true will store checkpoint data every 1000 episodes
#       to a file named <output_file_name>_checkpoint
#   load_path: file path for a checkpoint to load, and continue training from
#   step_size: step-size for RMSProp optimizer
#
#################################################################################################################
def dqn(env, replay_off, target_off, output_file_name, store_intermediate_result=False, load_path=None, step_size_policy=STEP_SIZE, step_size_env=STEP_SIZE, rollout_constant=H, seed=SEED, env_hidden_size=ENV_HIDDEN_SIZE, temp=TEMPERATURE, decay=DECAY):
    # Set up the results file
    f = open(f"{output_file_name}.results", "a")
    f.write("Score\t#Frames\t")
    for i in range(1, rollout_constant):
        f.write(f"AvgErrorPerStep{i+1}\t")
        f.write(f"MedianErrorPerStep{i+1}\t")
        f.write(f"25thErrorPerStep{i+1}\t")
        f.write(f"75thErrorPerStep{i+1}\t")
        f.write(f"MaxErrorPerStep{i+1}\t")
        f.write(f"MinErrorPerStep{i+1}\t")
    f.write(f"OverallTargetAvgError\t")
    f.write(f"OverallTargetMedianError\t")
    f.write(f"OverallTarget25thError\t")
    f.write(f"OverallTarget75thError\t")
    f.write(f"OverallTargetMaxError\t")
    f.write(f"OverallTargetMinError\t")
    f.write(f"OverallOneStepAvgError\t")
    f.write(f"OverallOneStepMedianError\t")
    f.write(f"OverallOneStep25thError\t")
    f.write(f"OverallOneStep75thError\t")
    f.write(f"OverallOneStepMaxError\t")
    f.write(f"OverallOneStepMinError\t")
    for i in range(rollout_constant):
        f.write(f"WeightsPerStep{i+1}\t")
    f.write("EffectivePlanningHorizon\tTDErrorAbsDiff\tTDErrorDiffDir\t")
    for i in range(1, rollout_constant):
        f.write(f"CorrelationPerStep{i+1}\t")
    f.write("OverallCorrelation")
    f.write("\n")
    f.close()
    
    # Set up the seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    
    # Get channels and number of actions specific to each game
    in_channels = env.observation_space.shape[0] + 1
    num_actions = env.action_space.n

    # Instantiate networks, optimizer, loss and buffer
    policy_net = QNetwork(in_channels, num_actions).to(device)
    env_model = QuantileEnvModel(in_channels, num_actions, QUANTILES, env_hidden_size).to(device)
    
    replay_start_size = 0
    if not target_off:
        target_net = QNetwork(in_channels, num_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())

    if not replay_off:
        r_buffer = replay_buffer(REPLAY_BUFFER_SIZE)
        env_buffer = rollout_replay_buffer(REPLAY_BUFFER_SIZE)
        replay_start_size = REPLAY_START_SIZE

    optimizer = optim.RMSprop(policy_net.parameters(), lr=step_size_policy, alpha=SQUARED_GRAD_MOMENTUM, centered=True, eps=MIN_SQUARED_GRAD)
    env_model_optimizer = optim.Adam(env_model.parameters(), lr=step_size_env, weight_decay=WEIGHT_DECAY)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size_env, gamma=GAMMA)

    # Set initial values
    e_init = 0
    t_init = 0
    policy_net_update_counter_init = 0
    avg_return_init = 0.0
    data_return_init = []
    frame_stamp_init = []

    # Load model and optimizer if load_path is not None
    if load_path is not None and isinstance(load_path, str):
        checkpoint = torch.load(load_path)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

        if not target_off:
            target_net.load_state_dict(checkpoint['target_net_state_dict'])

        if not replay_off:
            r_buffer = checkpoint['replay_buffer']

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        e_init = checkpoint['episode']
        t_init = checkpoint['frame']
        policy_net_update_counter_init = checkpoint['policy_net_update_counter']
        avg_return_init = checkpoint['avg_return']
        data_return_init = checkpoint['return_per_run']
        frame_stamp_init = checkpoint['frame_stamp_per_run']

        # Set to training mode
        policy_net.trainWithRollout()
        if not target_off:
            target_net.trainWithRollout()

    # Data containers for performance measure and model related data
    data_return = data_return_init
    frame_stamp = frame_stamp_init
    avg_return = avg_return_init

    # Train for a number of frames
    t = t_init
    e = e_init
    env_model_loss = torch.tensor(1)
    policy_net_update_counter = policy_net_update_counter_init
    t_start = time.time()
    t_prev = 0
    
    while t <= NUM_FRAMES:
        
        # Initialize the return for every episode (we should see this eventually increase)
        G = 0.0

        # Initialize the environment and start state
        
        s_cont = torch.tensor(env.reset()[0], dtype=torch.float32).to(device)
        noise = (torch.randn(1) * 50).to(device)
        s_cont = torch.cat((s_cont, noise), 0)
        
        is_terminated = False
        rollout = []
        
        errors_per_step = [[] for _ in range(rollout_constant)]  # Nested list to store errors at each step for each rollout
        overall_errors = []
        weights_per_step = [[] for _ in range(rollout_constant)]  # Nested list to store weights at each step for each rollout
        td_errors_diff = []  # List to store absolute difference between TD errors for each planning step
        td_errors_direction_diff = 0  # Counter for number of times TD errors had different signs
        effective_planning_horizons = []  # List to store effective planning horizons for each planning rollout
        correlations_per_step = []  # List to store correlations between uncertainty and target error at each rollout step
        full_target_errors = [[] for _ in range(rollout_constant)]
        uncertainty_measures = [[] for _ in range(rollout_constant)]
        
        while (not is_terminated):
            # Generate data  
            action = choose_action(t, replay_start_size, s_cont, policy_net, num_actions)

            s_cont_prime, reward, terminated, truncated, _ = env.step(action.item())
            is_terminated = terminated or truncated

            s_cont_prime = torch.tensor(s_cont_prime, dtype=torch.float32, device=device)
            noise = (torch.randn(1) * 50).to(device)
            s_cont_prime = torch.cat((s_cont_prime, noise), 0)
        
            # Add current transition to rollout buffer
            rollout.append(transition(s_cont, s_cont_prime, action, reward, is_terminated))
            
            # If we've collected enough transitions for a rollout, add it to the buffer and remove the oldest transition
            if len(rollout) == ENV_H:
                env_buffer.add(list(rollout))  # Make a copy of current rollout
                rollout.pop(0)  # Remove the first transition
        
            sample_policy = None
            sample_env = None
            if replay_off:
                sample_policy = [transition(s_cont, s_cont_prime, action, reward, is_terminated)]
                sample_env = [transition(s_cont, s_cont_prime, action, reward, is_terminated)]
            else:
                # Write the current frame to replay buffer
                r_buffer.add(s_cont, s_cont_prime, action, reward, is_terminated)

                # Start learning when there's enough data and when we can sample a batch of size BATCH_SIZE
                if t > REPLAY_START_SIZE and len(r_buffer.buffer) >= BATCH_SIZE:
                    # Sample a batch
                    sample_policy = r_buffer.sample(BATCH_SIZE)
                    
                    # Sample a rollout
                    if len(env_buffer.buffer) >= BATCH_SIZE:  # Make sure we have enough rollouts to sample
                        sample_env = env_buffer.sample(BATCH_SIZE)

            if t % TRAINING_FREQ == 0 and sample_policy is not None:
                if target_off:
                    errors_per_step, overall_errors, weights_per_step, effective_planning_horizons, td_errors_diff, td_errors_direction_diff, full_target_errors, uncertainty_measures = trainWithRollout(sample_policy, policy_net, policy_net, optimizer, rollout_constant, env_model, temp=temp, decay=decay, 
                                                                                                                                      errors_per_step=errors_per_step, overall_errors=overall_errors, weights_per_step=weights_per_step, effective_planning_horizons=effective_planning_horizons, 
                                                                                                                                      td_errors_diff=td_errors_diff, td_errors_direction_diff=td_errors_direction_diff, full_target_errors=full_target_errors, uncertainty_measures=uncertainty_measures)
                else:
                    policy_net_update_counter += 1
                    errors_per_step, overall_errors, weights_per_step, effective_planning_horizons, td_errors_diff, td_errors_direction_diff, full_target_errors, uncertainty_measures = trainWithRollout(sample_policy, policy_net, target_net, optimizer, rollout_constant, env_model, temp=temp, decay=decay, 
                                                                                                                                      errors_per_step=errors_per_step, overall_errors=overall_errors, weights_per_step=weights_per_step, effective_planning_horizons=effective_planning_horizons, 
                                                                                                                                      td_errors_diff=td_errors_diff, td_errors_direction_diff=td_errors_direction_diff, full_target_errors=full_target_errors, uncertainty_measures=uncertainty_measures)    
            # Train every n number of frames defined by TRAINING_FREQ
            if t % TRAINING_FREQ == 0 and sample_env is not None:
                env_model_quantile, env_model_mean, env_model_loss = train_env_model(sample_env, env_model, env_model_optimizer, device, scheduler=None, clip_grad=0.5, weight_decay=WEIGHT_DECAY)
                    
            # Update the target network only after some number of policy network updates
            if not target_off and policy_net_update_counter > 0 and policy_net_update_counter % TARGET_NETWORK_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            G += reward
            
            t += 1

            # Continue the process
            s_cont = s_cont_prime

        # Increment the episodes
        e += 1
        
        # calculate the quantities for each episode
        error_per_step_per_episode = [sum(sublist) / len(sublist) for sublist in errors_per_step]
        error_per_step_median_per_episode = [np.median(sublist) for sublist in errors_per_step]
        error_per_step_25th_percentile_per_episode = [np.percentile(sublist, 25) for sublist in errors_per_step]
        error_per_step_75th_percentile_per_episode = [np.percentile(sublist, 75) for sublist in errors_per_step]
        error_per_step_max_per_episode = [np.max(sublist) for sublist in errors_per_step]
        error_per_step_min_per_episode = [np.min(sublist) for sublist in errors_per_step]
        
        overall_error_per_episode = sum(overall_errors) / len(overall_errors)
        overall_error_median_per_episode = np.median(overall_errors)
        overall_error_25th_percentile_per_episode = np.percentile(overall_errors, 25)
        overall_error_75th_percentile_per_episode = np.percentile(overall_errors, 75)
        overall_error_max_per_episode = np.max(overall_errors)
        overall_error_min_per_episode = np.min(overall_errors)
        # print(weights_per_step)
        weights_per_step_per_episode = [sum(sublist) / len(sublist) for sublist in weights_per_step]
        effective_planning_horizon_per_episode = sum(effective_planning_horizons) / len(effective_planning_horizons)
        td_errors_diff_per_episode = sum(td_errors_diff) / len(td_errors_diff)

        # assuming full_target_errors and uncertainty_measures are both lists of lists
        assert len(full_target_errors) == len(uncertainty_measures)

        full_target_errors = full_target_errors[1:]
        uncertainty_measures = uncertainty_measures[1:]
        for i in range(len(full_target_errors)):
            sublist_full_target_errors = full_target_errors[i]
            sublist_uncertainty_measures = uncertainty_measures[i]

            # calculate the Pearson correlation coefficient and the p-value
            correlation, p_value = pearsonr(sublist_full_target_errors, sublist_uncertainty_measures)
            correlations_per_step.append(correlation)
        
        # Flatten the nested lists into single lists
        flattened_full_target_errors = [item for sublist in full_target_errors for item in sublist]
        flattened_uncertainty_measures = [item for sublist in uncertainty_measures for item in sublist]

        # Calculate the correlation coefficient and the p-value
        overall_correlation, p_value = pearsonr(flattened_full_target_errors, flattened_uncertainty_measures)
        
        full_target_error_per_episode = sum(flattened_full_target_errors) / len(flattened_full_target_errors)
        full_target_error_median_per_episode = np.median(flattened_full_target_errors)
        full_target_error_25th_percentile_per_episode = np.percentile(flattened_full_target_errors, 25)
        full_target_error_75th_percentile_per_episode = np.percentile(flattened_full_target_errors, 75)
        full_target_error_max_per_episode = np.max(flattened_full_target_errors)
        full_target_error_min_per_episode = np.min(flattened_full_target_errors)

        # Save the return for each episode
        data_return.append(G)
        frame_stamp.append(t)

        # Logging exponentiated return only when verbose is turned on and only at 1000 episode intervals
        avg_return = 0.99 * avg_return + 0.01 * G
        if e % 1 == 0:
            logging.info("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                         str(numpy.around(avg_return, 2)) + " | Frame: " + str(t)+" | Time per frame: " +str((time.time()-t_start)/t) + " | Env Model Loss: " + str(env_model_loss.item()) + " | Env Model Quantiles: " + str(env_model_quantile.item()) + " | Env Model Means: " + str(env_model_mean.item())
                        )                    

            f = open(f"{output_file_name}.txt", "a")
            f.write("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                         str(np.around(avg_return, 2)) + " | Frame: " + str(t)+" | Time per frame: " +str((time.time()-t_start)/t) + " | Env Model Loss: " + str(env_model_loss.item()) + " | Env Model Quantiles: " + str(env_model_quantile.item()) + " | Env Model Means: " + str(env_model_mean.item())+ "\n"
                        )
            f.close()
            f = open(f"{output_file_name}.results", "a")
            f.write(str(G) + "\t" + str(t-t_prev) + "\t")
            for i in range(1, rollout_constant):
                f.write(str(error_per_step_per_episode[i]) + "\t")
                f.write(str(error_per_step_median_per_episode[i]) + "\t")
                f.write(str(error_per_step_25th_percentile_per_episode[i]) + "\t")
                f.write(str(error_per_step_75th_percentile_per_episode[i]) + "\t")
                f.write(str(error_per_step_max_per_episode[i]) + "\t")
                f.write(str(error_per_step_min_per_episode[i]) + "\t")
            f.write(str(full_target_error_per_episode) + "\t")
            f.write(str(full_target_error_median_per_episode) + "\t")
            f.write(str(full_target_error_25th_percentile_per_episode) + "\t")
            f.write(str(full_target_error_75th_percentile_per_episode) + "\t")
            f.write(str(full_target_error_max_per_episode) + "\t")
            f.write(str(full_target_error_min_per_episode) + "\t")
            f.write(str(overall_error_per_episode) + "\t")
            f.write(str(overall_error_median_per_episode) + "\t")
            f.write(str(overall_error_25th_percentile_per_episode) + "\t")
            f.write(str(overall_error_75th_percentile_per_episode) + "\t")
            f.write(str(overall_error_max_per_episode) + "\t")
            f.write(str(overall_error_min_per_episode) + "\t")
            for i in range(rollout_constant):
                f.write(str(weights_per_step_per_episode[i]) + "\t")
            f.write(str(effective_planning_horizon_per_episode) + "\t" + str(td_errors_diff_per_episode) + "\t" + str(td_errors_direction_diff) + "\t")
            for i in range(rollout_constant-1):
                f.write(str(correlations_per_step[i]) + "\t")
            f.write(str(overall_correlation))
            f.write("\n")
            f.close()
            
        t_prev = t

        # Save model data and other intermediate data if the corresponding flag is true
        if store_intermediate_result and e % 1 == 0:
            torch.save({
                        'episode': e,
                        'frame': t,
                        'policy_net_update_counter': policy_net_update_counter,
                        'policy_net_state_dict': policy_net.state_dict(),
                        'target_net_state_dict': target_net.state_dict() if not target_off else [],
                        'optimizer_state_dict': optimizer.state_dict(),
                        'avg_return': avg_return,
                        'return_per_run': data_return,
                        'frame_stamp_per_run': frame_stamp,
                        'replay_buffer': r_buffer if not replay_off else []
            }, output_file_name + "_checkpoint")

    # Print final logging info
    logging.info("Avg return: " + str(numpy.around(avg_return, 2)) + " | Time per frame: " + str((time.time()-t_start)/t))
        
    # Write data to file
    torch.save({
        'returns': data_return,
        'frame_stamps': frame_stamp,
        'policy_net_state_dict': policy_net.state_dict()
    }, output_file_name + "_data_and_weights")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", "-g", type=str)
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--loadfile", "-l", type=str)
    parser.add_argument("--alpha1", "-a1", type=float, default=STEP_SIZE)
    parser.add_argument("--alpha2", "-a2", type=float, default=STEP_SIZE)
    parser.add_argument("--temp", "-tp", type=float, default=TEMPERATURE)
    parser.add_argument("--rollout", "-rc", type=int, default=H)
    parser.add_argument("--decay", "-dc", type=float, default=DECAY)
    parser.add_argument("--hidden", "-hs", type=int, default=ENV_HIDDEN_SIZE)
    parser.add_argument("--seed", "-d", type=int, default=SEED)
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--replayoff", "-r", action="store_true")
    parser.add_argument("--targetoff", "-t", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # If there's an output specified, then use the user specified output.  Otherwise, create file in the current
    # directory with the game's name.
    if args.output:
        file_name = args.output
    else:
        file_name = os.getcwd() + "/" + "test"

    load_file_path = None
    if args.loadfile:
        load_file_path = args.loadfile

    env = gym.make("Acrobot-v1")
    env.reset(seed=args.seed)

    print('Cuda available?: ' + str(torch.cuda.is_available()))
    dqn(env, args.replayoff, args.targetoff, file_name, args.save, load_file_path, args.alpha1, args.alpha2, args.rollout, args.seed, args.hidden, args.temp, args.decay)


if __name__ == '__main__':
    main()