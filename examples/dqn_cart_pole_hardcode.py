################################################################################################################
# Authors:                                                                                                     #
# Zoe
#
# This file contains DQN alg training with Cart Pole.
# Incorpeted MVE, NN for Env model
# The Env model predicts only the state
# The reward and termination rule are hardcoded
################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as f
import pytorch_lightning as pl
import torch.optim as optim
import time
import gym

import random, numpy, argparse, logging, os

import numpy as np
from tqdm import tqdm

from collections import namedtuple
from customCartPole import CustomCartPole

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
STEP_SIZE = 0.0001
WEIGHT_DECAY = 0.0001
GRAD_MOMENTUM = 0.95
SQUARED_GRAD_MOMENTUM = 0.95
MIN_SQUARED_GRAD = 0.01
GAMMA = 0.99
EPSILON = 1
H = 1 # rollout constant
SEED = 42

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

# Define the approximate model of the environment
class EnvModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(EnvModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.state = None
        
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization for hidden layer
        self.dropout = nn.Dropout(0.1)  # Dropout layer with dropout rate 0.1
        self.fc2 = nn.Linear(hidden_size, state_size)
        
    def load_state(self, state):
        self.state = state.clone().detach()

    def save_state(self):
        return self.state.clone().detach()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # Apply batch normalization
        x = f.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        
        state = x[:, :self.state_size]
        
        return state
    
    def single_forward(self, x):
        x = self.fc1(x)
        x = f.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        
        state = x[:self.state_size]
        
        return state
    
    def step(self, action):
        one_hot_action = torch.eye(self.action_size)[action.squeeze().long()]
        state_action_pair = torch.cat((self.state, one_hot_action), dim=-1)
        predicted_next_state = self.single_forward(state_action_pair)
        return predicted_next_state

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

def train_env_model(sample, env_model, optimizer, device, scheduler=None, clip_grad=None, weight_decay=0):
    batch_samples = transition(*zip(*sample))

    states = torch.vstack((batch_samples.state)).to(device)
    next_states = torch.vstack((batch_samples.next_state)).to(device)

    actions = torch.cat(batch_samples.action, 0)
    actions = actions.reshape(BATCH_SIZE, 1).type(torch.int64)

    rewards = torch.tensor(batch_samples.reward).to(device).reshape(BATCH_SIZE, 1)
    is_terminal = torch.tensor(batch_samples.is_terminal).to(device).reshape(BATCH_SIZE, 1).type(torch.float32)

    # One-hot encode the actions
    one_hot_actions = torch.eye(env_model.action_size)[actions.squeeze().long()]

    # Concatenate states and one-hot encoded actions
    state_action_pairs = torch.cat((states, one_hot_actions), dim=-1)

    # Forward pass to get predicted next states, rewards, and done
    predicted_next_states = env_model(state_action_pairs)

    # Compute the loss
    loss = f.mse_loss(predicted_next_states, next_states)

    # L2 regularization
    l2_reg = torch.tensor(0., device=device)
    for param in env_model.parameters():
        l2_reg += torch.norm(param, p=2)
    loss += weight_decay * l2_reg

    # Zero gradients, backprop, and update the weights of env_model
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    if clip_grad is not None:
        torch.nn.utils.clip_grad_norm_(env_model.parameters(), clip_grad)

    optimizer.step()

    return loss
    
    
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
        action = torch.max(actions_value, 1)[1].data.numpy() # pick the highest one
        action = torch.tensor(action).to(device)

    return action

def choose_greedy_action(state, policy_net):
    # print(state)
    x = state.to(device).clone().detach().unsqueeze(0)
    
    actions_value = policy_net(x) # score of actions
    action = torch.max(actions_value, 1)[1].data.numpy() # pick the highest one
    action = torch.tensor(action).to(device)

    return action

def trainWithRollout(sample, policy_net, target_net, optimizer, H):
    env = CustomCartPole()
    env.reset()
    
    # Initialize the environment model
    in_channels = env.observation_space.shape[0]
    num_actions = env.action_space.n
    env_model = EnvModel(in_channels, num_actions).to(device)

    # unzip the batch samples and turn components into tensors

    batch_samples = transition(*zip(*sample))

    states = torch.vstack((batch_samples.state)).to(device)
    next_states = torch.vstack((batch_samples.next_state)).to(device)

    actions = torch.cat(batch_samples.action, 0)
    actions = actions.reshape(BATCH_SIZE, 1).type(torch.int64)

    rewards = torch.tensor(batch_samples.reward).to(device).reshape(BATCH_SIZE, 1)
    is_terminal = torch.tensor(batch_samples.is_terminal).to(device).reshape(BATCH_SIZE, 1)

    Q_s_a = policy_net(states).gather(1, actions)

    avg_list = torch.empty((0))

    for i in range(BATCH_SIZE):
        cpu = False

        initial_state = torch.tensor(batch_samples.state[i], dtype=torch.float32).to(device)
        env_model.load_state(initial_state)
        
        state = states[i]
        next_state = next_states[i]
        done = is_terminal[i]

        reward_list = torch.zeros(H).to(device)
        value_list = torch.zeros(H).to(device)
        
        reward_list[0] = rewards[i]
        value_list[0] = 0 if done else target_net(next_state).max(0)[0].item()
        
        next_state = torch.tensor(batch_samples.next_state[i], dtype=torch.float32).to(device)
        env_model.load_state(next_state)

        for h in range(1, H):
            if not done:
                action = choose_greedy_action(state, policy_net)
                next_state = env_model.step(action)
                # hardcode termination rule
                position = next_state[0]
                angle = next_state[2]
                
                done = False
                if position <= -2.4 or position >= 2.4:
                    done = True
                if angle <= -.2095 or angle >= .2095:
                    done = True
                    
                if done:
                    reward = 0
                else:
                    reward = 1
                
                env_model.load_state(next_state)

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

        # Calculate the average
        avg = discounted_rewards.mean()
        avg = torch.Tensor([avg.item()]).detach()

        avg_list = torch.cat((avg_list, avg)).detach()

    avg_list.requires_grad = True
    avg_list = avg_list.reshape(BATCH_SIZE, 1)

    loss = f.mse_loss(Q_s_a, avg_list)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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
def dqn(env, replay_off, target_off, output_file_name, store_intermediate_result=False, load_path=None, step_size_policy=STEP_SIZE, step_size_env=STEP_SIZE, rollout_constant=H, seed=SEED):
    
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
    in_channels = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Instantiate networks, optimizer, loss and buffer
    policy_net = QNetwork(in_channels, num_actions).to(device)
    env_model = EnvModel(in_channels, num_actions).to(device)
    
    replay_start_size = 0
    if not target_off:
        target_net = QNetwork(in_channels, num_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())

    if not replay_off:
        r_buffer = replay_buffer(REPLAY_BUFFER_SIZE)
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
    
    while t <= NUM_FRAMES:
    # for t in tqdm(range(NUM_FRAMES)):
        # Initialize the return for every episode (we should see this eventually increase)
        G = 0.0

        # Initialize the environment and start state
        cpu = True
        
        if type(env.reset()) != numpy.ndarray:
            cpu = False
            s_cont = torch.tensor(env.reset()[0], dtype=torch.float32).to(device)
        else:
            s_cont = torch.tensor(env.reset(), dtype=torch.float32).to(device)

        is_terminated = False
        
        while (not is_terminated):
            # Generate data  
            action = choose_action(t, replay_start_size, s_cont, policy_net, num_actions)

            if cpu == False:
                s_cont_prime, reward, is_terminated, _, _ = env.step(action.item())
            else:
                s_cont_prime, reward, is_terminated, _ = env.step(action.item())

            s_cont_prime = torch.tensor(s_cont_prime, dtype=torch.float32, device=device)

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
                    sample_env = r_buffer.sample(BATCH_SIZE)

            # Train every n number of frames defined by TRAINING_FREQ
            if t % TRAINING_FREQ == 0 and sample_env is not None:
                env_model_loss = train_env_model(sample_env, env_model, env_model_optimizer, device, scheduler=None, clip_grad=0.5, weight_decay=WEIGHT_DECAY)

            if t % TRAINING_FREQ == 0 and sample_policy is not None:
                if target_off:
                    trainWithRollout(sample_policy, policy_net, policy_net, optimizer, rollout_constant)
                else:
                    policy_net_update_counter += 1
                    trainWithRollout(sample_policy, policy_net, target_net, optimizer, rollout_constant)
                    
            # Update the target network only after some number of policy network updates
            if not target_off and policy_net_update_counter > 0 and policy_net_update_counter % TARGET_NETWORK_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            G += reward
            
            t += 1

            # Continue the process
            s_cont = s_cont_prime

        # Increment the episodes
        e += 1

        # Save the return for each episode
        data_return.append(G)
        frame_stamp.append(t)

        # Logging exponentiated return only when verbose is turned on and only at 1000 episode intervals
        avg_return = 0.99 * avg_return + 0.01 * G
        if e % 100 == 0:
            logging.info("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                         str(numpy.around(avg_return, 2)) + " | Frame: " + str(t)+" | Time per frame: " +str((time.time()-t_start)/t) + " | Env Model Loss: " + str(env_model_loss.item())
                        )                    

            f = open(f"{output_file_name}.txt", "a")
            f.write("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                         str(np.around(avg_return, 2)) + " | Frame: " + str(t)+" | Time per frame: " +str((time.time()-t_start)/t) + " | Env Model Loss: " + str(env_model_loss.item()) + "\n"
                        )
            f.close()
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
    parser.add_argument("--rollout", "-rc", type=int, default=H)
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

    env = gym.make("CartPole-v1")
    env.seed(args.seed)

    print('Cuda available?: ' + str(torch.cuda.is_available()))
    dqn(env, args.replayoff, args.targetoff, file_name, args.save, load_file_path, args.alpha1, args.alpha2, args.rollout, args.seed)


if __name__ == '__main__':
    main()