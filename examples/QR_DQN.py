################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian(ttian@ualberta.ca)                                                                                 #
#                                                                                                              #
# python3 dqn.py -g <game>                                                                                     #
#   -o, --output <directory/file name prefix>                                                                  #
#   -v, --verbose: outputs the average returns every 1000 episodes                                             #
#   -l, --loadfile <directory/file name of the saved model>                                                    #
#   -a, --alpha <number>: step-size parameter                                                                  #
#   -s, --save: save model data every 1000 episodes                                                            #
#   -r, --replayoff: disable the replay buffer and train on each state transition                              #
#   -t, --targetoff: disable the target network                                                                #
#                                                                                                              #
# References used for this implementation:                                                                     #
#   https://pytorch.org/docs/stable/nn.html#                                                                   #
#   https://pytorch.org/docs/stable/torch.html                                                                 #
#   https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html                                   #
################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import time

import random, numpy, argparse, logging, os

import numpy as np

from collections import namedtuple
from environment import Environment
from velenvironment import Velenvironment

################################################################################################################
# Constants
#
################################################################################################################
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
TARGET_NETWORK_UPDATE_FREQ = 1000
TRAINING_FREQ = 1
NUM_FRAMES = 5000000
FIRST_N_FRAMES = 100000
REPLAY_START_SIZE = 5000
END_EPSILON = 0.1
STEP_SIZE = 0.00025
GRAD_MOMENTUM = 0.95
SQUARED_GRAD_MOMENTUM = 0.95
MIN_SQUARED_GRAD = 0.01
GAMMA = 0.99
EPSILON = 1.0
TAU = 1e-2
LR = 1e-3

n_step = 1
N = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
quantile_tau = torch.FloatTensor([i/N for i in range(1,N+1)]).to(device)


################################################################################################################
# class QNetwork
#
# One hidden 2D conv with variable number of input channels.  We use 16 filters, a quarter of the original DQN
# paper of 64.  One hidden fully connected linear layer with a quarter of the original DQN paper of 512
# rectified units.  Finally, the output layer is a fully connected linear layer with a single output for each
# valid action.
#
################################################################################################################
def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size,layer_size, n_step, seed, N, layer_type="ff"):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.N = N

        self.head_1 = nn.Linear(self.input_shape[0], layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size*N)
        weight_init([self.head_1, self.ff_1])

    def forward(self, input):
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        return out.view(input.shape[0], self.N, self.action_size)
    
    def get_action(self,input):
        x = self.forward(input)
        return x.mean(dim=1)


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
# get_state
#
# Converts the state given by the environment to a tensor of size (in_channel, 10, 10), and then
# unsqueeze to expand along the 0th dimension so the function returns a tensor of size (1, in_channel, 10, 10).
#
# Input:
#   s: current state as numpy array
#
# Output: current state as tensor, permuted to match expected dimensions
#
################################################################################################################
def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()

################################################################################################################
# world_dynamics
#
# It generates the next state and reward after taking an action according to the behavior policy.  The behavior
# policy is epsilon greedy: epsilon probability of selecting a random action and 1 - epsilon probability of
# selecting the action with max Q-value.
#
# Inputs:
#   t : frame
#   replay_start_size: number of frames before learning starts
#   num_actions: number of actions
#   s: current state
#   env: environment of the game
#   policy_net: policy network, an instance of QNetwork
#
# Output: next state, action, reward, is_terminated
#
################################################################################################################
def world_dynamics(t, replay_start_size, num_actions, s, env, policy_net):

    # A uniform random policy is run before the learning starts
    if t < replay_start_size:
        action = torch.tensor([[random.randrange(num_actions)]], device=device)
    else:
        # Epsilon-greedy behavior policy for action selection
        # Epsilon is annealed linearly from 1.0 to END_EPSILON over the FIRST_N_FRAMES and stays 0.1 for the
        # remaining frames
        epsilon = END_EPSILON if t - replay_start_size >= FIRST_N_FRAMES \
            else ((END_EPSILON - EPSILON) / FIRST_N_FRAMES) * (t - replay_start_size) + EPSILON

        if numpy.random.binomial(1, epsilon) == 1:
            action = torch.tensor([[random.randrange(num_actions)]], device=device)
        else:
            # State is 10x10xchannel, max(1)[1] gives the max action value (i.e., max_{a} Q(s, a)).
            # view(1,1) shapes the tensor to be the right form (e.g. tensor([[0]])) without copying the
            # underlying tensor.  torch._no_grad() avoids tracking history in autograd.
            with torch.no_grad():
                action = policy_net(s).max(1)[1].view(1, 1)

    # Act according to the action and observe the transition and reward
    reward, terminated = env.act(action)

    # Obtain s_prime
    s_prime = get_state(env.state())

    return s_prime, action, torch.tensor([[reward]], device=device).float(), torch.tensor([[terminated]], device=device)


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
def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 32, 32), "huber loss has wrong shape"
    return loss


def train(sample, qnetwork_local, qnetwork_target, optimizer):
    optimizer.zero_grad()
    
    # Batch is a list of namedtuple's, the following operation returns samples grouped by keys
    batch_samples = transition(*zip(*sample))
    # print(batch_samples)
    
    # states, next_states are of tensor (BATCH_SIZE, in_channel, 10, 10) - inline with pytorch NCHW format
    # actions, rewards, is_terminal are of tensor (BATCH_SIZE, 1)
    states = torch.cat(batch_samples.state)
    next_states = torch.cat(batch_samples.next_state)
    actions = torch.cat(batch_samples.action)
    rewards = torch.cat(batch_samples.reward)
    is_terminal = torch.cat(batch_samples.is_terminal)
    
    Q_targets_next = qnetwork_target(next_states).detach().cpu() #.max(2)[0].unsqueeze(1) #(batch_size, 1, N)
    action_indx = torch.argmax(Q_targets_next.mean(dim=1), dim=1, keepdim=True)

    Q_targets_next = Q_targets_next.gather(2, action_indx.unsqueeze(-1).expand(BATCH_SIZE, N, 1)).transpose(1,2)

    assert Q_targets_next.shape == (BATCH_SIZE,1, N)
    # Compute Q targets for current states 
    Q_targets = rewards.unsqueeze(-1) + (GAMMA**n_step * Q_targets_next.to(device) * (1 - is_terminal.unsqueeze(-1)))
    # Get expected Q values from local model
    Q_expected = qnetwork_local(states).gather(2, actions.unsqueeze(-1).expand(BATCH_SIZE, N, 1))
    # Compute loss
    td_error = Q_targets - Q_expected
    assert td_error.shape == (BATCH_SIZE, N, N), "wrong td error shape"
    huber_l = calculate_huber_loss(td_error, 1.0)
    quantil_l = abs(quantile_tau -(td_error.detach() < 0).float()) * huber_l / 1.0

    loss = quantil_l.sum(dim=1).mean(dim=1) # , keepdim=True if per weights get multipl
    loss = loss.mean()
    # Minimize the loss
    loss.backward()
    #clip_grad_norm_(self.qnetwork_local.parameters(),1)
    optimizer.step()

    # ------------------- update target network ------------------- #
    soft_update(qnetwork_local, qnetwork_target)
    return loss.detach().cpu().numpy()  


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
def dqn(env, replay_off, target_off, output_file_name, store_intermediate_result=False, load_path=None, step_size=STEP_SIZE):
    torch.set_num_threads(1)
    # Get channels and number of actions specific to each game
    in_channels = env.state_shape()[0]
    num_actions = 4
    seed = 1
    np.random.seed(seed)
    Q_updates = 0

    action_step = 4
    last_action = None
    
    action_size     = 4
    state_size = env.state().shape
    layer_size = 256

    # Q-Network
    qnetwork_local = QNetwork(state_size, action_size,layer_size, n_step, seed, N).to(device)
    optimizer = optim.Adam(qnetwork_local.parameters(), lr=LR)

    replay_start_size = 0
    if not target_off:
        network_target = QNetwork(state_size, action_size,layer_size, n_step, seed, N).to(device)
        network_target.load_state_dict(qnetwork_local.state_dict())

    if not replay_off:
        r_buffer = replay_buffer(REPLAY_BUFFER_SIZE)
        replay_start_size = REPLAY_START_SIZE

    optimizer = optim.RMSprop(qnetwork_local.parameters(), lr=step_size, alpha=SQUARED_GRAD_MOMENTUM, centered=True, eps=MIN_SQUARED_GRAD)

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
        qnetwork_local.load_state_dict(checkpoint['policy_net_state_dict'])

        if not target_off:
            network_target.load_state_dict(checkpoint['target_net_state_dict'])

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
        qnetwork_local.train()
        if not target_off:
            network_target.train()

    # Data containers for performance measure and model related data
    data_return = data_return_init
    frame_stamp = frame_stamp_init
    avg_return = avg_return_init

    # Train for a number of frames
    t = t_init
    e = e_init
    policy_net_update_counter = policy_net_update_counter_init
    t_start = time.time()
    while t < NUM_FRAMES:
        # Initialize the return for every episode (we should see this eventually increase)
        G = 0.0

        # Initialize the environment and start state
        env.reset()
        s = get_state(env.state())
        is_terminated = False
        while(not is_terminated) and t < NUM_FRAMES:
            # Generate data
            s_prime, action, reward, is_terminated = world_dynamics(t, replay_start_size, num_actions, s, env, qnetwork_local)
            
            sample = None
            if replay_off:
                sample = [transition(s, s_prime, action, reward, is_terminated)]
            else:
                # Write the current frame to replay buffer
                r_buffer.add(s, s_prime, action, reward, is_terminated)

                # Start learning when there's enough data and when we can sample a batch of size BATCH_SIZE
                if t > REPLAY_START_SIZE and len(r_buffer.buffer) >= BATCH_SIZE:
                    # Sample a batch
                    sample = r_buffer.sample(BATCH_SIZE)

            # Train every n number of frames defined by TRAINING_FREQ
            if t % TRAINING_FREQ == 0 and sample is not None:
                if target_off:
                    train(sample, qnetwork_local, network_target, optimizer)
                else:
                    policy_net_update_counter += 1
                    train(sample, qnetwork_local, network_target, optimizer)

            # Update the target network only after some number of policy network updates
            if not target_off and policy_net_update_counter > 0 and policy_net_update_counter % TARGET_NETWORK_UPDATE_FREQ == 0:
                network_target.load_state_dict(network_target.state_dict())

            G += reward.item()

            t += 1

            # Continue the process
            s = s_prime

        # Increment the episodes
        e += 1

        # Save the return for each episode
        data_return.append(G)
        frame_stamp.append(t)

        # Logging exponentiated return only when verbose is turned on and only at 1000 episode intervals
        avg_return = 0.99 * avg_return + 0.01 * G
        if e % 1000 == 0:
            logging.info("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                         str(numpy.around(avg_return, 2)) + " | Frame: " + str(t)+" | Time per frame: " +str((time.time()-t_start)/t) )

            f = open(f"{output_file_name}.txt", "a")
            f.write("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                         str(np.around(avg_return, 2)) + " | Frame: " + str(t)+" | Time per frame: " +str((time.time()-t_start)/t) + "\n" )
            f.close()
        # Save model data and other intermediate data if the corresponding flag is true
        if store_intermediate_result and e % 1000 == 0:
            torch.save({
                        'episode': e,
                        'frame': t,
                        'policy_net_update_counter': policy_net_update_counter,
                        'policy_net_state_dict': qnetwork_local.state_dict(),
                        'target_net_state_dict': network_target.state_dict() if not target_off else [],
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
        'policy_net_state_dict': qnetwork_local.state_dict()
    }, output_file_name + "_data_and_weights")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", "-g", type=str)
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--loadfile", "-l", type=str)
    parser.add_argument("--alpha", "-a", type=float, default=STEP_SIZE)
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
        file_name = os.getcwd() + "/" + args.game

    load_file_path = None
    if args.loadfile:
        load_file_path = args.loadfile

    env = Environment(args.game)

    print('Cuda available?: ' + str(torch.cuda.is_available()))
    dqn(env, args.replayoff, args.targetoff, file_name, args.save, load_file_path, args.alpha)


if __name__ == '__main__':
    main()


