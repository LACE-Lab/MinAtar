################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian (ttian@ualberta.ca)                                                                                #
################################################################################################################
import numpy as np


#####################################################################################################################
# Env
#
# The player controls a paddle on the bottom of the screen and must bounce a ball tobreak 3 rows of bricks along the 
# top of the screen. A reward of +1 is given for each brick broken by the ball.  When all bricks are cleared another 3 
# rows are added. The ball travels only along diagonals, when it hits the paddle it is bounced either to the left or 
# right depending on the side of the paddle hit, when it hits a wall or brick it is reflected. Termination occurs when
# the ball hits the bottom of the screen. The balls direction is indicated by a trail channel.
#
#####################################################################################################################
class Env:
    def __init__(self, ramping = None, seed = None):
        self.channels ={
            'player':0,
        }
        self.action_map = ['n','l','u','r','d','f']
        self.random = np.random.RandomState(seed)
        self.reset()

    # Update environment according to agent action
    def act(self, a):
        r = -1
        if(self.terminal):
            return r, self.terminal
            
        a = self.action_map[a]

        # Resolve player action
        if(a=='l'):
            self.player_x = max(0, self.player_x-1)
        elif(a=='r'):
            self.player_x = min(2, self.player_x+1)
        elif(a=='u'):
            self.player_y = max(0, self.player_y-1)
        elif(a=='d'):
            self.player_y = min(2, self.player_y+1)

        if ((self.player_y == 2) and (self.player_x == 2)):
            self.terminal = True
        
        return r, self.terminal

    # Query the current level of the difficulty ramp, difficulty does not ramp in this game, so return None
    def difficulty_ramp(self):
        return None  

    # Process the game-state into the 10x10xn state provided to the agent and return
    def state(self):
        state = np.zeros((10,10,len(self.channels)),dtype=bool)
        state[self.player_y,self.player_x,self.channels['player']] = 1
        return state

    # Reset to start state for new episode
    def reset(self):
        self.player_x = 1
        self.player_y = 1
        self.terminal = False

    # Dimensionality of the game-state (10x10xn)
    def state_shape(self):
        return [10,10,len(self.channels)]

    # Subset of actions that actually have a unique impact in this environment
    def minimal_action_set(self):
        minimal_actions = ['n','l','r']
        return [self.action_map.index(x) for x in minimal_actions]

    def continuous_state(self):
        objByColor = [[] for i in range(len(self.channels))]
        objByColor[self.channels['player']].append((float(self.player_x), float(self.player_y))) # Paddle
        return objByColor;
    
    def save_state(self):
        state_str  = str(self.player_x) + " "
        state_str += str(self.player_y) + " "
        state_str += str(int(self.terminal))
        return state_str
        
    def load_state(self, state_str):
        state_lst = state_str.split()
        state_iter = iter(state_lst)
        self.player_x = int(next(state_iter))
        self.player_y = int(next(state_iter))
        self.terminal = bool(int(next(state_iter)))
        
