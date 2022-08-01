                                                                         #
################################################################################################################
import numpy as np
width=5
runway_length=4
shooting_length=2

#####################################################################################################################
# Env
#
# 
#####################################################################################################################
class Env:
    def __init__(self, ramping = None, seed = None):
        self.channels ={
            'agent':0,
            'target':1,
            'bullet':2
        }

        self.action_map = ['0']
        for i in range(1,width+1): 
            self.action_map.append(i)
        self.random = np.random.RandomState(seed)
        self.runway_index=0
        self.runway_pos=0
        self.bullet_pos=0
        self.target_pos=0
        self.target_v=1
        self.reset()

    # Update environment according to agent action
    def act(self, a):
        r = 0
        if(self.terminal):
            return r, self.terminal
        
        # Update target position
        if self.target_pos==0: 
            self.target_v=1
        elif self.target_pos == width-1: 
            self.target_v=-1
        self.target_pos+=self.target_v
        if 0 < self.runway_pos < runway_length: 
            self.runway_pos +=1
        
        if self.bullet_pos==1: 
            self.bullet_pos +=1
            
        a = self.action_map[a]

        # Resolve player action
        if(a=='0'):
            if self.runway_pos == runway_length and self.bullet_pos == 0:
                self.bullet_pos = 1
        elif(a in self.action_map):
            if self.runway_pos == 0 and self.runway_index == 0: 
                self.runway_index = a
                self.runway_pos = 1
        
        if self.target_pos == self.runway_index and self.bullet_pos == shooting_length: 
            r+=1

        if self.bullet_pos == shooting_length: 
            self.terminal == True
        
        return r, self.terminal

    # Query the current level of the difficulty ramp, difficulty does not ramp in this game, so return None
    def difficulty_ramp(self):
        return None  

    # Process the game-state into the 10x10xn state provided to the agent and return
    def state(self):
        # state = np.zeros((width,runway_length+shooting_length,len(self.channels)),dtype=bool)
        state=[self.runway_index,self.runway_pos,self.target_pos,self.bullet_pos]
        # state[, :, self.channels['target']] = 1
        # state[(self.bullet_pos),:,self.channels['bullet']] = 1
        return state

    # Reset to start state for new episode
    def reset(self):
        self.runway_index=0
        self.runway_pos=0
        self.bullet_pos=0
        self.target_pos=0
        self.target_v=1
        self.terminal = False

    # Dimensionality of the game-state (10x10xn)
    def state_shape(self):
        return [10,10,len(self.channels)]



    def continuous_state(self):
        objByColor = [[] for i in range(len(self.channels))]
        objByColor[self.channels['agent']].append((float(self.runway_index), float(self.runway_pos))) # Paddle
        objByColor[self.channels['target']].append((float(self.target_pos), float(self.target_v))) 
        objByColor[self.channels['bullet']].append((float(self.bullet_pos))) # Trail
        return objByColor
    
    def save_state(self):
        state_str  = str(self.runway_index) + " "
        state_str += str(self.runway_pos) + " "
        state_str += str(self.target_pos) + " "
        state_str += str(self.target_v) + " "
        state_str += str(self.bullet_pos) + " "
        state_str += str(int(self.terminal))
        return state_str
        
    def load_state(self, state_str):
        state_lst = state_str.split()
        state_iter = iter(state_lst)
        self.runway_index= int(next(state_iter))
        self.runway_pos = int(next(state_iter))
        self.target_pos = int(next(state_iter))
        self.target_v = int(next(state_iter))       
        self.bullet_pos = int(next(state_iter))
        self.terminal = bool(int(next(state_iter)))
        