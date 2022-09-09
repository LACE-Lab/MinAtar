from environment import Environment
import numpy as np
import scipy.optimize 
import math
class Velenvironment():
    def __init__(self, env_name, sticky_action_prob = 0.1, difficulty_ramping = True, random_seed = None):
       
        self.env = Environment(env_name, sticky_action_prob = 0.1, difficulty_ramping = True, random_seed = None)
        self.name = env_name
        self.past_state = self.env.continuous_state()
    # Wrapper for env.act
    def act(self, a):
        self.past_state = self.env.continuous_state()
        return self.env.act(a)

    # Wrapper for env.state
    def state(self):
        return self.env.state()

    # Wrapper for env.reset
    def reset(self):
        return self.env.reset()

    # Wrapper for env.state_shape
    def state_shape(self):
        return self.env.state_shape()

    # All MinAtar environments have 6 actions
    def num_actions(self):
        return self.env.num_actions()

    # Name of the MinAtar game associated with this environment
    def game_name(self):
        return self.env.game_name()

    # Wrapper for env.minimal_action_set
    def minimal_action_set(self):
        return self.env.minimal_action_set()

    # Display the current environment state for time milliseconds using matplotlib
    def display_state(self, time=50):
        self.env.display_state(time = 50) 
    def close_display(self):
        self.env.close_display()

    def continuous_state(self):

        current_state = self.env.continuous_state()
        # print("current_state: ", current_state)
        # print(len(current_state))
        
        for i in range(len(current_state)): 
            one_hot = [0 for i in range(len(current_state))]  #encode color
            one_hot[i] = 1 
            one_hot = tuple(one_hot)
            # print("one_hot: ", one_hot)
            if  current_state[i] != [] and self.past_state[i] != [] : 
                
                l1 = np.asarray(current_state[i])
                l2 = np.asarray(self.past_state[i])

                #Match objects from the past and present state 
                dist_matrix = scipy.spatial.distance_matrix(l1,l2)
                assignmentscur, assignmentspast = scipy.optimize.linear_sum_assignment(dist_matrix)

                #calculate and insert velocities for matched objects 
                for j in range(len(assignmentscur)): 
                    one_hot_x = [0 for i in range(20)]
                    one_hot_y = [0 for i in range(20)]
                    one_hot_xvel = [0 for i in range(20)]
                    one_hot_yvel = [0 for i in range(20)]
                    curindex = assignmentscur[j]
                    pastindex = assignmentspast[j]
                    x = current_state[i][curindex][0]
                    one_hot_x[int(x)] = 1
                    y = current_state[i][curindex][1]
                    one_hot_y[int(y)] = 1
                    xvel = x - self.past_state[i][pastindex][0]
                    one_hot_xvel[int(xvel)+10] = 1
                    yvel = y - self.past_state[i][pastindex][1]
                    one_hot_yvel[int(yvel)+10] = 1
                    # current_state[i][curindex] = (x,y,xvel,yvel) + one_hot + (1,)
                    current_state[i][curindex] = tuple(one_hot_x) + tuple(one_hot_y) + tuple(one_hot_xvel) + tuple(one_hot_yvel) + one_hot + (1,)
                    # print(current_state[i][curindex])

            #Set velocities of unmatched objects to 0 
            for j in range(len(current_state[i])):
                if len(current_state[i][j]) == 2: 
                    one_hot_x = [0 for i in range(20)]
                    one_hot_y = [0 for i in range(20)]
                    one_hot_xvel = [0 for i in range(20)]
                    one_hot_yvel = [0 for i in range(20)]
                    one_hot_x[int(current_state[i][j][0])] = 1
                    one_hot_y[int(current_state[i][j][1])] = 1
                    current_state[i][j] = tuple(one_hot_x) + tuple(one_hot_y) + tuple(one_hot_xvel) + tuple(one_hot_yvel) + one_hot + (0,)
                    # current_state[i][j] = (current_state[i][j][0],current_state[i][j][1],0,0) + one_hot + (0,)
        
        # x = current_state[0][0][0]
        # y = current_state[0][0][1]
        # if (x == 0):
        #     if (y == 0):
        #         current_state = [1,0,0,0,0,0,0,0,0]
        #     elif (y == 1):
        #         current_state = [0,1,0,0,0,0,0,0,0]
        #     elif (y == 2):
        #         current_state = [0,0,1,0,0,0,0,0,0]
        # elif (x == 1):
        #     if (y == 0):
        #         current_state = [0,0,0,1,0,0,0,0,0]
        #     elif (y == 1):
        #         current_state = [0,0,0,0,1,0,0,0,0]
        #     elif (y == 2):
        #         current_state = [0,0,0,0,0,1,0,0,0]
        # elif (x == 2):
        #     if (y == 0):
        #         current_state = [0,0,0,0,0,0,1,0,0]
        #     elif (y == 1):
        #         current_state = [0,0,0,0,0,0,0,1,0]
        #     elif (y == 2):
        #         current_state = [0,0,0,0,0,0,0,0,1]
        # (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)

        return current_state
    
    def new_objects(self):

            
        current_state = self.env.continuous_state()
        new_state = []
        for i in range(len(current_state)): 
            one_hot = [0 for i in range(len(current_state))]
            one_hot[i] = 1 
            one_hot = tuple(one_hot)
            max_pad = 3 
            if  current_state[i] != [] and self.past_state[i] != [] : 
            
                
               
                l1 = np.asarray(current_state[i])
                l2 = np.asarray(self.past_state[i])

                #Match objects from the past and present state 
                dist_matrix = scipy.spatial.distance_matrix(l1,l2)
                assignmentscur, assignmentspast = scipy.optimize.linear_sum_assignment(dist_matrix)
                
                #calculate and insert velocities for matched objects 
                for j in range(len(assignmentscur)): 
                    curindex = assignmentscur[j]
                    pastindex = assignmentspast[j]
                    x = current_state[i][curindex][0]
                    y = current_state[i][curindex][1]
                    xvel = x - self.past_state[i][pastindex][0]
                    yvel = y - self.past_state[i][pastindex][1]
                    current_state[i][curindex] = (x,y) + one_hot 
            
            #Set velocities of unmatched objects to 0 
            for j in range(len(current_state[i])):
                if len(current_state[i][j]) == 2: 
                    new_state +=  [list((current_state[i][j][0],current_state[i][j][1]) + one_hot)]
        #pad with 0s up to a maximum number of new objects
        for i in range(max_pad - len(new_state)): 
            new_state += [[0,0] + [0 for i in range(len(current_state))]]

              
            
        
        return new_state
    # Return a string that represents the current state of the environment
    # (Not including the RNG state)
    def save_state(self):
        return self.env.save_state()

    # Take a string once returned by save_state and restore that state
    # Because RNG state is not restored, behavior will not necessarily be
    # the same every time the same state is loaded.
    # This means this function is suitable for planning using rollouts.
    def load_state(self, state_str):
        self.env.load_state(state_str)
    def distance(self,a,b): 
        return 