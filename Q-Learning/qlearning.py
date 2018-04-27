import numpy as np
import fun
import random


class QLearning:
    def __init__(self, environment, gamma=0.9, n_episodes=100):
        self.environment = environment
        self.environment_shape = environment.shape
        self.gamma = gamma
        # self.explore_mat = np.zeros(
        #     (environment.shape[0]*environment.shape[1], 4))
        self.q_table = self.build_q_table()
        self.r_table = self.build_r_table()
        self.current_state = (0, 0)

    # def reward_table(self):
    #     np.zeros_like(self.explore_mat)

    def build_q_table(self):
        q_table = {}
        for i in range(self.environment_shape[0]):
            for j in range(self.environment_shape[1]):
                q_table[(i, j)] = {"N": 0, "E": 0, "S": 0, "W": 0}
        return q_table

    def build_r_table(self):
        r_table = {}
        for i in range(self.environment_shape[0]):
            for j in range(self.environment_shape[1]):
                print(i, j)
                r_table[(i, j)] = {}  # []
                # to North
                if i-1 < 0:
                    r_table[(i, j)]["N"] = None  # r_table[(i, j)].append(None)
                else:
                    # r_table[(i, j)].append(matrix_r[i-1,j])
                    r_table[(i, j)]["N"] = self.environment[i-1, j]

                # to East
                if j+1 >= self.environment_shape[1]:
                    r_table[(i, j)]["E"] = None
                else:
                    r_table[(i, j)]["E"] = self.environment[i, j+1]

                # to South
                if i+1 >= self.environment_shape[0]:
                    r_table[(i, j)]["S"] = None
                else:
                    r_table[(i, j)]["S"] = self.environment[i+1, j]

                # to West
                if j-1 < 0:
                    r_table[(i, j)]["W"] = None
                else:
                    r_table[(i, j)]["W"] = self.environment[i, j-1]
        return r_table

    def update_q(self):
        action = self.select_action()
        if action == "N":
            pass
        elif action == "E":
            pass
        elif action == "S":
            pass
        elif action == "W":
            pass

    def select_action(self):
        possible_action = ["N", "E", "S", "W"]
        action = possible_action[random.randint(0, len(possible_action))]
        return action

    def q_formula(self):
        q_value = self.r_table + (self.gamma * max(self.q_table))
