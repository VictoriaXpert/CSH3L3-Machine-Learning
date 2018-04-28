import numpy as np
import fun
import random
import sys


class QLearning:
    """
    Class Q-Learning implementing the Q-Learning Algorithm for grid world.
    """

    def __init__(self, environment, gamma=0.9):
        """Inisialisasi variable yang dibutuhkan."""
        self.environment = environment
        self.environment_shape = environment.shape
        self.gamma = gamma
        self.q_table = self.build_q_table()
        self.r_table = self.build_r_table()
        self.current_state = (0, 0)
        self.finish_state = (9, 9)

    def build_q_table(self):
        """Membangun Q table sejumlah grid dikali jumlah action untuk menyimpan hasil learning."""
        q_table = {}
        for i in range(self.environment_shape[0]):
            for j in range(self.environment_shape[1]):
                q_table[(i, j)] = {"N": 0, "E": 0, "S": 0, "W": 0}
        return q_table

    def build_r_table(self):
        """Membangun table rewards sejumlah grid dikali jumlah action untuk menyimpan hasil learning."""
        r_table = {}
        for i in range(self.environment_shape[0]):
            for j in range(self.environment_shape[1]):
                r_table[(i, j)] = {}

                # to North
                if i-1 < 0:
                    r_table[(i, j)]["N"] = None
                else:
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

    def getNextState(self, action):
        if action == "N":
            next_state = (self.current_state[0]-1, self.current_state[1])
        elif action == "E":
            next_state = (self.current_state[0], self.current_state[1]+1)
        elif action == "S":
            next_state = (self.current_state[0]+1, self.current_state[1])
        elif action == "W":
            next_state = (self.current_state[0], self.current_state[1]-1)
        return next_state

    def update_q(self):
        """Mengupdate nilai Q di Q table pada current state."""
        action = self.select_action()
        next_state = self.getNextState(action)
        self.q_table[self.current_state][action] = self.q_formula(
            self.current_state, next_state, action)
        self.current_state = next_state

    def getPossibleAction(self):
        possible_action = []
        for action, reward in self.r_table[self.current_state].items():
            if reward != None:
                possible_action.append(action)
        return possible_action

    def select_action(self):
        """Memilih aksi yang dapat dilakukan."""
        possible_action = self.getPossibleAction()
        action = random.sample(possible_action, 1)
        return action[0]

    def q_formula(self, current_state, next_state, action):
        """Menghitung nilai q."""
        q_value = self.r_table[current_state][action] + \
            (self.gamma * max(self.q_table[next_state].values()))
        return q_value

    def fit(self, episodes):
        """Melakukan training."""
        self.current_state = (0, 0)
        for _ in range(episodes):
            while self.current_state != self.finish_state:
                self.update_q()
            self.current_state = (random.randint(0, 9), random.randint(0, 9))

    def predict(self):
        """Mendapatkan solusi terbaik berupa jalur dan total rewards."""
        try:
            tracks = []
            self.current_state = (0, 0)
            while self.current_state != self.finish_state:
                q_actions = list(self.q_table[self.current_state].keys())
                q_rewards = list(self.q_table[self.current_state].values())

                action = q_actions[np.argmax(q_rewards)]
                self.current_state = self.getNextState(action)
                tracks.append(self.current_state)
            rewards = 0
            for track in tracks:
                rewards += self.environment[track[0], track[1]]
            return tracks, rewards
        except:
            print("Add more episodes please!")
            sys.exit(0)
