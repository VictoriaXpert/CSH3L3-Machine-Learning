import numpy as np
import random

class QLearning :
    def __init__(self, gamma, load_reward):
        self.gamma = gamma
        self.load_reward = load_reward
        self.reward = np.zeros((10,10), dtype=int)
        self.q = np.zeros((10,10), dtype=int)

    def initialize_reward(self):
        load_reward = np.loadtxt(self.load_reward, dtype='i', delimiter='\t')
        self.reward = load_reward[::-1]

    def initialize_q(self):
        self.q = np.zeros((10,10), dtype=int)

    def next_action(self, state):
        next = []
        for i in range(self.reward[state].size):
            if (self.reward[state,i] != -1):
                next.append(i)
        return random.choice(next)

    def max_q(self, next):
        list = []
        value = []
        for i in range(self.reward[next].size):
            if (self.reward[next, i] != -1):
                list.append(i)
        for j in range(len(list)):
            value.append(list[j])
        return max(value)

    def update_q(self, state, action):
        self.q[state, action] = self.reward[state, action] + self.gamma * self.max_q(action)

    def main(self):
        current = random.randint(0, 9)
        print(current)


# MAIN PROGRAM #
file_reward = "DataTugasML3.txt"
qlearn = QLearning(0.8, file_reward)
qlearn.initialize_reward()
qlearn.initialize_q()
for i in range(100):
	qlearn.main()

print(qlearn.reward)
print(qlearn.q)
print(qlearn.max_q)
