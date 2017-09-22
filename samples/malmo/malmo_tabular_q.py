# Copyright (c) 2017 Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================================================================

from argparse import ArgumentParser
from os import path
from time import sleep

import numpy as np

from malmopy.agent import BaseAgent
from malmopy.environment.malmo import MalmoEnvironment


def state_id(state):
    assert len(state) == 2, 'state has wrong shape'
    return '%s:%d' % ((state[0]), int(state[1]))

class TabularQLearnerAgent(BaseAgent):
    """
    Tabular Q-learning agent for discrete state/action spaces.
    """

    def __init__(self, name, nb_actions, epsilon=0.1, alpha=0.5, gamma=1.0):
        super(TabularQLearnerAgent, self).__init__(name, nb_actions)

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self._q_table = {}

    def act(self, step, state, is_training=False):
        """
        Take 1 action in response to the current world state
        """
#        sleep(1)
        current_s = state_id(state)
        self._ensure_state_exists(current_s)

        # select the next action
        if np.random.random() < self.epsilon:
            a = np.random.randint(0, self.nb_actions - 1)
        else:
            m = max(self._q_table[current_s])
            print('q values: ', self._q_table[current_s])
            indexes = np.where(self._q_table[current_s] == m)[0]
            a = np.random.choice(indexes, 1).squeeze()

        return a

    def observe(self, old_state, action, new_state, reward, is_terminal):
        previous_state = state_id(old_state)
        current_state = state_id(new_state)

        self._ensure_state_exists(previous_state)
        self._ensure_state_exists(current_state)

        old_q = self._q_table[previous_state][action]
        if is_terminal:
            new_q = reward
        else:
            new_q = old_q + self.alpha * (reward + self.gamma * max(self._q_table[current_state]) - old_q)

        self._q_table[previous_state][action] = new_q

    def _ensure_state_exists(self, state):
        if state not in self._q_table:
            self._q_table[state] = np.zeros(self.nb_actions, dtype=np.float32)


class CliffEnvironment(MalmoEnvironment):
    ACTIONS_SET = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]

    def __init__(self, mission, remotes):
        super(CliffEnvironment, self).__init__(mission, self.ACTIONS_SET, remotes)

    @property
    def state(self):
        return self._previous_pos[[0, 2]]


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-d', '--directory', type=str, help='Missions directory', default='../..')
    arg_parser.add_argument('clients', nargs='+', help='Minecraft clients endpoints (ip(:port)?)+')
    args = arg_parser.parse_args()

    mission_dir = path.abspath(args.directory)
    clients = [str.split(client, ':') for client in args.clients]

    with open(path.join(mission_dir, 'cliff_walking_1.xml'), 'r') as f:
        mission = f.read()

        env = CliffEnvironment(mission, clients)
        agent = TabularQLearnerAgent('Agent Q', len(CliffEnvironment.ACTIONS_SET), gamma=0.9)

        for i in range(10000):
            if env.done:
                env.reset()

            current_state = env.state
            action = agent.act(i, current_state, True)
            new_state, reward = env.do(action)
            agent.observe(current_state, action, new_state, reward, env.done)
