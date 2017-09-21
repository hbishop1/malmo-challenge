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
from subprocess import Popen
from malmopy.agent import RandomAgent
import sys
import numpy as np
from malmopy.environment.malmo import MalmoEnvironment, MalmoALEStateBuilder

try:
    from malmopy.visualization.tensorboard import TensorboardVisualizer
    from malmopy.visualization.tensorboard.cntk import CntkConverter
except ImportError:
    print('Cannot import tensorboard, using ConsoleVisualizer.')
    from malmopy.visualization import ConsoleVisualizer


MALMO_MAZE_FOLDER = 'results/baselines/malmo/maze/dqn/cntk'
EMPTY_FRAME = np.zeros((84, 84), dtype=np.float32)


class MazeEnvironment(MalmoEnvironment):
    MAZE_ACTIONS = ["move 1", "turn 1", "turn -1"]

    def __init__(self, mission, remotes, recording_path=None):
        super(MazeEnvironment, self).__init__(mission, MazeEnvironment.MAZE_ACTIONS, remotes,
                                              recording_path=recording_path)

        self._builder = MalmoALEStateBuilder()

    @property
    def state(self):
        return self._builder(self)


def on_episode_end(experiment, nb_actions_taken, rewards):
    print("Episode %d/%d (%.2f%%) (Timesteps: %d, Warming Up: %s, Training: %s) -> Actions Taken: %d, Rewards: %.3f" %
          (experiment.episode, experiment.max_episode, (experiment.episode / float(experiment.max_episode)) * 100.,
           experiment._looper.current_timestep, experiment._looper.is_warming_up, experiment._looper.is_training,
           nb_actions_taken, rewards))


def visualize_training(visualizer, step, rewards, tag='Training'):
    visualizer.add_entry(step, '%s/reward per episode' % tag, sum(rewards))
    visualizer.add_entry(step, '%s/max.reward' % tag, max(rewards))
    visualizer.add_entry(step, '%s/min.reward' % tag, min(rewards))
    visualizer.add_entry(step, '%s/actions per episode' % tag, len(rewards)-1)


def run_maze_learner(mission, clients):

    if 'malmopy.visualization.tensorboard' in sys.modules:
        visualizer = TensorboardVisualizer()
        visualizer.initialize(logdir, None)

    else:
        visualizer = ConsoleVisualizer()

    env = MazeEnvironment(mission, [str.split(client, ':') for client in clients])
    env.recording = False

    agent = RandomAgent("rand",3,delay_between_action=1.5)

            #taking random actions
    EPOCH_SIZE = 250000
    max_training_steps = 50 * EPOCH_SIZE
    state = env.reset()
    reward = 0
    agent_done = False
    viz_rewards = []
    for step in range(1, max_training_steps + 1):

        action = agent.act(state, reward, agent_done, is_training=True)
                # check if env needs reset
        if env.done:
            visualize_training(visualizer, step, viz_rewards)
            agent.inject_summaries(step)
            viz_rewards = []
            state = env.reset()

                # select an action
        action = agent.act(state, reward, agent_done, is_training=True)
        print('ACTION BEING TAKEN: ', action)

                # take a step
        state, reward, agent_done = env.do(action)
        viz_rewards.append(reward)

        if (step % EPOCH_SIZE) == 0:
            model.save('%s-%s-dqn_%d.model' %
                        (backend, environment, step / EPOCH_SIZE))


if __name__ == '__main__':
    # Look for the mission
    MISSION = '5-grid_2-lava_0x-0z-0yaw-agent_0-1-0-2-lava.xml'

    arg_parser = ArgumentParser()
    arg_parser.add_argument('-d', '--directory', type=str, help='Missions directory', default='missions')
    arg_parser.add_argument('clients', nargs='+', help='Minecraft clients endpoints (ip(:port)?)+')
    args = arg_parser.parse_args()

    mission_dir = path.abspath(args.directory)

    # Load mission an register agent
    with open(path.join(mission_dir, MISSION), 'r') as f:
        mission = f.read()
        run_maze_learner(mission, args.clients)

