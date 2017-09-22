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
from datetime import datetime
from malmo_tabular_q import TabularQLearnerAgent
from malmopy.environment.malmo.malmo import MalmoStateBuilder
import sys
import json
from time import sleep
import numpy as np
from malmopy.environment.malmo import MalmoEnvironment, MalmoALEStateBuilder

try:
    from malmopy.visualization.tensorboard import TensorboardVisualizer
    from malmopy.visualization.tensorboard.cntk import CntkConverter
except ImportError:
    print('Cannot import tensorboard, using ConsoleVisualizer.')
    from malmopy.visualization import ConsoleVisualizer


MALMO_MAZE_FOLDER = 'results/baselines/malmo/maze/dqn/cntk'
ROOT_FOLDER = 'results/baselines/tab_q/%s'
EMPTY_FRAME = np.zeros((84, 84), dtype=np.float32)

class MazeTabQStateBuilder(MalmoStateBuilder):
    """
    This class builds a state made up of the agent's location and orientation
    """

    def __init__(self):
        pass

    def build(self, environment):
        """
        """
        assert isinstance(environment,
                          MazeEnvironment), 'environment is not a MalmoMaze Environment instance'

        world_obs = environment.world_observations
        if world_obs is None:
            return None

        current_x = world_obs.get(u'XPos', 0)
        current_z = world_obs.get(u'ZPos', 0)
        yaw = world_obs.get(u'Yaw', 0)

        location= str(current_x) + ":" + str(current_z)
        orientation=yaw

        return (location, orientation)



class MazeEnvironment(MalmoEnvironment):
    MAZE_ACTIONS = ["move 1", "turn 1", "turn -1"]

    def __init__(self, mission, remotes, recording_path=None):
        super(MazeEnvironment, self).__init__(mission, MazeEnvironment.MAZE_ACTIONS, remotes,
                                              recording_path=recording_path)

        self._builder = MazeTabQStateBuilder()

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
#    visualizer.add_entry(step, '%s/max.reward' % tag, max(rewards))
 #   visualizer.add_entry(step, '%s/min.reward' % tag, min(rewards))
    visualizer.add_entry(step, '%s/actions per episode' % tag, len(rewards)-1)


def run_maze_learner(mission, clients):

    if 'malmopy.visualization.tensorboard' in sys.modules:
        visualizer = TensorboardVisualizer()
        visualizer.initialize(logdir, None)

    else:
        visualizer = ConsoleVisualizer()

    env = MazeEnvironment(mission, [str.split(client, ':') for client in clients])
    env.recording = False

    agent = TabularQLearnerAgent("rand",3)

            #taking random actions
    EPOCH_SIZE = 250000
    max_training_steps = 50 * EPOCH_SIZE
    state = env.reset()
    reward = 0
    agent_done = False
    viz_rewards = []
    for step in range(1, max_training_steps + 1):

                # check if env needs reset
        if env.done:
            visualize_training(visualizer, step, viz_rewards)
            agent.inject_summaries(step)
            viz_rewards = []
            state = env.reset()

                # select an action
        action = agent.act(step, state, is_training=True)
        if type(action) == int:
            print('ACTION BEING TAKEN: ', action)
        else:
            print('ACTION BEING TAKEN: ', np.asscalar(action))

                # take a step
        old = state
        state, reward, agent_done = env.do(action)
        agent.observe(old, action, state, reward, env.done)
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

    logdir = ROOT_FOLDER % (datetime.utcnow().isoformat())
    if TENSORBOARD_AVAILABLE:
        visualizer = TensorboardVisualizer()
        visualizer.initialize(logdir, None)
        print('Starting tensorboard ...')
        p = Popen(['tensorboard', '--logdir=results', '--port=%d' % args.port])

    else:
        visualizer = ConsoleVisualizer()
    # Load mission an register agent
    with open(path.join(mission_dir, MISSION), 'r') as f:
        mission = f.read()
        run_maze_learner(mission, args.clients)

