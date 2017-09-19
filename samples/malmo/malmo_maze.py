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

import numpy as np

from malmopy.agent import TemporalMemory, DQNAgent, LinearEpsilonGreedyExplorer
from malmopy.environment.malmo import DiscreteMalmoEnvironment
from malmopy.experiment import SingleAgentExperiment
from malmopy.model.cntk import DeepQNeuralNetwork
from malmopy.visualization.tensorboard import TensorboardVisualizer
from malmopy.visualization.tensorboard.cntk import CntkConverter

MALMO_MAZE_FOLDER = 'results/baselines/malmo/maze/dqn/cntk'
EMPTY_FRAME = np.zeros((84, 84), dtype=np.float32)


class MazeEnvironment(DiscreteMalmoEnvironment):
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


def run_maze_learner(mission, clients):
    with TensorboardVisualizer() as visualizer:
        env = MazeEnvironment(mission, [str.split(client, ':') for client in clients])
        env.recording = False

        explorer = LinearEpsilonGreedyExplorer(1, 0.1, 10000)
        model = DeepQNeuralNetwork((4, 84, 84), (env.available_actions,), momentum=0, visualizer=visualizer)
        memory = TemporalMemory(50000, model.input_shape[1:], model.input_shape[0], False)

        agent = DQNAgent("Maze DQN Agent", env.available_actions, model, memory, explorer=explorer,
                         visualizer=visualizer)

        exp = SingleAgentExperiment("Malmo Cliff Walking", agent, env, 500000, warm_up_timesteps=500,
                                    visualizer=visualizer)
        exp.episode_end += on_episode_end

        visualizer.initialize(MALMO_MAZE_FOLDER, model, CntkConverter())

        with Popen(['tensorboard', '--logdir=%s' % path.join(MALMO_MAZE_FOLDER, path.pardir), '--port=6006']):
            exp.run()


if __name__ == '__main__':
    # Look for the mission
    MISSION = '5-grid_2-lava_0x-0z-0yaw-agent_0-1-0-2-lava.xml'

    arg_parser = ArgumentParser()
    arg_parser.add_argument('-d', '--directory', type=str, help='Missions directory', default='../..')
    arg_parser.add_argument('clients', nargs='+', help='Minecraft clients endpoints (ip(:port)?)+')
    args = arg_parser.parse_args()

    mission_dir = path.abspath(args.directory)

    # Load mission an register agent
    with open(path.join(mission_dir, MISSION), 'r') as f:
        mission = f.read()
        run_maze_learner(mission, args.clients)
