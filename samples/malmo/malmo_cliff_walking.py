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

from malmopy.agent.gui import GuiAgent, ARROW_KEYS_MAPPING
from malmopy.environment.malmo import DiscreteMalmoEnvironment
from malmopy.experiment import SingleAgentExperiment


class CliffEnvironment(DiscreteMalmoEnvironment):
    def __init__(self, mission, actions, remotes):
        super(CliffEnvironment, self).__init__(mission, actions, remotes)

    def state(self):
        pass


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-d', '--directory', type=str, help='Missions directory', default='../..')
    arg_parser.add_argument('clients', nargs='+', help='Minecraft clients endpoints (ip(:port)?)+')
    args = arg_parser.parse_args()

    mission_dir = path.abspath(args.directory)
    clients = [str.split(client, ':') for client in args.clients]

    with open(path.join(mission_dir, 'cliff_walking_1.xml'), 'r') as f:
        mission = f.read()

        env = CliffEnvironment(mission, list(ARROW_KEYS_MAPPING.values()), clients)
        env.recording = True

        agent = GuiAgent("GUI agent", env, list(ARROW_KEYS_MAPPING.keys()))
        exp = SingleAgentExperiment("Malmo Cliff Walking", agent, env, 100, 30, looper=agent.looper)

        exp.run()
