import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from server.routing import TASK_IDS


class Model:
    def __init__(self):
        self.calls = 0

    def reset(self):
        pass

    def infer(self, obs):
        self.calls += 1
        actions = np.zeros((32, 23), np.float32)
        actions[:, 3] = self.calls
        actions[:, 4] = TASK_IDS.get(obs['prompt'], -1)
        actions[:, [14, 22]] = 1
        return {'actions': actions}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--plain', action='store_true')
    parser.add_argument('--fail', choices=('raise', 'nan', 'hang'), help='plain backend whose every act raises, returns NaN or hangs')
    args = parser.parse_args()
    if args.plain or args.fail:
        from server.protocol import serve

        class Plain:
            def reset(self):
                pass

            def act(self, obs):
                if args.fail == 'raise':
                    raise RuntimeError('fake backend failure')
                if args.fail == 'hang':
                    time.sleep(3600)
                return np.full((1, 23), np.nan if args.fail == 'nan' else 7, np.float32)

        serve(Plain(), args.port)
    else:
        from solutions.comet2.wrapper import B1KPolicyWrapper
        from solutions.comet2.serving import WebsocketPolicyServer
        WebsocketPolicyServer(B1KPolicyWrapper(Model()), args.port).serve_forever()
