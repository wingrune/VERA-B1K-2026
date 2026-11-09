import argparse
import logging
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from solutions.comet2.config import CheckpointConfig, PolicyConfig, configure_runtime

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    configure_runtime()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s comet2[%(process)d] %(message)s')
    from solutions.comet2.policy_config import create_policy
    from solutions.comet2.wrapper import B1KPolicyWrapper
    from solutions.comet2.serving import WebsocketPolicyServer
    config = PolicyConfig()
    policy = create_policy(config, CheckpointConfig(path=args.checkpoint))
    WebsocketPolicyServer(B1KPolicyWrapper(policy, config), args.port, policy.metadata).serve_forever()

if __name__ == '__main__':
    main()
