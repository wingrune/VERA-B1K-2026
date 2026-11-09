"""Official WebsocketPolicyServer adapter. This process never constructs a simulator/environment."""
import argparse,sys,types
stub=types.ModuleType('omnigibson.envs.lerobot_data_wrapper')
class UnavailableRecording:
    def __init__(self,*args,**kwargs):raise RuntimeError('Optional LeRobot recorder unavailable')
stub.LeRobotDataWrapper=stub.LeRobotPlaybackWrapper=UnavailableRecording
sys.modules[stub.__name__]=stub
from observation_policy import Policy
from omnigibson.eval.utils.network_utils import WebsocketPolicyServer
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18013);p.add_argument('--host',default='127.0.0.1');a=p.parse_args()
WebsocketPolicyServer(Policy(),host=a.host,port=a.port,metadata={'policy':'organizing_school_stuff','action_dim':23}).serve_forever()
