import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from lib.test.vot.mcitrack_class import run_vot_exp


print("mcitrack_b224_deepth")
run_vot_exp('mcitrack', 'mcitrack_b224_deepth', vis=False, out_conf=True, channel_type='rgbd',run_id=20,dataset_name='depthtrack')
