import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse
prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

from lib.test.tracker.mcitrack import MDTRACK
import lib.test.parameter.mdtrack as rgbt_params
import multiprocessing
import torch
from lib.train.dataset.depth_utils import get_x_frame
import time


def genConfig(seq_path, set_type):

    RGB_img_list = sorted([seq_path + '/RGB/' + p for p in os.listdir(seq_path + '/RGB') if p.endswith(".jpg")])
    T_img_list = sorted([seq_path + '/TIR/' + p for p in os.listdir(seq_path + '/TIR') if p.endswith(".jpg")])

    RGB_gt = np.loadtxt(seq_path + '/init.txt', delimiter=' ')
    T_gt = np.loadtxt(seq_path + '/init.txt', delimiter=' ')

    return RGB_img_list, T_img_list, RGB_gt, T_gt


def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=300, debug=0, script_name='prompt',save_path=''):
    if 'LasHeR' or 'RGBT234' in dataset_name:
        task_index = [0]
    
    if 'VTUAV' in dataset_name:
        seq_txt = seq_name.split('/')[1]
    else:
        seq_txt = seq_name
    # save_name = '{}_ep{}'.format(yaml_name, epoch)
    save_name = '{}'.format(yaml_name)
    save_path = os.path.join(save_path, seq_txt + '.txt')

    save_folder = os.path.dirname(save_path)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
 
    if os.path.exists(save_path):
        print(f'-1 {seq_name}')
        return
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    params = rgbt_params.parameters(yaml_name, epoch)
    params.debug = 0
    # sttrack = STTrack(params)  
    mdtrack = MDTRACK(params,dataset_name,task_index)
    tracker = MDTrack_RGBT(tracker=mdtrack)
    # tracker = STTRACK_RGBT(tracker=sttrack)

    seq_path = seq_home + '/' + seq_name
    print('——————————Process sequence: '+seq_name +'——————————————')
    RGB_img_list, T_img_list, RGB_gt, T_gt = genConfig(seq_path, dataset_name)
    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)

    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)

    result[0] = np.copy(RGB_gt[0])

    toc = 0
    for frame_idx, (rgb_path, T_path) in enumerate(zip(RGB_img_list, T_img_list)):
        tic = cv2.getTickCount()
        if frame_idx == 0:
            # initialization
            image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
            tracker.initialize(image, RGB_gt.tolist())  # xywh
        elif frame_idx > 0:
            # track
            image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
            region, confidence = tracker.track(image)  # xywh
            result[frame_idx] = np.array(region)
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    if not debug:
        np.savetxt(save_path, result)

    print('{} , fps:{}'.format(seq_name, frame_idx / toc))


class MDTrack_RGBT(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        
        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        '''TRACK'''
        # print(img_RGB.shape)
        # print(img_RGB.sum())
        if img_RGB.sum() < 5 :
            return -1,-1
        elif img_RGB[:3].sum() < 5 and img_RGB[3:].sum() > 5:
            img_RGB[:3] = img_RGB[3:6].copy()
        elif img_RGB[:3].sum() > 5 and img_RGB[3:].sum() < 5:
            img_RGB[3:] = img_RGB[:3].copy()
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
       
        return pred_bbox, pred_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on RGBT dataset.')
    parser.add_argument('--script_name', type=str, default='mdtrack', help='Name of tracking method(ostrack, prompt, ftuning).')
    parser.add_argument('--yaml_name', type=str, default='mdtrack_b224_lasher', help='Name of tracking method.')  # vitb_256_mae_ce_32x4_ep300 vitb_256_mae_ce_32x4_ep60_prompt_i32v21_onlylasher_rgbt
    parser.add_argument('--dataset_name', type=str, default='LasHeR', help='Name of dataset (GTOT,RGBT234,LasHeR,VTUAVST,VTUAVLT).')
    parser.add_argument('--threads', default=1, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=1, type=int, help='Number of gpus')
    parser.add_argument('--epoch', default='', type=str, help='epochs of ckpt')
    parser.add_argument('--mode', default='sequential', type=str, help='sequential or parallel')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', default='', type=str, help='specific video name')
    parser.add_argument('--save_path', default='/disk3/wsl_tmp/Workspace210/MDTrack/save_path', type=str, help='save_path')
    parser.add_argument('--data_path', default='/disk3/wsl_tmp/Workspace210/test_data', type=str, help='save_path')


    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    # path initialization
    seq_list = None

    seq_home = args.data_path
    seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
    seq_list.sort()


    start = time.time()
    if args.mode == 'parallel':
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name, args.save_path) for s in seq_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)
    else:
        seq_list = [args.video] if args.video != '' else seq_list
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name, args.save_path) for s in seq_list]
        for seqlist in sequence_list:
            run_sequence(*seqlist)
    print(f"Totally cost {time.time()-start} seconds!")
