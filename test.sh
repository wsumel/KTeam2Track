#!/bin/sh
DIR="/disk3/wsl_tmp/Workspace210/test_data/"
OUTPUT_DIR_BASE="./tracking_results/"
ck_pth="/disk3/wsl_tmp/Workspace210/MDTrack/0225/checkpoints/train/mdtrack/mdtrack_b224_lasher/MDTrack_ep0030.pth.tar"

python ./rgbt_workspace/test_rgbt_mgpus.py --save_path "$OUTPUT_DIR_BASE" --data_path "$DIR" --epoch "$ck_pth"

python ./rgbt_workspace/linear.py --folder "$OUTPUT_DIR_BASE" --outputdir "$OUTPUT_DIR_BASE"

