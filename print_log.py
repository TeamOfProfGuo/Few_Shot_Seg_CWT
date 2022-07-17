
import os
import argparse
from collections import defaultdict
from src.util import AverageMeter


parser = argparse.ArgumentParser(description='Summarize log')
parser.add_argument('--fpath', type=str, required=True, help='path of log file')
parser.add_argument('--start', type=int, default=0, help='where to start summarizing')
parser.add_argument('--end', type=int, default=-1, help='where to end summarizing')
args = parser.parse_args()

fpath = './results/' + args.fpath + '/log.txt'
assert os.path.exists(fpath), 'can not find the file'

eval_cnt =0
dt = defaultdict(AverageMeter)
max_iou = (0.0, 0.0, 0.0)

f = open(fpath, 'r')
line = f.readline()
while line:

    if "mIoU---Val result" in line:
        eval_cnt += 1
        if eval_cnt >= args.start:
            print(line)
            mIoU0, mIoU1, mIoU = float(line[25:31]), float(line[39:45]), float(line[51:57])
            dt['iou0'].update(mIoU0)
            dt['iou1'].update(mIoU1)
            dt['iou'].update(mIoU)
            if mIoU>max_iou[-1]:
                max_iou = (mIoU0, mIoU1, mIoU)

    if args.end >= 1 and eval_cnt>=args.end:
        break

    line = f.readline()

f.close()

print(f"avg mIOU0: {dt['iou0'].avg:.4f}, mIOU1: {dt['iou1'].avg:.4f}, mIOU: {dt['iou'].avg:.4f}, runs: {dt['iou0'].count}\n")
print('max iou ' + str(max_iou))