import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

from datasets import find_dataset_def
from utils import *
from datasets.data_io import save_pfm, write_red_cam
import matplotlib.pyplot as plt

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth for whu-omvs test set')
parser.add_argument('--model', default='adamvs', help='select model from [msrednet, adamvs]')
parser.add_argument('--dataset', default='predict_oblique', help='select dataset')
parser.add_argument('--data_folder', default='H:/prepared/meitan_oblique/predict/source', help='test datapath')
parser.add_argument('--output_folder', default='H:/prepared/meitan_oblique/workspace_rednet/MVS', help='output dir')
parser.add_argument('--loadckpt', default='./checkpoints/adamvs_whuomvs/model_000014_0.1409.ckpt', help='load a specific checkpoint')

# input parameters
parser.add_argument('--view_num', type=int, default=5, help='Number of images (1 ref image and view_num - 1 view images).')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--max_w', type=int, default=3712, help='Maximum image width')
parser.add_argument('--max_h', type=int, default=5504, help='Maximum image height')
parser.add_argument('--min_interval', type=float, default=0.1, help='min_interval in the bottom stage')

parser.add_argument('--fext', type=str, default='.jpg', help='Type of images.')
parser.add_argument('--normalize', type=str, default='mean', help='methods of center_image, mean[mean var] or standard[0-1].') # attention: CasMVSNet [mean var];; CasREDNet [0-255]
parser.add_argument('--resize_scale', type=float, default=0.5, help='output scale for depth and image (W and H)')
parser.add_argument('--sample_scale', type=float, default=1, help='Downsample scale for building cost volume (W and H)')
parser.add_argument('--interval_scale', type=float, default=1, help='the number of depth values')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--display', default=True, help='display depth images')

# Cascade parameters
parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')
parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)


# run MVS model to save depth maps and confidence maps
def predict_depth():
    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)

    test_dataset = MVSDataset(args.data_folder, args.view_num, args)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=1, drop_last=False)

    # build model
    model = None
    if args.model == 'msrednet':
        from models.msrednet import Infer_CascadeREDNet
        model = Infer_CascadeREDNet(num_depth=args.numdepth, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                              depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                              share_cr=args.share_cr,
                              cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch])

    elif args.model == 'adamvs':
        from models.adamvs import Infer_AdaMVSNet
        model = Infer_AdaMVSNet(num_depth=args.numdepth, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                               depth_intervals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                               share_cr=args.share_cr,
                               cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch])
    else:
        raise Exception("{}? Not implemented yet!".format(args.model))

    model = nn.DataParallel(model)
    model.cuda()

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
    model.eval()

    with torch.no_grad():
        # create output folder
        output_folder = args.output_folder
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        step = 0
        first_start_time = time.time()

        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            sample_cuda = tocuda(sample)

            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            outputs = tensor2numpy(outputs)
            depth_est = outputs["depth"]
            photometric_confidence = outputs["photometric_confidence"]
            duration = time.time()

            # save results
            depth_est = np.float32(np.squeeze(tensor2numpy(depth_est)))
            prob = np.float32(np.squeeze(tensor2numpy(photometric_confidence)))
            ref_image = np.squeeze(tensor2numpy(sample["outimage"]))
            ref_cam = np.squeeze(tensor2numpy(sample["outcam"]))
            #  aerial dataset
            vid = sample["out_view"][0]
            name = sample["out_name"][0]
            ref_path = np.squeeze(sample["ref_image_path"])

            # paths
            output_folder2 = output_folder + ('/%s/' % vid)
            if not os.path.exists(output_folder2 + '/color/'):
                os.mkdir(output_folder2)
                os.mkdir(output_folder2 + '/color/')

            init_depth_map_path = output_folder2 + ('/%s_init.pfm' % name)
            prob_map_path = output_folder2 + ('/%s_prob.pfm' % name)
            out_ref_image_path = output_folder2 + ('/%s.jpg' % name)
            out_ref_cam_path = output_folder2 + ('/%s.txt' % name)

            if args.display:
                # color output
                size1 = len(depth_est)
                size2 = len(depth_est[1])
                e = np.ones((size1, size2), dtype=np.float32)
                out_init_depth_image = e * 36000 - depth_est
                color_depth_map_path = output_folder2 + ('/color/%s_init.png' % name)
                color_prob_map_path = output_folder2 + ('/color/%s_prob.png' % name)


                for i in range(out_init_depth_image.shape[1]):
                    col = out_init_depth_image[:, i]
                    col[np.isinf(col)] = np.nan
                    col[np.isnan(col)] = np.nanmin(col) - 1
                    out_init_depth_image[:, i] = col

                plt.imsave(color_depth_map_path, out_init_depth_image, format='png')
                plt.imsave(color_prob_map_path,  np.nan_to_num(prob).clip(0, 1), format='png')

            save_pfm(init_depth_map_path, depth_est)
            save_pfm(prob_map_path, prob)
            plt.imsave(out_ref_image_path, ref_image, format='png')
            write_red_cam(out_ref_cam_path, ref_cam, ref_path)

            del outputs, sample_cuda

            step = step + 1
            save_tesult_time = time.time()
            print('depth inference {} finished, image {} finished, ({:3f}s and {:3f} sec/step)'.format(step, name, duration-start_time, save_tesult_time-duration))

        print("final, total_cnt = {}, total_time = {:3f}".format(step, time.time() - first_start_time))


if __name__ == '__main__':
    # step1. save all the depth maps and the masks in outputs directory
    predict_depth()

