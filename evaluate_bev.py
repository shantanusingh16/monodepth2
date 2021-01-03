from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import imageio

import torch
from torch.utils.data import DataLoader

from utils import readlines
from options import MonodepthOptions
import datasets
import networks

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def gen_load_bev(opt):
    """Evaluates a pretrained model using a specified test set
    """

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "train_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "bev.pth")

        encoder_dict = torch.load(encoder_path)

        dataset = datasets.ScanNetDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 1, is_train=False)
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        bev_decoder = networks.BEVDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

        bev_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        bev_decoder.cuda()
        bev_decoder.eval()

        pred_bevs = []
        filepaths = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                filepaths.extend(data["filepath"])

                output = bev_decoder(encoder(input_color))

                # pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_bev = output[("bev", 0)]
                pred_bev = pred_bev.cpu()[:, 0].numpy()
                pred_bev = (pred_bev * 255).astype(np.uint8)

                pred_bevs.append(pred_bev)

        pred_bevs = np.concatenate(pred_bevs)

    else:
        # todo: Load predictions from file
        print("-> Loading predictions from np arrays")
        pred_bevs = np.load('')
        filepaths = np.load('')

    return pred_bevs, filepaths


if __name__ == "__main__":
    options = MonodepthOptions()
    pred_bevs, filepaths = gen_load_bev(options.parse())
    output_dir = 'data/pred_bevs'
    os.makedirs(output_dir, exist_ok=True)
    for i, filepath in enumerate(filepaths):
        folder, fileidx = filepath.split()
        output_path = os.path.join(output_dir, folder, '{}.png'.format(fileidx))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.imwrite(output_path, pred_bevs[i, :, :])
