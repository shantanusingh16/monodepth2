from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import random
import torch
from torchvision import transforms

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset
from PIL import Image
from scipy.spatial.transform import Rotation


class HabitatDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(HabitatDataset, self).__init__(*args, **kwargs)

        self.full_res_shape = (640, 480)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = False  #self.is_train and random.random() > 0.5 #todo implement flip for get_pose

        line = self.filenames[index].split()
        folder = line[0]

        frame_index = int(line[1])

        inputs["filepath"] = self.filenames[index]

        for i in self.frame_idxs:
            inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, None, do_flip)
            inputs[("pose", i)] = torch.from_numpy(self.get_pose(folder, frame_index + i, do_flip))

        # adjusting intrinsics to match each scale in the pyramid
        K = self.get_K(folder, np.array(inputs[("color", 0, -1)]).shape)
        for scale in range(self.num_scales):
            K = K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, None, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs

    def get_K(self, folder, img_shape):
        w, h = self.full_res_shape

        intrinsics = np.array([[640 / w, 0., 320 / w, 0.],
                               [0., 480 / h, 240 / h, 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]], dtype="float32")

        return intrinsics

    def check_depth(self):
        if len(self.filenames) == 0:
            return False
        line = self.filenames[0].split()
        folder = line[0]
        frame_index = int(line[1])

        depth_path = os.path.join(
            self.data_path,
            folder,
            "0/left_depth",
            "{}.png".format(frame_index))

        return os.path.isfile(depth_path)

    def get_image_path(self, folder, frame_index):
        image_path = os.path.join(
            self.data_path,
            folder,
            "0/left_rgb",
            "{}.jpg".format(frame_index))
        return image_path

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_pose(self, folder, frame_index, do_flip):
        pose_path = os.path.join(
            self.data_path,
            folder,
            "0/pose",
            "{}.npy".format(frame_index))

        pose = np.load(pose_path, allow_pickle=True).tolist()

        T = np.eye(4)
        rot = pose['rotation']
        T[:3, :3] = Rotation.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
        T[:3, 3] = pose['position']

        if do_flip:
            pass #todo implement flip for pose

        return T

    def get_depth(self, folder, frame_index, side, do_flip):
        depth_path = os.path.join(
            self.data_path,
            folder,
            "0/left_depth",
            "{}.png".format(frame_index))

        depth = np.array(Image.open(depth_path)).astype(np.float32) / 1000.0
        if do_flip:
            depth = np.fliplr(depth)
        return depth