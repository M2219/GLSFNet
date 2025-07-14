import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import os
import copy
import random
from glob import glob
import cv2
from PIL import Image
import torchvision.transforms as transforms
from utils import frame_utils
from typing import List, Tuple
import torchvision
import numbers

class RandomCrop(object):
    """Randomly crop images"""

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, target, valid_gt, depth_gt):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w < tw:
            tw = w
        if h < th:
            th = h

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs[0] = inputs[0][y1 : y1 + th, x1 : x1 + tw]
        inputs[1] = inputs[1][y1 : y1 + th, x1 : x1 + tw]
        inputs[2] = inputs[2][y1 : y1 + th, x1 : x1 + tw]
        return inputs, target[y1 : y1 + th, x1 : x1 + tw], valid_gt[y1 : y1 + th, x1 : x1 + tw], depth_gt[y1 : y1 + th, x1 : x1 + tw]


class Compose(object):
    """Composes several co_transforms together."""

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target, valid, depth):
        for t in self.co_transforms:
            input, target, valid, depth = t(input, target, valid, depth)
        return input, target, valid, depth


def read_all_lines(filename: str) -> List[str]:
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines

def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

def get_transform_sparse():
    return transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )


class StereoDataset(data.Dataset):
    def __init__(self, datapath_im=None, datapath_s=None, datapath_gt=None,  list_filename=None, kitti_completion=False, is_vkitti2=False, train_status=False):

        self.kitti_completion = kitti_completion
        self.is_vkitti2 = is_vkitti2
        self.train_status = train_status

        self.datapath_im = datapath_im
        self.datapath_s = datapath_s
        self.datapath_gt = datapath_gt

        self.left_filenames, self.right_filenames, self.s_filenames, self.disp_filenames = self.load_path(list_filename)

        self.train_status = train_status

    def __getitem__(self, index):

        if self.is_vkitti2:
            disparity, valid, conversion_rate = self.disp_load_vkitti2(os.path.join(self.datapath_gt, self.disp_filenames[index]))

        if self.kitti_completion:
            depth_gt, disparity_gt, valid_gt, conversion_rate_gt = self.disp_load_kitti_completion(os.path.join(self.datapath_gt, self.disp_filenames[index]))
            depth, disparity, valid, conversion_rate = self.disp_load_kitti_completion(os.path.join(self.datapath_s, self.s_filenames[index]))

        left_img = self.load_image(os.path.join(self.datapath_im, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath_im, self.right_filenames[index]))

        if self.train_status:
            th, tw = 256, 512
            random_brightness = np.random.uniform(0.5, 2.0, 2)
            random_gamma = np.random.uniform(0.8, 1.2, 2)
            random_contrast = np.random.uniform(0.8, 1.2, 2)
            random_saturation = np.random.uniform(0, 1.4, 2)

            left_img = torchvision.transforms.functional.adjust_brightness(
                left_img, random_brightness[0]
            )
            right_img = torchvision.transforms.functional.adjust_brightness(
                right_img, random_brightness[1]
            )

            left_img = torchvision.transforms.functional.adjust_gamma(
                left_img, random_gamma[0]
            )
            right_img = torchvision.transforms.functional.adjust_gamma(
                right_img, random_gamma[1]
            )

            left_img = torchvision.transforms.functional.adjust_contrast(
                left_img, random_contrast[0]
            )
            right_img = torchvision.transforms.functional.adjust_contrast(
                right_img, random_contrast[1]
            )

            left_img = torchvision.transforms.functional.adjust_saturation(
                left_img, random_saturation[0]
            )
            right_img = torchvision.transforms.functional.adjust_saturation(
                right_img, random_saturation[1]
            )

            left_img = np.array(left_img)
            right_img = np.array(right_img)
            disparity = np.array(disparity)

            angle = 0
            px = 0
            if np.random.binomial(1, 0.5):
                angle = 0.05
                px = 1

            co_transform = Compose([RandomCrop((th, tw)),])
            augmented, disparity_gt, valid_gt, depth_gt = co_transform([left_img, right_img, disparity], disparity_gt, valid_gt, depth_gt)
            left_img = augmented[0]
            right_img = augmented[1]
            disparity = augmented[2]

            right_img.flags.writeable = True
            if np.random.binomial(1, 0.5):
                sx = int(np.random.uniform(35, 100))
                sy = int(np.random.uniform(25, 75))
                cx = int(np.random.uniform(sx, right_img.shape[0] - sx))
                cy = int(np.random.uniform(sy, right_img.shape[1] - sy))
                right_img[cx - sx : cx + sx, cy - sy : cy + sy] = np.mean(
                    np.mean(right_img, 0), 0
                )[np.newaxis, np.newaxis]

            disparity_gt = np.ascontiguousarray(disparity_gt, dtype=np.float32)
            valid_gt = np.ascontiguousarray(valid_gt, dtype=np.float32)
            depth_gt = np.ascontiguousarray(depth_gt, dtype=np.float32)

            disparity_gt_low = []
            for res in [2, 4, 8, 16]:
                disparity_gt_low.append(cv2.resize(disparity_gt, (tw//res, th//res), interpolation=cv2.INTER_NEAREST))

            processed = get_transform()
            processed_s = get_transform_sparse()
            left_img = processed(left_img)
            right_img = processed(right_img)
            disparity = processed_s(disparity)

            return {
                "left": left_img,
                "right": right_img,
                "disparity": disparity,
                "disparity_gt": disparity_gt,
                "disparity_gt_low": disparity_gt_low,
                "depth_gt": depth_gt,
                "conversion_rate": conversion_rate,
                "valid": valid_gt,
            }


        else:
            w, h = right_img.size

            processed = get_transform()
            processed_s = get_transform_sparse()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()
            disparity = processed_s(disparity).numpy()

            top_pad = 384 - h
            right_pad = 1248 - w

            assert top_pad > 0 and right_pad > 0
            left_img = np.lib.pad(
                left_img,
                ((0, 0), (top_pad, 0), (0, right_pad)),
                mode="constant",
                constant_values=0,
            )
            right_img = np.lib.pad(
                right_img,
                ((0, 0), (top_pad, 0), (0, right_pad)),
                mode="constant",
                constant_values=0,
            )

            disparity = np.lib.pad(
                disparity,
                ( (0, 0), (top_pad, 0), (0, right_pad)),
                mode="constant",
                constant_values=0,
            )

            depth_gt = np.lib.pad(
                depth_gt,
                ((top_pad, 0), (0, right_pad)),
                mode="constant",
                constant_values=0,
            )

            disparity_gt = np.lib.pad(
                disparity_gt,
                ((top_pad, 0), (0, right_pad)),
                mode="constant",
                constant_values=0,
            )

            valid_gt = np.lib.pad(
                valid_gt,
                ((top_pad, 0), (0, right_pad)),
                mode="constant",
                constant_values=0,
            )


            return {
                "left": left_img,
                "right": right_img,
                "disparity": disparity,
                "disparity_gt": disparity_gt,
                "depth_gt": depth_gt,
                "conversion_rate": conversion_rate,
                "valid": valid_gt,
            }


    def __len__(self) -> int:
        return len(self.left_filenames)

    def load_path(self, list_filename: str):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp = [x[2] for x in splits]
        if len(splits[0]) == 2:
            return left_images, right_images, None
        else:
            disp_gt_images = [x[3] for x in splits]
            return left_images, right_images, disp, disp_gt_images

    def load_image(self, filename):
        return Image.open(filename).convert("RGB")

    def disp_load_kitti_completion(self, filename):

        data = Image.open(filename)
        w, h = data.size
        baseline = np.array(0.54, dtype=np.float32)
        width_to_focal = dict()
        width_to_focal[1242] = np.array(721.5377, dtype=np.float32)
        width_to_focal[1241] = np.array(718.856, dtype=np.float32)
        width_to_focal[1224] = np.array(707.0493, dtype=np.float32)
        width_to_focal[1226] = np.array(708.2046, dtype=np.float32)
        width_to_focal[1238] = np.array(718.3351, dtype=np.float32)

        data = np.array(data, dtype=np.float32) / 256.  # depth,m
        depth = np.copy(data)

        conversion_rate = width_to_focal[w] * baseline

        data[data > 0.01] = conversion_rate / (data[data > 0.01])  # disp
        data[data < 0.01] = 0
        valid_hint = (data > 0.1)
        data = data * valid_hint
        return depth, data, valid_hint, conversion_rate

    def disp_load_vkitti2(self, filename):
        depth = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = (depth / 100).astype(np.float32)

        valid = (depth > 0) & (depth < 655)
        focal_length = 725.0087
        baseline = 0.532725

        disp = baseline * focal_length / depth

        disp[~valid] = 0.

        return disp, valid, baseline * focal_length

"""
class VKITTI2(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/vkitti2', image_set='training',
                 args=None, single_scena=True):
        super(VKITTI2, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI, args=args,
                                      kitti_completion=True, is_vkitti2=True)

        assert os.path.exists(root)

        train_seq_list = ['Scene01', 'Scene02']
        test_seq_list = ['Scene06', 'Scene18', 'Scene20']

        scena = '15-deg-left' if single_scena else '*'

        use_seqs = train_seq_list if image_set == 'training' else test_seq_list

        img_left_list = []
        img_right_list = []
        disp_gt_list = []

        for seq in use_seqs:
            img_left_list += sorted(glob(root + f'/{seq}/{scena}/frames/rgb/Camera_0/rgb*.jpg'))

        for samp in img_left_list:
            img_right_list.append(samp.replace('Camera_0', 'Camera_1'))
            disp_gt_list.append(samp.replace('rgb', 'depth').replace('jpg', 'png'))

        if image_set == 'val':
            state = np.random.get_state()
            np.random.seed(1000)
            val_idxs = set(np.random.permutation(len(img_left_list))[:300])
            np.random.set_state(state)

            for idx, (left_img, right_img, disp) in enumerate(
                    zip(img_left_list, img_right_list, disp_gt_list)):
                if idx in val_idxs:
                    self.image_list += [[left_img, right_img]]
                    self.disparity_list += [disp]
        else:
            for idx, (left_img, right_img, disp) in enumerate(
                    zip(img_left_list, img_right_list, disp_gt_list)):
                self.image_list += [[left_img, right_img]]
                self.disparity_list += [disp]



"""
