import os
import numpy as np
import torch
from torch.utils.data import Dataset

def get_data_path(root_dir, cur_data):
    paths = [None, None, None]
    cur1_data = root_dir + '/' + cur_data
    OLCI_data_dir = root_dir + '/' + cur_data
    MSI_data_dir = root_dir + '/' + cur_data
    Mask_data_dir = root_dir + '/' + cur_data
    for filename in os.listdir(cur1_data):
        if filename[:2] == 'S3':
            paths[0] = os.path.join(OLCI_data_dir, filename)
        elif filename[:2] == 'S2':
            paths[1] = os.path.join(MSI_data_dir, filename)
        elif filename[:2] == 'Ma':
            paths[2] = os.path.join(Mask_data_dir, filename)
    return paths

def load_image_pair(root_dir, cur_data):
    paths = get_data_path(root_dir, cur_data)
    images = []
    for p in paths:
        im = np.load(p)
        images.append(im)
    return images

def get_s3_data_path(root_dir, cur_data):
    s3_path = None
    cur_data_dir = os.path.join(root_dir, cur_data)
    for filename in os.listdir(cur_data_dir):
        if filename[:3] == 'S23':
            s3_path = os.path.join(cur_data_dir, filename)
    return s3_path

def load_s3_image(root_dir, cur_data):
    s3_path = get_s3_data_path(root_dir, cur_data)
    if s3_path is not None:
        image = np.load(s3_path)
        return image
    else:
        raise FileNotFoundError(f"No S3 data found in {cur_data}")

def transform_image(image, flip_num, rotate_num0, rotate_num):
    image_mask = np.ones(image.shape)
    inf_mask = np.where(image > 1000)
    image_mask[inf_mask] = 0.0
    image[inf_mask] = 1.0
    image = image.astype(np.float32)

    if flip_num == 1:
        image = image[:, :, ::-1]

    C, H, W = image.shape

    if rotate_num0 == 1:
        if rotate_num == 2:
            image = image.transpose(0, 2, 1)[::-1, :]
        elif rotate_num == 1:
            image = image.transpose(0, 2, 1)[:, ::-1]
        else:
            image = image.reshape(C, H * W)[:, ::-1].reshape(C, H, W)

    image = torch.from_numpy(image.copy())
    image_mask = torch.from_numpy(image_mask)

    return image, image_mask

def transform_image_test(image, flip_num, rotate_num0, rotate_num):
    image_mask = np.ones(image.shape)
    inf_mask = np.where(image > 1000)
    image_mask[inf_mask] = 0.0
    image[inf_mask] = 1.0
    image = image.astype(np.float32)

    if flip_num == 1:
        image = image[:, :, ::-1]

    C, H, W = image.shape

    if rotate_num0 == 1:
        if rotate_num == 2:
            image = image.transpose(0, 2, 1)[::-1, :]
        elif rotate_num == 1:
            image = image.transpose(0, 2, 1)[:, ::-1]
        else:
            image = image.reshape(C, H * W)[:, ::-1].reshape(C, H, W)

    image = torch.from_numpy(image.copy())
    image_mask = torch.from_numpy(image_mask)

    return image, image_mask

def transform_imagemask(image, flip_num, rotate_num0, rotate_num):
    image_mask = np.ones(image.shape)
    inf_mask = np.where(image > 1000)
    image_mask[inf_mask] = 0.0
    image[inf_mask] = 1.0

    if flip_num == 1:
        image = image[:, :, ::-1]

    C, H, W = image.shape

    if rotate_num0 == 1:
        if rotate_num == 2:
            image = image.transpose(0, 2, 1)[::-1, :]
        elif rotate_num == 1:
            image = image.transpose(0, 2, 1)[:, ::-1]
        else:
            image = image.reshape(C, H * W)[:, ::-1].reshape(C, H, W)

    image = torch.from_numpy(image.copy())
    image_mask = torch.from_numpy(image_mask)

    return image, image_mask

class PatchSet(Dataset):
    def __init__(self, root_dir, image_dates, image_size, patch_size):
        super(PatchSet, self).__init__()
        self.root_dir = root_dir
        self.image_dates = image_dates
        self.image_size = image_size
        self.patch_size = patch_size

        PATCH_STRIDE = self.patch_size // 2
        end_h = (self.image_size[0] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
        end_w = (self.image_size[1] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
        h_index_list = [i for i in range(0, end_h, PATCH_STRIDE)]
        w_index_list = [i for i in range(0, end_w, PATCH_STRIDE)]
        if (self.image_size[0] - PATCH_STRIDE) % PATCH_STRIDE != 0:
            h_index_list.append(self.image_size[0] - self.patch_size)
        if (self.image_size[1] - PATCH_STRIDE) % PATCH_STRIDE != 0:
            w_index_list.append(self.image_size[1] - self.patch_size)

        self.total_index = 1343

    def __getitem__(self, item):
        images = []

        im = np.load(os.path.join(self.root_dir, str(item) + '.npy'))

        images.append(im[: 14, :, :])
        images.append(im[14: 23, :, :])
        images.append(im[23: 24, :, :])

        patches = [None] * len(images)
        masks = [None] * len(images)
        flip_num = np.random.choice(2)
        rotate_num0 = np.random.choice(2)
        rotate_num = np.random.choice(3)

        for i in range(2):
            im = images[i]
            im, im_mask = transform_image(im, flip_num, rotate_num0, rotate_num)
            patches[i] = im
            masks[i] = im_mask

        im1 = images[2]
        im1, im1_mask = transform_imagemask(im1, flip_num, rotate_num0, rotate_num)
        patches[2] = im1

        gt_mask1 = masks[0]
        gt_mask2 = masks[1]

        return patches[0], patches[1], patches[2], gt_mask1, gt_mask2

    def __len__(self):
        return self.total_index