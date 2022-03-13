
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import os.path
import scipy.misc
import imageio
from torchvision import transforms
from tool import pyutils, imutils, torchutils
import pdb
import cv2
import random

IMG_FOLDER_NAME = "JPEGImages"
MASK_FOLDER_NAME = "SegmentationClass"


def load_image_label_list_from_npy(img_name_list):

    cls_labels_dict = np.load('data/cls_labels_coco.npy', allow_pickle=True).item()

    return [cls_labels_dict[img_name] for img_name in img_name_list]

    
def get_img_path(img_name, coco_root):
    return os.path.join(coco_root, 'train2014', img_name + '.jpg')

def get_mask_path(mask_name, coco_root):
    return os.path.join(coco_root, MASK_FOLDER_NAME, mask_name + '.png')

def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_gt_name for img_gt_name in img_gt_name_list]

    return img_name_list


class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

class COCOImageDataset(Dataset):
    def __init__(self, img_name_list_path, coco_root, resize=None,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.coco_root = coco_root

        self.resize = resize
        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch

    def __len__(self):
        return len(self.img_name_list)


    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = np.array(PIL.Image.open(get_img_path(name, self.coco_root)).convert('RGB'))

        if self.resize:
            img = imutils.pil_resize(img, size=self.resize, order=3)

        if self.resize_long:
            img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])

        if self.rescale:
            img = imutils.random_scale(img, scale_range=self.rescale, order=3)

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img = imutils.random_lr_flip(img)

        if self.crop_size:
            valid_mask = torch.zeros((81, self.crop_size, self.crop_size))
            if self.crop_method == "random":
                img, box = imutils.random_crop(img, self.crop_size, 0)
                valid_mask[:,box[0]:box[1],box[2]:box[3]] = 1
            else:
                img = imutils.top_left_crop(img, self.crop_size, 0)
                valid_mask = torch.ones((81, self.crop_size, self.crop_size))
            
        else:
            valid_mask = torch.ones((81, img.shape[0], img.shape[1]))

        if self.to_torch:
            img = np.ascontiguousarray(imutils.HWC_to_CHW(img))

        return {'name': name, 'img': img, 'valid_mask': valid_mask}

class COCOClsDataset(COCOImageDataset):

    def __init__(self, img_name_list_path, coco_root, resize=None,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None):
        super().__init__(img_name_list_path, coco_root, resize,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        out['label'] = torch.from_numpy(self.label_list[idx])

        return out

class COCOClsDatasetMSF(COCOClsDataset):

    def __init__(self, img_name_list_path, coco_root, img_normal=TorchvisionNormalize(), scales=(1.0,)):
        super().__init__(img_name_list_path, coco_root)
        self.scales = scales

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = np.asarray(PIL.Image.open(get_img_path(name, self.coco_root)).convert('RGB'))
        # img = imageio.imread(get_img_path(name, self.coco_root)).convert('RGB')

        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
            s_img = self.img_normal(s_img)
            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))

        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]
        
        # mask = PIL.Image.open(get_mask_path(name, self.coco_root))

        out = {"name": name, "img": ms_img_list, "size": (img.shape[0], img.shape[1]),
               "label": torch.from_numpy(self.label_list[idx]), "img_path": get_img_path(name, self.coco_root)}


        return out

class COCOSegmentationDataset(Dataset):

    def __init__(self, img_name_list_path, label_dir, crop_size, coco_root,
                 rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_method = 'random'):

        self.img_name_list = load_img_name_list(img_name_list_path)
        self.coco_root = coco_root

        self.label_dir = label_dir

        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.coco_root)).convert('RGB')
        label = imageio.imread(os.path.join(self.label_dir, name + '.png'))

        img = np.asarray(img)

        if self.rescale:
            img, label = imutils.random_scale((img, label), scale_range=self.rescale, order=(3, 0))

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img, label = imutils.random_lr_flip((img, label))

        
        if self.crop_method == "random":
            img, label = imutils.random_crop((img, label), self.crop_size, (0, 255))
        else:
            img = imutils.top_left_crop(img, self.crop_size, 0)
            label = imutils.top_left_crop(label, self.crop_size, 255)

        label = label.astype(np.long)
        img = imutils.HWC_to_CHW(img)
        return {'name': name, 'img': img, 'label': label}

class GetAffinityLabelFromIndices():

    def __init__(self, indices_from, indices_to):

        self.indices_from = indices_from
        self.indices_to = indices_to

    def __call__(self, segm_map):

        segm_map_flat = np.reshape(segm_map, -1)

        segm_label_from = np.expand_dims(segm_map_flat[self.indices_from], axis=0)
        segm_label_to = segm_map_flat[self.indices_to]

        valid_label = np.logical_and(np.less(segm_label_from, 81), np.less(segm_label_to, 81))

        equal_label = np.equal(segm_label_from, segm_label_to)

        pos_affinity_label = np.logical_and(equal_label, valid_label)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), \
               torch.from_numpy(neg_affinity_label)

class COCOAffinityDataset(COCOSegmentationDataset):
    def __init__(self, img_name_list_path, label_dir, crop_size, coco_root,
                 indices_from, indices_to,
                 rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False, crop_method=None):
        super().__init__(img_name_list_path, label_dir, crop_size, coco_root, rescale, img_normal, hor_flip, crop_method=crop_method)

        self.extract_aff_lab_func = GetAffinityLabelFromIndices(indices_from, indices_to)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        reduced_label = imutils.pil_rescale(out['label'].astype(np.uint8), 0.25, 0)

        out['aff_bg_pos_label'], out['aff_fg_pos_label'], out['aff_neg_label'] = self.extract_aff_lab_func(reduced_label)

        return out
