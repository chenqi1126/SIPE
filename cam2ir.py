import os
import numpy as np
import imageio
from torch import multiprocessing
from torch.utils.data import DataLoader
from data import data_voc, data_coco
from tool import torchutils, imutils
import argparse

def _work(process_id, infer_dataset, args):

    databin = infer_dataset[process_id]
    infer_data_loader = DataLoader(databin, shuffle=False, num_workers=0, pin_memory=False)
    cam_out_dir = os.path.join(args.session_name, 'npy')
    ir_label_out_dir = os.path.join(args.session_name, 'ir_label')

    for iter, pack in enumerate(infer_data_loader):
        img_name = pack['name'][0]
        img = pack['img'][0].numpy()

        cam_dict = np.load(os.path.join(cam_out_dir, img_name + '.npy'), allow_pickle=True).item()
        cams = cam_dict['IS_CAM']
        keys = cam_dict['keys']

        # 1. find confident fg & bg
        fg_conf_cam = cams
        fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
        pred = imutils.crf_inference_label(img, fg_conf_cam, n_labels=keys.shape[0])
        fg_conf = keys[pred]

        bg_conf_cam = cams
        bg_conf_cam[0] = bg_conf_cam[0]*0.5
        bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
        pred = imutils.crf_inference_label(img, bg_conf_cam, n_labels=keys.shape[0])
        bg_conf = keys[pred]

        # 2. combine confident fg & bg
        conf = fg_conf.copy()
        conf[fg_conf == 0] = 255
        conf[bg_conf + fg_conf == 0] = 0

        imageio.imwrite(os.path.join(ir_label_out_dir, img_name + '.png'), conf.astype(np.uint8))

        if process_id == args.num_workers - 1 and iter % (len(databin) // 20) == 0:
            print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="voc", type=str)
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--session_name", default='exp', type=str)
    args = parser.parse_args()

    assert args.dataset in ['voc', 'coco'], 'Dataset must be voc or coco in this project.'

    if args.dataset == 'voc':
        dataset_root = '../PascalVOC2012/VOCdevkit/VOC2012'
        dataset = data_voc.VOC12ImageDataset('data/train_' + args.dataset + '.txt', voc12_root=dataset_root, img_normal=None, to_torch=False)

    elif args.dataset == 'coco':
        dataset_root = "../../dataset/ms_coco_14&15/images"
        dataset = data_coco.COCOImageDataset('data/train_' + args.dataset + '.txt', coco_root=dataset_root, img_normal=None, to_torch=False)

    dataset = torchutils.split_dataset(dataset, args.num_workers)
    os.makedirs(os.path.join(args.session_name, 'ir_label'), exist_ok=True)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=args.num_workers, args=(dataset, args), join=True)
    print(']')
