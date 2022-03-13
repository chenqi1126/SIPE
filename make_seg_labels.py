import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import argparse
import numpy as np
import importlib
import os
import imageio

from data import data_voc
from tool import torchutils, indexing

cudnn.enabled = True

palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,
					 64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,
					 0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,
					 64,64,0,  192,64,0,  64,192,0, 192,192,0]

def _work(process_id, model, dataset, args):

    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin,
                             shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    cam_out_dir = os.path.join(args.session_name, 'npy')

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            orig_img_size = np.array(pack['size'])

            edge, _ = model(pack['img'][0].cuda(non_blocking=True))

            cam_dict = np.load(cam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()
            cam_dict['IS_CAM'] = cam_dict['IS_CAM']
            keys = cam_dict['keys']
            cams = F.interpolate(torch.tensor(cam_dict['IS_CAM']).unsqueeze(1), edge.shape[1:], mode='bilinear', align_corners=False)[:,0]
            cams = np.power(cams, 1.5)
            
            cam_downsized_values = cams.cuda()
            rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)
            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0], :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up) 
            rw_pred = torch.argmax(rw_up, dim=0).cpu().numpy()
            rw_pred = keys[rw_pred]

            imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--infer_list", default="data/trainaug_voc.txt", type=str)
    parser.add_argument("--voc12_root", default="../PascalVOC2012/VOCdevkit/VOC2012", type=str)
    parser.add_argument("--irn_network", default="network.resnet50_irn", type=str)
    parser.add_argument("--session_name", default="exp", type=str)
    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--sem_seg_out_dir", default="", type=str)
    args = parser.parse_args()


    model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')()
    irn_weights_path = os.path.join(args.session_name, 'ckpt', 'irn.pth')
    model.load_state_dict(torch.load(irn_weights_path), strict=False)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = data_voc.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root, scales=(1.0,))
    dataset = torchutils.split_dataset(dataset, n_gpus)

    args.sem_seg_out_dir = os.path.join(args.session_name, 'pseudo_label')
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)

    print("[", end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print("]")

    torch.cuda.empty_cache()
