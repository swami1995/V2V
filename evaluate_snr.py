import cv2
import os
import argparse
import numpy as np
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dpvo.data_readers.factory import dataset_factory

from dpvo.lietorch import SE3
from dpvo.logger import Logger
import torch.nn.functional as F

from dpvo.net import VONet
from evaluate_tartan import evaluate as validate
import random
import ipdb

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def image2gray(image):
    image = image.mean(dim=0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def kabsch_umeyama(A, B):
    n, m = A.shape
    EA = torch.mean(A, axis=0)
    EB = torch.mean(B, axis=0)
    VarA = torch.mean((A - EA).norm(dim=1)**2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = torch.svd(H)

    c = VarA / torch.trace(torch.diag(D))
    return c

def seeding(seed=0, torch_deterministic=False):
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # if torch_deterministic:
    #     # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    #     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True
    #     torch.use_deterministic_algorithms(True)
    # else:
    #     torch.backends.cudnn.benchmark = True
    #     torch.backends.cudnn.deterministic = False

    return seed

def load_ckpt(net, args, step):
    # ipdb.set_trace()
    ckpt = args.ckpt + '_%06d.pth' % step
    state_dict = torch.load(ckpt)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    net.load_state_dict(new_state_dict, strict=False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)
    return optimizer

def compute_snr(gradient_vectors):
    # Calculate the mean gradient vector (signal)
    gradient_vectors = torch.stack(gradient_vectors).reshape(-1, gradient_vectors[0].reshape(-1).shape[0])
    mean_gradient = torch.mean(gradient_vectors, dim=0).reshape(1, -1)

    # Calculate the deviation of each vector from the mean (noise)
    noise = gradient_vectors - mean_gradient

    # Compute the magnitude of the mean gradient (signal strength)
    signal_magnitude = torch.norm(mean_gradient)

    # Compute the magnitude of noise for each vector and then the mean noise magnitude
    noise_magnitudes = torch.norm(noise, dim=1)
    mean_noise_magnitude = torch.mean(noise_magnitudes)

    # Compute SNR
    snr = signal_magnitude / mean_noise_magnitude

    # Convert SNR to decibels (optional)
    snr_db = 20 * torch.log10(snr)

    return snr, snr_db

def train(args):
    """ main training loop """

    # legacy ddp code
    rank = 0
    seeding(12)

    db = dataset_factory(['tartan'], datapath="/project_data/datasets/TartanAir", n_frames=args.n_frames, test=args.test)
    train_loader = DataLoader(db, batch_size=1, shuffle=True, num_workers=4)

    net = VONet()
    # net.train()
    net.cuda()
    torch.set_printoptions(precision=6, threshold=1000, linewidth=160, sci_mode=False)
    snr_fs = []
    snr_ws = []
    total_steps = 0
    wtd_obj = True
    flow_coeff = 1.0
    ro_coeff = 1.0
    STEPS = 8

    for step in range(0, args.steps, 10000):
        try:
            optimizer = load_ckpt(net, args, step+10000)
        except:
            ipdb.set_trace()
        grad_f = []
        grad_w = []
        for data_blob in train_loader:
            images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob]
            optimizer.zero_grad()

            # fix poses to gt for first 1k steps
            so = False
            
            poses = SE3(poses).inv()
            traj, stats, logging, _ = net(images, poses, disps, intrinsics, M=1024, STEPS=STEPS, structure_only=so, wtd_loss=wtd_obj, total_steps=total_steps)

            tr_list = []
            ro_list = []
            ef_list = []
            loss = 0.0
            pose_loss = 0.0
            flow_loss = 0.0
            ro_loss = 0.0
            tr_loss = 0.0
            for i, (v, x, y, P1, P2, _, wtk, _, _, vf, _, xf, yf, _) in enumerate(traj):
                if wtd_obj:
                    wtk_d = wtk[:, :, None, None, :].detach()
                    e = ((x-y)*wtk_d).norm(dim=-1)
                    e = e.reshape(-1, net.P**2)[(v > 0.5).reshape(-1)].min(dim=-1).values
                else:
                    e = (x - y).norm(dim=-1)
                    e = e.reshape(-1, net.P**2)[(v > 0.5).reshape(-1)].min(dim=-1).values
                    
                ef = (xf - yf).norm(dim=-1)
                ef = ef.reshape(-1, net.P**2)[(vf > 0.5).reshape(-1)].min(dim=-1).values

                N = P1.shape[1]
                ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
                ii = ii.reshape(-1).cuda()
                jj = jj.reshape(-1).cuda()

                k = ii != jj
                ii = ii[k]
                jj = jj[k]

                P1 = P1.inv()
                P2 = P2.inv()

                t1 = P1.matrix()[...,:3,3]
                t2 = P2.matrix()[...,:3,3]

                s = kabsch_umeyama(t2[0], t1[0]).detach().clamp(max=10.0)
                P1 = P1.scale(s.view(1, 1))

                dP = P1[:,ii].inv() * P1[:,jj]
                dG = P2[:,ii].inv() * P2[:,jj]

                e1 = (dP * dG.inv()).log()
                tr = e1[...,0:3].norm(dim=-1)
                ro = e1[...,3:6].norm(dim=-1)

                tr_list.append(tr)
                ro_list.append(ro)
                ef_list.append(ef)
                
                flow_loss += args.flow_weight * e.mean()

            loss = flow_coeff*flow_loss
            
            loss.backward()
            grad_f.append(net.update.d[1].weight.grad.detach().clone())
            grad_w.append(net.update.w[1].weight.grad.detach().clone())
            optimizer.zero_grad()
            total_steps +=1 
            if total_steps % 200 == 0 and args.test:
                snr_f = compute_snr(grad_f)[0].item()
                snr_w = compute_snr(grad_w)[0].item()
                snr_fs.append(snr_f)
                snr_ws.append(snr_w)
                print(f"step {step}, SNR_f: {snr_f}, SNR_w: {snr_w}")
                break

        torch.cuda.empty_cache()
        print("step: ", step, " finished")
    ipdb.set_trace()
    np.save('snrs/snr_f_' + args.ckpt.split('/')[1] + f'_wtdobj{wtd_obj}_{STEPS}_200.npy', snr_fs)
    np.save('snrs/snr_w_' + args.ckpt.split('/')[1] + f'_wtdobj{wtd_obj}_{STEPS}_200.npy', snr_ws)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--steps', type=int, default=240000)
    parser.add_argument('--lr', type=float, default=0.00008)
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--n_frames', type=int, default=8)
    parser.add_argument('--pose_weight', type=float, default=10.0)
    parser.add_argument('--flow_weight', type=float, default=0.1)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--save_dir', type=str, default='runs')
    parser.add_argument('--resume_train', action='store_true')
    args = parser.parse_args()

    train(args)
