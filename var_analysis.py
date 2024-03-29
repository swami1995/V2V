import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.optim as optim
from torch.utils.data import DataLoader
from dpvo.data_readers.factory import dataset_factory
from dpvo.net import VONet
import argparse

import torch_scatter
from torch_scatter import scatter_sum

from dpvo import fastba
from dpvo import altcorr
from dpvo import lietorch
from dpvo.lietorch import SE3

from dpvo.extractor import BasicEncoder, BasicEncoder4
from dpvo.blocks import GradientClip, GatedResidual, SoftAgg, GradientsAlign, GradientsAlignLog

from dpvo.utils import *
from dpvo.ba import BA
from dpvo import projective_ops as pops
import ipdb
autocast = torch.cuda.amp.autocast
import matplotlib.pyplot as plt
from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)
plt.style.use("ggplot")

def flow_loss(x, y):
    return (x - y).norm(dim=-1).mean()

def pose_loss(P1, P2):
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

    ro_l = ro.mean()
    tr_l = tr.mean()
    pose_loss = ( tr_l + 5.0*ro_l )

    return pose_loss

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

def kabsch_umeyama(A, B):
    n, m = A.shape
    EA = torch.mean(A, axis=0)
    EB = torch.mean(B, axis=0)
    VarA = torch.mean((A - EA).norm(dim=1)**2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = torch.svd(H)

    c = VarA / torch.trace(torch.diag(D))
    return c

def load_ckpt(net, args, step):
    ckpt = args.ckpt + '_%06d.pth' % step
    state_dict = torch.load(ckpt)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    net.load_state_dict(new_state_dict, strict=False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)
    return optimizer

def groundtruths(images, disps, intrinsics, poses, net):
    images = 2 * (images / 255.0) - 0.5
    intrinsics = intrinsics / 4.0
    disps = disps[:, :, 1::4, 1::4].float()

    _, _, _, patches, ix = net.patchify(images, disps=disps)
    p = net.P

    patches_gt = patches.clone()
    Ps = poses

    d = patches[..., 2, p//2, p//2]
    patches = set_depth(patches, torch.rand_like(d))

    kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0,8, device="cuda"))
    ii = ix[kk]
    return Ps, patches, patches_gt, ii, jj, kk

def flow_loss_interference(args):
    seeding(12)
    db = dataset_factory(['tartan'], datapath="/project_data/datasets/TartanAir", n_frames=args.n_frames, test=args.test)
    train_loader = DataLoader(db, batch_size=1, shuffle=True, num_workers=4)
    net = VONet()

    noise_level = 11
    num_trials = 40
    total_examples = 20
    depth_grad_pose_error = [[] for _ in range(noise_level)]
    pose_grad_pose_error = [[] for _ in range(noise_level)]
    depth_grad_depth_error = [[] for _ in range(noise_level)]
    pose_grad_depth_error = [[] for _ in range(noise_level)]
    depth_grad_error_var = [[] for _ in range(noise_level)]
    pose_grad_error_var = [[] for _ in range(noise_level)]
    depth_grad_error_80 = [[] for _ in range(noise_level)]
    
    num_examples = 0
    net.cuda()
    
    for data_blob in train_loader:
        images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob]
        poses = SE3(poses).inv()
        Ps, _, patches_gt, ii, jj, kk = groundtruths(images, disps, intrinsics, poses, net)
        Gs = SE3(Ps.data.clone())
        patches = patches_gt.clone()
        coords_gt = pops.transform(Ps, patches_gt, intrinsics, ii, jj, kk)
        depth_grads_pose = [[] for _ in range(noise_level)]
        pose_grads_pose = [[] for _ in range(noise_level)]
        pose_grads_depth = [[] for _ in range(noise_level)]
        depth_grads_depth = [[] for _ in range(noise_level)]
        
        for i in range(noise_level):
            for j in range(num_trials):
                pose_noise = torch.randn_like(Gs.data)[:, 0] * 0.02*(i)
                Gs_new = SE3(Gs.data.clone())
                Gs_new.data[:, 0] = Ps.data[:, 0] + pose_noise
                Gs_new.normalize_()
                Gs_new.data.requires_grad_(True)
                depth_gt = patches_gt[...,2,:,:].clone()
                depth_gt.requires_grad_(True)
                patches_gt[...,2] = depth_gt
                coords = pops.transform(Gs_new, patches_gt, intrinsics, ii, jj, kk)
                fl_loss = flow_loss(coords, coords_gt)
                depth_grad_pose = torch.autograd.grad(fl_loss, depth_gt, retain_graph=True)[0]
                pose_grad_pose = torch.autograd.grad(fl_loss, Gs_new.data)[0][:, 1:]
                depth_grads_pose[i].append(depth_grad_pose)
                pose_grads_pose[i].append(pose_grad_pose)
        
        for i in range(noise_level):
            for j in range(num_trials):
                depth_noise = torch.rand_like(patches[:,:80].data[...,2,:,:]) * 0.1*(i) 
                depth = patches.data[:,80:,2,:,:].clone()*0.0
                depth.requires_grad_(True)
                patches.data[:,:80,2,:,:] = patches_gt.data[:,:80,2,:,:] + depth_noise
                patches[:,80:,2,:,:] = patches[:,80:,2,:,:] + depth
                Ps.data.requires_grad_(True)
                coords = pops.transform(Ps, patches, intrinsics, ii, jj, kk)
                fl_loss = flow_loss(coords, coords_gt)
                pose_grad = torch.autograd.grad(fl_loss, Ps.data, retain_graph=True)[0]
                depth_grad = torch.autograd.grad(fl_loss, depth, retain_graph=True)[0]
                pose_grads_depth[i].append(pose_grad)
                depth_grads_depth[i].append(depth_grad)

        for i in range(noise_level):
            depth_grad_pose = torch.stack(depth_grads_pose[i], dim=0).view(num_trials, -1, 9).norm(dim=-1)
            pose_grad_pose = torch.stack(pose_grads_pose[i], dim=0).view(num_trials, -1, 7).norm(dim=-1)
            depth_grad_depth = torch.stack(depth_grads_depth[i], dim=0).view(num_trials, -1, 9).norm(dim=-1)
            pose_grad_depth = torch.stack(pose_grads_depth[i], dim=0).view(num_trials, -1, 7).norm(dim=-1)
            depth_grad_pose_error[i].append(depth_grad_pose.mean())
            pose_grad_pose_error[i].append(pose_grad_pose.mean())
            depth_grad_depth_error[i].append(depth_grad_depth.mean())
            pose_grad_depth_error[i].append(pose_grad_depth.mean())
        
        print(num_examples)
        num_examples += 1
        if num_examples > total_examples:
            break
    for i in range(noise_level):
        depth_grad_pose_error[i] = torch.stack(depth_grad_pose_error[i], dim=0).mean(dim=0).item()
        pose_grad_pose_error[i] = torch.stack(pose_grad_pose_error[i], dim=0).mean(dim=0).item()
        depth_grad_depth_error[i] = torch.stack(depth_grad_depth_error[i], dim=0).mean(dim=0).item()
        pose_grad_depth_error[i] = torch.stack(pose_grad_depth_error[i], dim=0).mean(dim=0).item()
        
    noise_level_poses = torch.arange(0, 0.22, 0.02)
    noise_level_depth = torch.arange(0, 1.1, 0.1)
    flow_loss_interference_plot_all(depth_grad_pose_error, pose_grad_pose_error, depth_grad_depth_error, pose_grad_depth_error, noise_level_poses, noise_level_depth)
    return None

def flow_loss_interference_plot_all(depth_grad_pose_error, pose_grad_pose_error, depth_grad_depth_error, pose_grad_depth_error, noise_level_poses, noise_level_depth):
    folder = 'fl_loss_int/test_old4/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    fig, ax1 = plt.subplots(figsize=(5, 4))
    line1, = ax1.plot(noise_level_depth, pose_grad_depth_error, label='Pose gradient error', color='#1f77b4')
    line2, = ax1.plot(noise_level_depth, depth_grad_depth_error, label='Depth gradient error', color='#ff7f0e')
    ax1.set_xlabel('Noise added to depth')
    ax1.set_ylabel('Avg Grad Error Norm')
    ax1.legend()
    plt.savefig(folder+'vary_depth_flint.pdf')

    fig, ax1 = plt.subplots(figsize=(6.2, 4))
    line1, = ax1.plot(noise_level_poses, pose_grad_pose_error, label='Pose gradient error', color='#1f77b4')
    ax1.set_xlabel('Noise added to pose')
    ax1.set_ylabel('Avg Grad Error Norm', color='#1f77b4')
    ax2 = ax1.twinx()
    line2, = ax2.plot(noise_level_poses, depth_grad_pose_error, label='Depth gradient error', color='#ff7f0e')
    ax2.set_ylabel('Avg Grad Error Norm', color='#ff7f0e')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    handles, labels = [], []
    for ax in [ax1, ax2]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Creating a combined legend inside the plot area
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.85, 0.15))
    plt.savefig(folder+'vary_pose_flint.pdf')


def depth_pose_grad_linearization(args):
    seeding(12)
    db = dataset_factory(['tartan'], datapath="/project_data/datasets/TartanAir", n_frames=args.n_frames, test=args.test)
    train_loader = DataLoader(db, batch_size=1, shuffle=True, num_workers=4)
    net = VONet()
    net.cuda()
    total_steps = 0
    num_examples = 0
    total_examples = 20
    wtd_obj = False
    STEPS = 8
    optimizer = load_ckpt(net, args, 240000)
    noise_level = 21
    flow_grad_depth_var = [[] for _ in range(noise_level)]
    pose_grad_depth_var = [[] for _ in range(noise_level)]
    flow_grad_pose_var = [[] for _ in range(noise_level)]
    pose_grad_pose_var = [[] for _ in range(noise_level)]
    for data_blob in train_loader:
        images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob]
        optimizer.zero_grad()

        so = False
        
        poses = SE3(poses).inv()
        with torch.no_grad():
            traj, stats, logging, more_outs = net(images, poses, disps, intrinsics, M=1024, STEPS=STEPS, structure_only=so, wtd_loss=wtd_obj, total_steps=total_steps)
            _, _, _, Gs, Ps, _, _, delta, weight_net, valid, target, _, _, _ = traj[-1]
            patches, patches_gt, ii, jj, kk, bounds =  more_outs[-1]
        depth_grad_linearization(Gs, patches, patches_gt, intrinsics, target, Ps, weight_net, ii, jj, kk, bounds, noise_level, flow_grad_depth_var, pose_grad_depth_var, flow_grad_pose_var, pose_grad_pose_var)
        num_examples += 1
        if num_examples > total_examples:
            break
    for i in range(noise_level):
        flow_grad_depth_var[i] = torch.stack(flow_grad_depth_var[i], dim=0).mean().item()
        pose_grad_depth_var[i] = torch.stack(pose_grad_depth_var[i], dim=0).mean().item()
        flow_grad_pose_var[i] = torch.stack(flow_grad_pose_var[i], dim=0).mean().item()
        pose_grad_pose_var[i] = torch.stack(pose_grad_pose_var[i], dim=0).mean().item()
    noise_level_poses = torch.arange(0, 2.1, 0.1)
    noise_level_depth = torch.arange(0, 2.1, 0.1)
    depth_pose_grad_linearization_plot(flow_grad_depth_var, pose_grad_depth_var, flow_grad_pose_var, pose_grad_pose_var, noise_level_poses, noise_level_depth)
    return None

def depth_pose_grad_linearization_plot(flow_grad_depth_var, pose_grad_depth_var, flow_grad_pose_var, pose_grad_pose_var, noise_levels_poses, noise_levels_depth):
    folder = 'ba_lin_figs/final_snrdb_paper/'
    plt.figure(figsize=(5, 4))
    plt.plot(noise_levels_depth[1:], flow_grad_depth_var[1:], label='Flow loss gradient')
    plt.plot(noise_levels_depth[1:], pose_grad_depth_var[1:], label='Pose loss gradient')
    plt.legend()
    plt.xlabel('Noise added to depth')
    plt.ylabel('SNR (db)')
    plt.savefig(folder + 'linearization_issues_3.pdf')
    return None


def depth_grad_linearization(Gs, patches, patches_gt, intrinsics, target, Ps, weight_net, ii, jj, kk, bounds, noise_level, flow_grad_depth_var, pose_grad_depth_var, flow_grad_pose_var, pose_grad_pose_var):    
    lmbda = 1e-4
    ep = 10
    structure_only = False
    num_trials = 10
    flow_grad_depth = [[] for _ in range(noise_level)]
    pose_grad_depth = [[] for _ in range(noise_level)]
    flow_grad_pose = [[] for _ in range(noise_level)]
    pose_grad_pose = [[] for _ in range(noise_level)]
    for i in range(noise_level):
        for j in range(num_trials):
            depth_noise = torch.randn_like(patches.data) * 0.1*(i)
            patches_new = patches + depth_noise
            target.requires_grad_(True)
            Gs_out, patches_out = BA(Gs, patches_new, intrinsics, target, weight_net, lmbda, ii, jj, kk, 
                bounds, ep=ep, fixedp=1, structure_only=structure_only)
            patches_out = patches_out + (patches - patches_out).detach().clone()
            Gs_out.data = Gs_out.data + (Gs.data - Gs_out.data).detach().clone()
            coords = pops.transform(Gs, patches_out, intrinsics, ii, jj, kk)
            coords_gt = pops.transform(Ps, patches_gt, intrinsics, ii, jj, kk)
            fl_loss = flow_loss(coords, coords_gt)
            p_loss = pose_loss(Gs_out, Ps)
            flow_grad_depth[i].append(torch.autograd.grad(fl_loss, target, retain_graph=True)[0])
            pose_grad_depth[i].append(torch.autograd.grad(p_loss, target)[0])

    for i in range(noise_level):
        for j in range(num_trials):
            pose_noise = torch.randn_like(Gs.data)[:, 0] * 0.01*(i)
            Gs_new = SE3(Gs.data.clone())
            Gs_new.data[:, 0] = Ps.data[:, 0] + pose_noise
            Gs_new.normalize_()
            target.requires_grad_(True)
            Gs_out, patches_out = BA(Gs_new, patches, intrinsics, target, weight_net, lmbda, ii, jj, kk, 
                bounds, ep=ep, fixedp=1, structure_only=structure_only)
            patches_out = patches_out + (patches - patches_out).detach().clone()
            Gs_out.data = Gs_out.data + (Gs.data - Gs_out.data).detach().clone()
            coords = pops.transform(Ps, patches_out, intrinsics, ii, jj, kk)
            coords_gt = pops.transform(Ps, patches_gt, intrinsics, ii, jj, kk)
            fl_loss = flow_loss(coords, coords_gt) 
            p_loss = pose_loss(Gs_out, Ps) 
            flow_grad_pose[i].append(torch.autograd.grad(fl_loss, target, retain_graph=True)[0])
            pose_grad_pose[i].append(torch.autograd.grad(p_loss, target)[0])
    
    ## no noise gradient
    target.requires_grad_(True)
    Gs_out, patches_out = BA(Gs, patches, intrinsics, target, weight_net, lmbda, ii, jj, kk,
        bounds, ep=ep, fixedp=1, structure_only=structure_only)
    patches_out = patches_out + (patches - patches_out).detach().clone()
    Gs_out.data = Gs_out.data + (Gs.data - Gs_out.data).detach().clone()
    coords = pops.transform(Gs, patches_out, intrinsics, ii, jj, kk)
    coords_gt = pops.transform(Ps, patches_gt, intrinsics, ii, jj, kk)
    fl_loss = flow_loss(coords, coords_gt)
    p_loss = pose_loss(Gs_out, Ps)
    flow_grad_mean = torch.autograd.grad(fl_loss, target, retain_graph=True)[0].view(-1)
    pose_grad_mean = torch.autograd.grad(p_loss, target)[0].view(-1)

    for i in range(noise_level):
        flow_grad_depth[i] = torch.stack(flow_grad_depth[i], dim=0).view(num_trials, -1)
        flow_grad_depth_var[i].append(10 * torch.log10(flow_grad_mean.norm()/((flow_grad_depth[i] - flow_grad_mean).norm(dim=1).mean())))
        pose_grad_depth[i] = torch.stack(pose_grad_depth[i], dim=0).view(num_trials, -1)
        pose_grad_depth_var[i].append(10 * torch.log10(pose_grad_mean.norm()/((pose_grad_depth[i] - pose_grad_mean).norm(dim=1).mean())))
        flow_grad_pose[i] = torch.stack(flow_grad_pose[i], dim=0).view(num_trials, -1)
        flow_grad_pose_var[i].append(10 * torch.log10(flow_grad_mean.norm()/((flow_grad_pose[i] - flow_grad_mean).norm(dim=1).mean())))
        pose_grad_pose[i] = torch.stack(pose_grad_pose[i], dim=0).view(num_trials, -1)
        pose_grad_pose_var[i].append(10 * torch.log10(pose_grad_mean.norm()/((pose_grad_pose[i] - pose_grad_mean).norm(dim=1).mean())))
        
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--steps', type=int, default=240000)
    parser.add_argument('--lr', type=float, default=0.00008)
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--n_frames', type=int, default=15)
    parser.add_argument('--pose_weight', type=float, default=10.0)
    parser.add_argument('--flow_weight', type=float, default=0.1)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--save_dir', type=str, default='runs')
    parser.add_argument('--resume_train', action='store_true')
    args = parser.parse_args()

    flow_loss_interference(args)
    depth_pose_grad_linearization(args)