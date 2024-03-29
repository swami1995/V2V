import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_scatter
import ipdb

class LayerNorm1D(nn.Module):
    def __init__(self, dim):
        super(LayerNorm1D, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-4)

    def forward(self, x):
        return self.norm(x.transpose(1,2)).transpose(1,2)

class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid())

        self.res = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.gate(x) * self.res(x)

class SoftAgg(nn.Module):
    def __init__(self, dim=512, expand=True):
        super(SoftAgg, self).__init__()
        self.dim = dim
        self.expand = expand
        self.f = nn.Linear(self.dim, self.dim)
        self.g = nn.Linear(self.dim, self.dim)
        self.h = nn.Linear(self.dim, self.dim)

    def forward(self, x, ix):
        _, jx = torch.unique(ix, return_inverse=True)
        w = torch_scatter.scatter_softmax(self.g(x), jx, dim=1)
        y = torch_scatter.scatter_sum(self.f(x) * w, jx, dim=1)

        if self.expand:
            return self.h(y)[:,jx]
            
        return self.h(y)

class SoftAggBasic(nn.Module):
    def __init__(self, dim=512, expand=True):
        super(SoftAggBasic, self).__init__()
        self.dim = dim
        self.expand = expand
        self.f = nn.Linear(self.dim, self.dim)
        self.g = nn.Linear(self.dim,        1)
        self.h = nn.Linear(self.dim, self.dim)

    def forward(self, x, ix):
        _, jx = torch.unique(ix, return_inverse=True)
        w = torch_scatter.scatter_softmax(self.g(x), jx, dim=1)
        y = torch_scatter.scatter_sum(self.f(x) * w, jx, dim=1)

        if self.expand:
            return self.h(y)[:,jx]
            
        return self.h(y)


### Gradient Clipping and Zeroing Operations ###

GRAD_CLIP = 0.1

class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        return grad_x.clamp(min=-0.01, max=0.01)

class GradientClip(nn.Module):
    def __init__(self):
        super(GradientClip, self).__init__()

    def forward(self, x):
        return GradClip.apply(x)

class GradZero(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        grad_x = torch.where(torch.abs(grad_x) > GRAD_CLIP, torch.zeros_like(grad_x), grad_x)
        return grad_x

class GradientZero(nn.Module):
    def __init__(self):
        super(GradientZero, self).__init__()

    def forward(self, x):
        return GradZero.apply(x)


class GradMag(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        print(grad_x.abs().mean())
        return grad_x


class GradsAlign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, del_mask, del_ratio, targ_grad, logging, step, delta ,idxs, tr_step):
        ctx.save_for_backward(targ_grad, y, step, del_mask, del_ratio, delta, idxs, tr_step)
        return x, y

    @staticmethod
    def backward(ctx, grad_x, grad_y):
        # ipdb.set_trace()
        targ_grad = ctx.saved_tensors[0]
        wts = ctx.saved_tensors[1]
        step = ctx.saved_tensors[2]
        del_mask = ctx.saved_tensors[3]
        del_ratio = ctx.saved_tensors[4]
        delta = ctx.saved_tensors[5]
        idxs = ctx.saved_tensors[6]
        tr_step = ctx.saved_tensors[7]
        grad_mask = (torch.logical_or((targ_grad*grad_x)>=0, del_mask)).float()
        # grad_mask = ((targ_grad*grad_x)>=0).float()
        # grad_y_mask = (grad_y>0).float().mean()
        # grad_y_pos = grad_y.reshape(-1)[(grad_y>=0).reshape(-1)]
        # grad_y_pos_median = grad_y_pos.median()
        # grad_y_pos_mean = grad_y_pos.mean()
        # grad_y_neg = grad_y.reshape(-1)[(grad_y<0).reshape(-1)].abs()
        # grad_y_neg_median = grad_y_neg.median()
        # grad_y_neg_mean = grad_y_neg.mean()
        # logging_grad = torch.stack([(grad_mask).mean(), grad_y_mask, grad_y_pos_median, grad_y_pos_mean, grad_y_neg_median, grad_y_neg_mean]).cpu()
        # grad_y = grad_y - 0.000001        
        grad_x1 = grad_x*grad_mask - grad_x*(1-grad_mask)
        grad_del = grad_x.abs()*(delta*grad_x).sign()
        grad_del1 = grad_x1.abs()*(delta*grad_x1).sign()
        if len(idxs) == 1:
            pos_grad_del_ratio = (grad_del.reshape(-1)[grad_del.reshape(-1) > 0].sum()/grad_del1.reshape(-1)[grad_del1.reshape(-1) > 0].sum())
            neg_grad_del_ratio = (grad_del.reshape(-1)[grad_del.reshape(-1) < 0].sum()/grad_del1.reshape(-1)[grad_del1.reshape(-1) < 0].sum())
            grad_x1.view(-1)[grad_del1.reshape(-1) > 0] *= pos_grad_del_ratio/del_ratio
            grad_x1.view(-1)[grad_del1.reshape(-1) < 0] *= neg_grad_del_ratio*del_ratio
        else:
            prev_idx = grad_del.shape[1]
            pos_grad_del_ratios = []
            neg_grad_del_ratios = []
            for j, idx in enumerate(idxs):
                grad_delj = grad_del[:,-idx:prev_idx]
                grad_del1j = grad_del1[:,-idx:prev_idx]
                grad_x1j = grad_x1[:,-idx:prev_idx]
                pos_grad_del_ratio = (grad_delj.reshape(-1)[grad_delj.reshape(-1) > 0].sum()/grad_del1j.reshape(-1)[grad_del1j.reshape(-1) > 0].sum())
                neg_grad_del_ratio = (grad_delj.reshape(-1)[grad_delj.reshape(-1) < 0].sum()/grad_del1j.reshape(-1)[grad_del1j.reshape(-1) < 0].sum())
                if pos_grad_del_ratio.isinf():
                    pos_grad_del_ratio = torch.tensor(1.0).to(grad_del.device)
                if neg_grad_del_ratio.isinf():
                    neg_grad_del_ratio = torch.tensor(1.0).to(grad_del.device)
                try:
                    grad_x1j.view(-1)[grad_del1j.reshape(-1) > 0] *= pos_grad_del_ratio/del_ratio[j]
                    grad_x1j.view(-1)[grad_del1j.reshape(-1) < 0] *= neg_grad_del_ratio*del_ratio[j]
                except:
                    ipdb.set_trace()
                prev_idx = -idx
                pos_grad_del_ratios.append(pos_grad_del_ratio)
                neg_grad_del_ratios.append(neg_grad_del_ratio)
            pos_grad_del_ratio = torch.stack(pos_grad_del_ratios).mean()
            neg_grad_del_ratio = torch.stack(neg_grad_del_ratios).mean()
            del_ratio = del_ratio.mean()
        wts_ratio = torch.clamp(0.2/wts.mean(), min=0.5, max=2)
        # print(step, grad_y.abs().median().item(), grad_y.abs().mean().item(), grad_x1.abs().median().item(), grad_x1.abs().mean().item())
        # print(tr_step.item(), step.item(), pos_grad_del_ratio.item(), neg_grad_del_ratio.item(), delta.abs().median().item())
        if torch.isnan(pos_grad_del_ratio) or torch.isnan(neg_grad_del_ratio) or pos_grad_del_ratio.isinf().any() or neg_grad_del_ratio.isinf().any():
            ipdb.set_trace()

        # if step==0:
        #     ipdb.set_trace()
        if step >=2:
            grad_y.view(-1)[grad_y.reshape(-1) > 0] /= wts_ratio
            grad_y.view(-1)[grad_y.reshape(-1) < 0] *= wts_ratio
                

        wtdel_ratio = torch.clamp(torch.clamp(grad_x1.abs(), max=0.01).mean()/torch.clamp(grad_y.abs(), max=0.01).mean(), max=4.0)
        # grad_y = grad_y*wtdel_ratio

        # logging_grad = torch.stack([(grad_mask).mean(), grad_y_mask, grad_y_pos_median, grad_y_pos_mean, grad_y_neg_median, grad_y_neg_mean, pos_grad_del_ratio, neg_grad_del_ratio, pos_grad_del_ratio/del_ratio, neg_grad_del_ratio*del_ratio, del_ratio, wts_ratio, wtdel_ratio]).cpu()

        # ipdb.set_trace()
        # if grad_y is not None:
        #     grad_y = grad_y*grad_mask
        return grad_x1, grad_y, None, None, None, None, None, None, None, None

class GradientsAlign(nn.Module):
    def __init__(self):
        super(GradientsAlign, self).__init__()

    def forward(self, x, y, del_mask, del_ratio, targ_grad, logging, step, delta, idxs_tensor, tr_step):
        return GradsAlign.apply(x, y, del_mask, del_ratio, targ_grad, logging, step, delta, idxs_tensor, tr_step)

class GradsAlignLog(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, del_mask, del_ratio, targ_grad, logging, step, delta):
        ctx.save_for_backward(targ_grad, y, step, del_mask, del_ratio, delta)
        return x, y

    @staticmethod
    def backward(ctx, grad_x, grad_y):
        targ_grad = ctx.saved_tensors[0]
        wts = ctx.saved_tensors[1]
        step = ctx.saved_tensors[2]
        del_mask = ctx.saved_tensors[3]
        del_ratio = ctx.saved_tensors[4]
        delta = ctx.saved_tensors[5]
        grad_mask = (torch.logical_or((targ_grad*grad_x)>=0, del_mask)).float()
        # grad_mask = ((targ_grad*grad_x)>=0).float()
        grad_y_mask = (grad_y>0).float().mean()
        grad_y_pos = grad_y.reshape(-1)[(grad_y>=0).reshape(-1)]
        grad_y_pos_median = grad_y_pos.median()
        grad_y_pos_mean = grad_y_pos.mean()
        grad_y_neg = grad_y.reshape(-1)[(grad_y<0).reshape(-1)].abs()
        grad_y_neg_median = grad_y_neg.median()
        grad_y_neg_mean = grad_y_neg.mean()     
        grad_x1 = grad_x*grad_mask - grad_x*(1-grad_mask)
        grad_del = grad_x.abs()*(delta*grad_x).sign()
        grad_del1 = grad_x1.abs()*(delta*grad_x1).sign()
        pos_grad_del_ratio = (grad_del.reshape(-1)[grad_del.reshape(-1) > 0].sum()/grad_del1.reshape(-1)[grad_del1.reshape(-1) > 0].sum())
        neg_grad_del_ratio = (grad_del.reshape(-1)[grad_del.reshape(-1) < 0].sum()/grad_del1.reshape(-1)[grad_del1.reshape(-1) < 0].sum())
        
        grad_x1.view(-1)[grad_del1.reshape(-1) > 0] *= pos_grad_del_ratio/del_ratio
        grad_x1.view(-1)[grad_del1.reshape(-1) < 0] *= neg_grad_del_ratio*del_ratio
   
        # wts_ratio = torch.clamp(0.2/torch.sigmoid(wts).mean(), min=0.5, max=2)
        wts_ratio = torch.clamp(0.2/wts.mean(), min=0.5, max=2)
        # print(step, grad_y.abs().median().item(), grad_y.abs().mean().item(), grad_x1.abs().median().item(), grad_x1.abs().mean().item())
        # if step==0:
        #     ipdb.set_trace()
        # if step >=2:
        #     grad_y.view(-1)[grad_y.reshape(-1) > 0] /= wts_ratio
        #     grad_y.view(-1)[grad_y.reshape(-1) < 0] *= wts_ratio

        wtdel_ratio = torch.clamp(torch.clamp(grad_x1.abs(), max=0.01).mean()/torch.clamp(grad_y.abs(), max=0.01).mean(), max=4.0)
        # grad_y = grad_y*wtdel_ratio

        logging_grad = torch.stack([(grad_mask).mean(), grad_y_mask, grad_y_pos_median, grad_y_pos_mean, grad_y_neg_median, grad_y_neg_mean, pos_grad_del_ratio, neg_grad_del_ratio, pos_grad_del_ratio/del_ratio, neg_grad_del_ratio*del_ratio, del_ratio, wts_ratio, wtdel_ratio]).cpu()

        return grad_x, grad_y, None, None, None, logging_grad, None, None

class GradientsAlignLog(nn.Module):
    def __init__(self):
        super(GradientsAlignLog, self).__init__()

    def forward(self, x, y, del_mask, del_ratio, targ_grad, logging, step, delta):
        return GradsAlignLog.apply(x, y, del_mask, del_ratio, targ_grad, logging, step, delta)

class LogGradsAlign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, del_mask, del_ratio, targ_grad, logging, step, delta):
        ctx.save_for_backward(targ_grad, y, step, del_mask, del_ratio, delta)
        return x, y

    @staticmethod
    def backward(ctx, grad_x, grad_y):
        targ_grad = ctx.saved_tensors[0]
        wts = ctx.saved_tensors[1]
        step = ctx.saved_tensors[2]
        del_mask = ctx.saved_tensors[3]
        del_ratio = ctx.saved_tensors[4]
        delta = ctx.saved_tensors[5]
        grad_mask = (torch.logical_or((targ_grad*grad_x)>=0, del_mask)).float()
        # grad_mask = ((targ_grad*grad_x)>=0).float()
        grad_y_mask = (grad_y>0).float().mean()
        grad_y_pos = grad_y.reshape(-1)[(grad_y>=0).reshape(-1)]
        grad_y_pos_median = grad_y_pos.median()
        grad_y_pos_mean = grad_y_pos.mean()
        grad_y_neg = grad_y.reshape(-1)[(grad_y<0).reshape(-1)].abs()
        grad_y_neg_median = grad_y_neg.median()
        grad_y_neg_mean = grad_y_neg.mean()     
        grad_x1 = grad_x*grad_mask - grad_x*(1-grad_mask)
        grad_del = grad_x.abs()*(delta*grad_x).sign()
        grad_del1 = grad_x1.abs()*(delta*grad_x1).sign()
        pos_grad_del_ratio = (grad_del.reshape(-1)[grad_del.reshape(-1) > 0].sum()/grad_del1.reshape(-1)[grad_del1.reshape(-1) > 0].sum())
        neg_grad_del_ratio = (grad_del.reshape(-1)[grad_del.reshape(-1) < 0].sum()/grad_del1.reshape(-1)[grad_del1.reshape(-1) < 0].sum())

        #### various metrics ####
        # ipdb.set_trace()
        # (targ_grad/targ_grad.norm()*grad_x/grad_x.norm())
        # print(step, ((grad_x/grad_x.norm())*(targ_grad/targ_grad.norm())).reshape(-1)[((grad_x/grad_x.norm())*(targ_grad/targ_grad.norm())).reshape(-1) < 0].sum(), ((grad_x/grad_x.norm())*(targ_grad/targ_grad.norm())).reshape(-1)[((grad_x/grad_x.norm())*(targ_grad/targ_grad.norm())).reshape(-1) > 0].sum())
        

        
        grad_x1.view(-1)[grad_del1.reshape(-1) > 0] *= pos_grad_del_ratio/del_ratio
        grad_x1.view(-1)[grad_del1.reshape(-1) < 0] *= neg_grad_del_ratio*del_ratio
   
        # wts_ratio = torch.clamp(0.2/torch.sigmoid(wts).mean(), min=0.5, max=2)
        wts_ratio = torch.clamp(0.2/wts.mean(), min=0.5, max=2)
        # print(step, grad_y.abs().median().item(), grad_y.abs().mean().item(), grad_x1.abs().median().item(), grad_x1.abs().mean().item())
        # if step==0:
        #     ipdb.set_trace()
        # if step >=2:
        #     grad_y.view(-1)[grad_y.reshape(-1) > 0] /= wts_ratio
        #     grad_y.view(-1)[grad_y.reshape(-1) < 0] *= wts_ratio

        wtdel_ratio = torch.clamp(torch.clamp(grad_x1.abs(), max=0.01).mean()/torch.clamp(grad_y.abs(), max=0.01).mean(), max=4.0)
        # grad_y = grad_y*wtdel_ratio

        logging_grad = torch.stack([(grad_mask).mean(), grad_y_mask, grad_y_pos_median, grad_y_pos_mean, grad_y_neg_median, grad_y_neg_mean, pos_grad_del_ratio, neg_grad_del_ratio, pos_grad_del_ratio/del_ratio, neg_grad_del_ratio*del_ratio, del_ratio, wts_ratio, wtdel_ratio]).cpu()

        return grad_x, grad_y, None, None, None, logging_grad, None, None

class LogGradientsAlign(nn.Module):
    def __init__(self):
        super(LogGradientsAlign, self).__init__()

    def forward(self, x, y, del_mask, del_ratio, targ_grad, logging, step, delta):
        return LogGradsAlign.apply(x, y, del_mask, del_ratio, targ_grad, logging, step, delta)
