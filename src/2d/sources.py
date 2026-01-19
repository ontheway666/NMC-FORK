import torch
import torch.nn.functional as F
import numpy as np
import math
import os
import gpytoolbox
from utils.prms import *


def get_source_velocity(src, i = 1):

    if src == 'taylorgreen':
        source_func = taylorgreen_velocity
    elif src =='liddriven':
        source_func = liddriven_velocity
    elif src =='lidmid':
        source_func = lidmid_velocity
    elif src == 'karman':
        source_func = karman_vortex_velocity
    elif src =='karmanEmpty':
        source_func = karmanEmpty_velocity
    elif src =='karman4edge':
        source_func = karman4edge_velocity
    elif src == 'jpipe':
        source_func = jpipe_velocity
    else:
        raise NotImplementedError
    return source_func

def taylorgreen_velocity(samples: torch.FloatTensor, scene_size=None):
    # samples: [-1, 1], rescale to (0, 2 * pi)
    A = 1
    a = 1
    B = -1
    b = 1
    x = ((samples[:, 0]-scene_size[0])/(scene_size[1] - scene_size[0])) * 2*np.pi
    y = ((samples[:, 1]-scene_size[2])/(scene_size[3] - scene_size[2])) * 2*np.pi
    u = A * torch.sin(a * x) * torch.cos(b * y)
    v = B * torch.cos(a * x) * torch.sin(b * y)
    vel = torch.stack([u, v], dim=-1)
    print(x.shape)      #N 
    print(vel.shape)    #N 2

    #TEST   3vor
    mask = (x > np.pi) & (y > np.pi)
    vel[mask] = 0

    #TEST   vel_H
    vel[mask,1] = -2

    return vel


def karmanEmpty_velocity(samples: torch.FloatTensor, karman_vel, scene_size, eps):
    vel = torch.zeros_like(samples)

    if(bBasicVel):
        vel[..., 1] = 0.5
    

    return vel

def liddriven_velocity(samples: torch.FloatTensor, scene_size=None):
    
    x = ((samples[:, 0]-scene_size[0])/(scene_size[1] - scene_size[0]))
    y = ((samples[:, 1]-scene_size[2])/(scene_size[3] - scene_size[2]))

    lid_y_area = [0.97,0.98]

    frac = torch.clamp((y - lid_y_area[0]) /
                                (lid_y_area[1] - lid_y_area[0]),
                                0.0, 1.0)


    mask =  (y > 0.9) & (x > 0.05) 

    mask = (y>0.6) &(x>0.05)
    one = torch.ones_like(x)
    u = torch.where(mask, one, 0) * 2



    if(bBasicVel):
        u += torch.ones_like(u) * 0.3

    v = torch.zeros_like(x)

    vel = torch.stack([u, v], dim = -1)
    
    return vel


def lidmid_velocity(samples: torch.FloatTensor, scene_size=None):
    
    x = ((samples[:, 0]-scene_size[0])/(scene_size[1] - scene_size[0]))
    y = ((samples[:, 1]-scene_size[2])/(scene_size[3] - scene_size[2]))




    mask =  (y > 0.45) & (y < 0.55) & (x > 0.1) & (x < 0.9)
    one = torch.ones_like(x)
    u = torch.where(mask, one, 0)
    
    if(bBasicVel):
        u += torch.ones_like(u) * 0.3

    v = torch.zeros_like(x)

    vel = torch.stack([u, v], dim = -1)
    
    return vel

def karman_vortex_velocity(samples: torch.FloatTensor, karman_vel, obs_func, scene_size, eps):
    vel = torch.zeros_like(samples)
    # print('[init vel] ',vel.shape)
    vel[..., 0] = karman_vel
    # basic vel


    #TEST  vertVel
    # vel[..., 1] = karman_vel


    dist = obs_func(samples)
    threshold = eps
    weight = torch.clamp(dist, 0, threshold) / threshold

    #zxc 速度根据到边界的距离而有所压缩
    vel *= weight.unsqueeze(-1)

    return vel



def karman4edge_velocity(samples: torch.FloatTensor, karman_vel, obs_func, scene_size, eps):
    vel = torch.zeros_like(samples)

    if(bBasicVel):
        vel[..., 1] = 0.5
    
    dist = obs_func(samples)
    threshold = eps
    weight = torch.clamp(dist, 0, threshold) / threshold

    vel *= weight.unsqueeze(-1)

    return vel


def jpipe_velocity(samples: torch.FloatTensor, karman_vel, obs_func, eps):
    vel = torch.zeros_like(samples)
    mask = samples[..., 0] < 1.4
    vel[..., 0][mask] = karman_vel

    dist = obs_func(samples)
    threshold = eps
    weight = torch.clamp(dist, 0, threshold) / threshold
    vel *= weight.unsqueeze(-1)

    d = torch.sqrt((samples[..., 0]-1)**2 + (samples[..., 1]-1)**2)
    mask1 = (samples[..., 0]>=0.0) & (samples[..., 0]<=1.0) & (samples[..., 1]>=0.0) & (samples[..., 1]<=0.5)
    mask2 = (samples[..., 0]>=1.5) & (samples[..., 0]<=2.0) & (samples[..., 1]>=1.0) & (samples[..., 1]<=2.0)
    mask3 = (d >= 0.5) & (d <= 1.0) & (samples[..., 0]>=1.0) & (samples[..., 1]<=1.0)
    mask = mask1 | mask2 | mask3
    vel[~mask] = 0.0

    # dist = obs_func(samples)
    # threshold = eps
    # weight = torch.clamp(dist, 0, threshold) / threshold
    # vel *= weight.unsqueeze(-1)

    return vel

def _smoothstep_linear(x, xm, e):
        y = torch.abs(x-xm)
        return y/e

def _smoothstep_poly(x, xm, e):
        y = torch.abs(x-xm)
        return y + (((3-2*e)/e**2) * y**2) + (((e-2)/e**3) * y**3)

def _smoothstep_tanh(x, xm, e):
        y = torch.abs(x-xm)
        return ((torch.exp(y) - torch.exp(-y))/(torch.exp(y) + torch.exp(-y))) * ((np.exp(e) + np.exp(-e))/(np.exp(e) - np.exp(-e)))

# 仅圆柱，没有墙
def circle_obstable_functions(center, radius):
    def sdf_func(samples):
        d = torch.sqrt((samples[..., 0] - center[0]) ** 2 + (samples[..., 1] - center[1]) ** 2) - radius
        return d
    
    return sdf_func

def rectangle_obstable_functions(center, hy, hx):
    def sdf_func(samples):
        # d =  torch.max(  torch.abs(samples[..., 0] - center[0])  , torch.abs(samples[..., 1] - center[1]) )  - half_size
        

        """
        Axis-aligned rectangle SDF.

        samples:    (..., 2)
        center: (2,) or broadcastable to (..., 2)
        half:   (2,)  半宽 (hx, hy)
        return: (...,)
        """

        nonlocal center

        half = np.array([hx,hy])
        center = torch.as_tensor(center, dtype=samples.dtype, device=samples.device)
        half   = torch.as_tensor(half,   dtype=samples.dtype, device=samples.device)

        q = torch.abs(samples - center) - half          # (..., 2)

        outside = torch.linalg.norm(
            torch.clamp(q, min=0),
            dim=-1
        )                                            # (...)

        inside = torch.minimum(
            torch.maximum(q[..., 0], q[..., 1]),
            torch.zeros((), device=samples.device)
        )                                            # (...)

        return outside + inside

        
    return sdf_func



def rect3_obstable_functions(center0,hy0,hx0,center1,hy1,hx1,center2,hy2,hx2):
    def sdf_func(samples):
        func0=rectangle_obstable_functions(center0,hy0,hx0)
        func1=rectangle_obstable_functions(center1,hy1,hx1)
        func2=rectangle_obstable_functions(center2,hy2,hx2)
        sdf0=func0(samples)
        sdf1=func1(samples)
        sdf2=func2(samples)

        sdf=torch.minimum(sdf2,torch.minimum(sdf0,sdf1))
        return sdf
    return sdf_func


def jpipe_obstable_functions():
    def sdf_func(samples):
        dist = torch.zeros_like(samples[..., 0])

        mask1 = (samples[..., 0] >= 0.0) & (samples[..., 0] <= 1.0)
        mask2 = (samples[..., 1] >= 1.0) & (samples[..., 1] <= 2.0)
        mask = ~mask1 & ~mask2

        dist[mask1] = torch.minimum(torch.abs(samples[..., 1][mask1] - 0.5), torch.abs(samples[..., 1][mask1]))
        dist[mask2] = torch.minimum(torch.abs(samples[..., 0][mask2] - 1.5), torch.abs(samples[..., 0][mask2] - 2.0))
        dist[mask] = torch.minimum(torch.abs(torch.sqrt((samples[..., 0][mask]-1)**2 + (samples[..., 1][mask]-1)**2) - 0.5), torch.abs(torch.sqrt((samples[..., 0][mask]-1)**2 + (samples[..., 1][mask]-1)**2) - 1))

        return dist
    return sdf_func

def obstacle_function(v, l):
    def sdf_func(samples):
        samples = samples.detach().cpu().numpy()
        dist = np.zeros(samples[..., 0].shape)
        if len(samples.shape) == 3:
            for i in range(samples.shape[0]):
                winding_number = gpytoolbox.winding_number(samples[i], v, l)
                d, _, _ = gpytoolbox.signed_distance(samples[i], v, F=l, use_cpp=True)
                d = d * np.sign(winding_number)
                dist[i] = d
        else:
            winding_number = gpytoolbox.winding_number(samples, v, l)
            d, _, _ = gpytoolbox.signed_distance(samples, v, F=l, use_cpp=True)
            d = d * np.sign(winding_number)
            dist = d
        return torch.Tensor(dist).cuda()
        
    return sdf_func
