import os
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from .base import NeuralFluidBase
from .networks import get_network
from utils.diff_ops import curl2d_fdiff, laplace, divergence, jacobian, gradient, curl2d
from utils.model_utils import sample_uniform_2D, sample_random_2D
from utils.vis_utils import draw_scalar_field2D, draw_vector_field2D, save_figure, save_figure_nopadding
from utils.prms import *

import sys
sys.path.append("/w/nmc/Neural-Monte-Carlo-Fluid-Simulation/bindings/zombie/build/")
import zombie_bindings
import json
import matplotlib.pyplot as plt
import sys
import cv2
from sklearn.neighbors import KDTree
import gpytoolbox
from torch_cubic_spline_grids import CubicBSplineGrid2d

np.set_printoptions(threshold=sys.maxsize)

# torch.autograd.set_detect_anomaly(True)


def toCpuSafe(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    else:
        return x
    
debugcnt=0
debugframe=0




sampleNum=None
outer_mat=None
def outBCData(mat,width=0.25,onlyOutData=True):
    global outer_mat
    global sampleNum

    assert(len(mat.shape)==2 and mat.shape[1]==2)
    device = mat.device
    dtype = mat.dtype

    xmin = torch.min(mat[:,0]).item()
    xmax = torch.max(mat[:,0]).item()
    ymin = torch.min(mat[:,1]).item()
    ymax = torch.max(mat[:,1]).item()

   

    area_inner = (xmax - xmin) * (ymax - ymin)
    
    density = mat.shape[0] / area_inner

    area_outer = (
        (xmax - xmin) * 2 * width)
       
    
    num_outer = int(area_outer * density)

    # 为了避免过滤后不够，多采一点
    oversample = int(num_outer * 1.3)
    if(not(outer_mat is None) and (sampleNum == mat.shape[0])):
        pass
    else:
        # print('[outBCData] ')
        # print(xmin, xmax, ymin, ymax)

        # 在大矩形里均匀采样
        rand = torch.rand(oversample, 2, device=device, dtype=dtype)
        rand[:, 0] = rand[:, 0] * (xmax - xmin) + xmin
        rand[:, 1] = rand[:, 1] * (ymax - ymin + 2 * width) + (ymin - width)

        # 过滤掉内部矩形
        mask_inner = (
            (rand[:, 0] >= xmin) & (rand[:, 0] <= xmax) &
            (rand[:, 1] >= ymin) & (rand[:, 1] <= ymax)
        )
        outer_mat = rand[~mask_inner]
        
        # 截断到目标数量
        outer_mat = outer_mat[:num_outer]

        #if not, error
        outer_mat = outer_mat.detach()

        sampleNum = mat.shape[0]



    if onlyOutData:
        return outer_mat
    
    mat = torch.cat([mat, outer_mat], dim = 0)

    #   optional
    # perm= torch.randperm(mat.shape[0],device=device) 
    # mat = mat[perm]

  

    assert(len(mat.shape)==2 and mat.shape[1]==2)
    return mat


def checkTensorNan(mat):

    if isinstance(mat, np.ndarray):
        mat=torch.as_tensor(mat)
        # NaN 的个数
    nan_count = torch.isnan(mat).sum().item()
    inf_count = torch.isinf(mat).sum().item()
       # 元素总数
    total_count = mat.numel()

    # 占比
    illegal_ratio = (inf_count+nan_count) / total_count
    if(illegal_ratio>0):
        print('[illegal_ratio]\t'+str(illegal_ratio))
    # if(illegal_ratio>0.5):
        # assert(False)


    # if(not torch.isfinite(mat).all()):
        # print('[NaN detected in Tensor]!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    mat1 = torch.nan_to_num(mat, nan=0.0,  posinf=0.0, neginf=0.0)
    mat1 = torch.clamp(mat1, min=-10.0, max=10.0)
    
    # print(torch.min(mat1), torch.max(mat1), torch.mean(mat1))
    
    return mat1


class NeuralFluidSplit(NeuralFluidBase):
    def __init__(self, cfg):
        super(NeuralFluidSplit, self).__init__(cfg)
        
        f = open(self.cfg.wost_json)
        self.wost_data = json.load(f)
        f.close()

        self.vis_mag_dir = os.path.join(self.cfg.results_dir, 'magnitude')
        self.vis_pressure_dir = os.path.join(self.cfg.results_dir, 'pressure')
        self.grad_p = None
        self.pressure_samples = None
        self.wost_samples_x = None
        self.wost_samples_y = None
        self.wost_samples = None
        self.wost_flag = False

    @property
    def _trainable_networks(self):
        return {'velocity': self.velocity_field}

    def step(self):
        if self.cfg.reset_wts == 1:
            reset = True
        else:
            reset=False

        self.velocity_field_prev.load_state_dict(self.velocity_field.state_dict())
        self.velocity_field_tilde.load_state_dict(self.velocity_field.state_dict())

        if self.cfg.adv_ref == 0:
            # karman 
            # taylor green

            self.create_optimizer(reset=reset)    
            self.advect_velocity(dt=self.cfg.dt, flag=False)

            self.velocity_field_tilde.load_state_dict(self.velocity_field.state_dict()) #u_tilde
            self.velocity_field_prev.load_state_dict(self.velocity_field.state_dict())
            self.create_optimizer(reset=reset)
            self.project_velocity()
            self.wost_flag = False

        else:
            self.create_optimizer(reset=reset)
            self.advect_velocity(dt=self.cfg.dt/2, flag=False)

            self.velocity_field_tilde.load_state_dict(self.velocity_field.state_dict()) #u_tilde
            self.velocity_field_prev.load_state_dict(self.velocity_field.state_dict())
            self.create_optimizer(reset=reset)
            self.project_velocity()
            self.wost_flag = False

            self.velocity_field_prev.load_state_dict(self.velocity_field.state_dict())
            self.create_optimizer(reset=reset)
            self.advect_velocity(dt=self.cfg.dt/2, flag=True)

            self.velocity_field_prev.load_state_dict(self.velocity_field.state_dict()) #u_tilde
            self.create_optimizer(reset=reset)
            self.project_velocity()
            self.wost_flag = False

        self.velocity_field_prev.load_state_dict(self.velocity_field.state_dict())

    def advect_velocity(self, dt, flag):
        self._advect_velocity(dt, flag)
    
    @NeuralFluidBase._training_loop
    def _advect_velocity(self, dt, flag):
        """velocity advection: dudt = -(u\cdot grad)u"""
        samples = self.sample_in_training()

        if(bOutBC):
            samples = outBCData(samples,onlyOutData=False)

        # dudt
        with torch.no_grad():
            assert(len(samples.shape)==2 and samples.shape[1]==2)
            

            # 这个查询过程不参与梯度计算
            prev_u = self.query_velocity(samples, use_prev=True).detach()
        

            # print('[prev u1]')
      

        if self.cfg.time_integration == 'semi_lag':
            # backtracking
            backtracked_position = samples - prev_u * dt # Eqn 9 in Rundi's paper INSR
            backtracked_position[..., 0] = torch.clamp(torch.clone(backtracked_position[..., 0]), min=self.scene_size[0], max=self.scene_size[1])
            backtracked_position[..., 1] = torch.clamp(torch.clone(backtracked_position[..., 1]), min=self.scene_size[2], max=self.scene_size[3])
            
            with torch.no_grad():
                # 这里面计算出来的量，是不参与梯度回传的

                if not flag:
                    # 查找向后追踪位置时的速度
                    advected_u = self.query_velocity(backtracked_position, use_prev=True).detach()
                else:
                    print('[USE tilde]')
                    advected_u = 2*self.query_velocity(backtracked_position, use_prev=True).detach() - self.query_velocity(backtracked_position, use_tilde=True)

            curr_u = self.query_velocity(samples)

            assert(not advected_u.requires_grad)

            # 也就是要求velocity_field直接编码向后追踪时的速度
            loss = torch.mean((curr_u - advected_u) ** 2)
            loss_dict = {'main': loss}

        else:
            raise NotImplementedError

        # if self.boundary_cond == 'zero':
        #     bc_loss_dict = self._velocity_boundary_loss(samples.shape[0] // 100)
        #     loss_dict.update(bc_loss_dict)

        return loss_dict

    def laplacian_smoothing(self, samples, grad_p, lda=1.0):
        kdtree = KDTree(samples)
        _, nearest_ind = kdtree.query(samples, k=10)
        for _ in range(1):
            mdn1 = np.median(grad_p[nearest_ind, 0], axis=1)
            mdn2 = np.median(grad_p[nearest_ind, 1], axis=1)
            grad_p[..., 0] = mdn1
            grad_p[..., 1] = mdn2
        
        return grad_p
    
    def get_area(self, v1, v2, v3):
        D = torch.ones((v1.shape[0], 3, 3))
        D[..., 0][..., 1:] = v1
        D[..., 1][..., 1:] = v2
        D[..., 2][..., 1:] = v3

        area = torch.abs(0.5 * torch.linalg.det(D))
        return area.to(self.device)

    
    def find_closest_index(self, s, grad):
        x_ind = torch.searchsorted(self.wost_samples_x, s[..., 0], side='right')
        y_ind = torch.searchsorted(self.wost_samples_y, s[..., 1], side='right')

        x2 = self.wost_samples_x[torch.clamp(x_ind, min=0, max=self.wost_samples_x.shape[0]-1)]
        y2 = self.wost_samples_y[torch.clamp(y_ind, min=0, max=self.wost_samples_y.shape[0]-1)]
        x1 = self.wost_samples_x[torch.clamp(x_ind-1, min=0, max=self.wost_samples_x.shape[0]-1)]
        y1 = self.wost_samples_y[torch.clamp(y_ind-1, min=0, max=self.wost_samples_y.shape[0]-1)]

        x = s[..., 0]
        y = s[..., 1]

        ind11 = torch.clamp((x_ind-1) * len(self.wost_samples_y) + (y_ind-1), min=0, max=grad.shape[0]-1)
        ind12 = torch.clamp((x_ind-1) * len(self.wost_samples_y) + (y_ind), min=0, max=grad.shape[0]-1)
        ind21 = torch.clamp((x_ind) * len(self.wost_samples_y) + (y_ind-1), min=0, max=grad.shape[0]-1)
        ind22 = torch.clamp((x_ind) * len(self.wost_samples_y) + (y_ind), min=0, max=grad.shape[0]-1) 

        w11 = (x2 - x)*(y2-y) / ((x2-x1)*(y2-y1))
        w12 = (x2 - x)*(y-y1) / ((x2-x1)*(y2-y1))
        w21 = (x - x1)*(y2-y) / ((x2-x1)*(y2-y1))
        w22 = (x - x1)*(y-y1) / ((x2-x1)*(y2-y1))

        # s_x = torch.linalg.norm(self.wost_samples[ind1][..., 0] - self.wost_samples[ind3][..., 0], axis=-1)
        # s_y = torch.linalg.norm(self.wost_samples[ind1][..., 1] - self.wost_samples[ind2][..., 1], axis=-1)
        
        # w1 = torch.linalg.norm(s[..., 0] - self.wost_samples[ind1][..., 0], axis=-1)/s_x
        # w2 = torch.linalg.norm(s[..., 1] - self.wost_samples[ind1][..., 1], axis=-1)/s_y

        # g = torch.clone(grad[ind1]).to(self.device)
        # g[..., 0] = (1-w1)*grad[ind1][..., 0] + w1*grad[ind3][..., 0]
        # g[..., 1] = (1-w2)*grad[ind1][..., 1] + w2*grad[ind2][..., 1]

        # return g
        out = w11[:, None]*grad[ind11] + w12[:, None]*grad[ind12] + w21[:, None]*grad[ind21] + w22[:, None]*grad[ind22]
        out[torch.isnan(out)] = 0.0
        return out
        # return (grad[ind1] + grad[ind2] + grad[ind3] + grad[ind4])/4.0

    # def find_closest_index(self, s, grad):
    #     grid = CubicBSplineGrid2d(resolution=grad.shape, n_channels=1)


    # div 散度
    def wost_pressure(self, div, mag_path):
        print('[WOST]')
        sceneConfig = self.wost_data["scene"]
        sceneConfig["sourceValue"] = mag_path
        solverConfig = self.wost_data["solver"]
        outputConfig = self.wost_data["output"]

        scene = zombie_bindings.Scene(sceneConfig, div)
        # scene = zombie_bindings.Scene(sceneConfig)
        
        print('P sample ',self.pressure_samples.detach().cpu().numpy().shape)
        # n0 2

        # 送入散度,求eq.7 
        samples, p_arr, grad_arr = zombie_bindings.wost(scene, solverConfig, outputConfig, self.pressure_samples.detach().cpu().numpy())
        samples = np.array(samples)
        p = np.array(p_arr)
        grad_p = np.array(grad_arr)

        print('sample = wost', samples.shape)
        # n0 2
        print('p = wost', p.shape)
        # n0 1
        print('grad = wost ',grad_p.shape)
        # n0 2


        # grad_p[np.isnan(grad_p)] = 0.0

        # grad_p = self.laplacian_smoothing(samples, grad_p)

        # grad_p[np.abs(grad_p) < 1e-3] = 0.0

        # min_x = np.min(grad_p[..., 0])
        # min_y = np.min(grad_p[..., 1])
        # max_x = np.max(grad_p[..., 0])
        # max_y = np.max(grad_p[..., 1])

        # grad_p[..., 0] = np.where((grad_p[..., 0] > max_x/2) & (grad_p[..., 0] > 0.0), 0.0, grad_p[..., 0])
        # grad_p[..., 0] = np.where((grad_p[..., 0] < min_x/2) & (grad_p[..., 0] < 0.0), 0.0, grad_p[..., 0])
        # grad_p[..., 1] = np.where((grad_p[..., 1] > max_y/2) & (grad_p[..., 1] > 0.0), 0.0, grad_p[..., 1])
        # grad_p[..., 1] = np.where((grad_p[..., 1] < min_y/2) & (grad_p[..., 1] < 0.0), 0.0, grad_p[..., 1])
        
        # self.wost_samples = torch.Tensor(samples).to(self.device)
        self.wost_samples_x = torch.Tensor(np.unique(samples[..., 0])).to(self.device)
        self.wost_samples_y = torch.Tensor(np.unique(samples[..., 1])).to(self.device)

        self.P = np.mean(p)

        # mask = (samples[..., 0] < self.scene_size[0] + self.bdry_eps) | (samples[..., 0] > self.scene_size[1] - self.bdry_eps) | (samples[..., 1] < self.scene_size[2] + self.bdry_eps) | (samples[..., 1] > self.scene_size[3] - self.bdry_eps)
        # grad_p[..., 0][mask] = 0.0
        # grad_p[..., 1][mask] = 0.0

        # print(self.wost_samples_x)
        # print(self.wost_samples_y)

        return samples, p, grad_p

    # 速度场的散度
    def get_divergence(self, resolution, save_path_png, save_path_pfm, vmin=None, vmax=None):
        grid_values, grid_samples = self.sample_velocity_field(resolution, to_numpy=False, return_samples=True, require_grad=True)
        div = divergence(grid_values, grid_samples).detach().cpu().numpy()
        div = -div[..., 0] # Wost solves lap u = -f

        min = np.min(div)
        max = np.max(div)
        # print(div.shape)
        
        fig = draw_scalar_field2D(div, vmin=vmin, vmax=vmax, figsize=self.fig_size, cmap='viridis', colorbar=True)
        save_figure_nopadding(fig, save_path_png)
        # mag 
        
        return div

    @NeuralFluidBase._training_loop
    def _project_velocity(self, flag=False):
        """projection step for velocity: u <- u - grad(p)"""

        save_path_png = os.path.join(self.vis_mag_dir, f'mag_t{self.timestep:03d}.png')
        save_path_pfm = os.path.join(self.vis_mag_dir, f'mag_t{self.timestep:03d}.pfm')
        

        # 每个step只算一次
        if not self.wost_flag:
            self.wost_flag=True

            #位置坐标
            self.pressure_samples = self.sample_in_training(resolution=self.cfg.wost_resolution)
            div = self.get_divergence(1000, save_path_png, save_path_pfm)


            print('wost(div) ',div.shape)


            #only use
            samples_arr, p, grad_p = self.wost_pressure(div, save_path_pfm)

            # grad_p = self.laplacian_smoothing(samples_arr, grad_p)
            

            # [delete] 画图的当中，只有很少一部分会被使用上，其他的grad_p从未被使用

            self.grad_p = torch.Tensor(grad_p).to(self.device)
            self.samples_arr=torch.Tensor(samples_arr).to(self.device)


        # print('[grad p sample ratio]\t' + str(float((self.cfg.sample_resolution))**2/self.pressure_samples.shape[0]))
        # about 6% 

      

       
        assert(self.grad_p.shape[1]==2 and len(self.grad_p.shape)==2)


        if(bSampleBigger):
            topkratio=0.2
            specialSampleNum = int(int((self.cfg.sample_resolution))**2 * topkratio)
            
            # temp1 = checkTensorNan(self.grad_p)
            temp1=self.grad_p
      
            norm2 = temp1.square().sum(dim = 1)   # [N]
            _, random_indices = torch.topk(norm2, k = specialSampleNum)


            leftSampleNum= int((self.cfg.sample_resolution))**2 - specialSampleNum


            mask = torch.ones(norm2.shape[0], device=random_indices.device, dtype=torch.bool)
            print(mask.shape)
            mask.scatter_(0, random_indices, False)
            print(mask.shape)

            remain_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            print(remain_idx.shape)
            otherSample = remain_idx[torch.randint(0, len(remain_idx), (leftSampleNum,))]
            print(otherSample.shape)
        else:
            random_indices = torch.randint(0, self.pressure_samples.shape[0]-1, (int((self.cfg.sample_resolution))**2, ))

        global debugcnt
        global debugframe
        if(debugcnt==9997):
            # 在获取到随机下标以后再画图更合适
            TEMP_grad_p= toCpuSafe(self.grad_p)
            TEMP_random_indices = toCpuSafe(random_indices)
            TEMP_samples_arr = toCpuSafe(self.samples_arr)
            TEMP_otherSample=toCpuSafe(otherSample)

            fig = self.draw_wost_pressure(TEMP_grad_p[TEMP_random_indices][:,1], TEMP_samples_arr[TEMP_random_indices])
            save_path_pressure = os.path.join(self.vis_pressure_dir, f'selected_gradp_y_t{debugframe:03d}.png')
            save_figure_nopadding(fig, save_path_pressure)

            fig = self.draw_wost_pressure(TEMP_grad_p[TEMP_random_indices][:,0], TEMP_samples_arr[TEMP_random_indices])
            save_path_pressure = os.path.join(self.vis_pressure_dir, f'selected_gradp_x_t{debugframe:03d}.png')
            save_figure_nopadding(fig, save_path_pressure)

            if(bSampleBigger):
                fig = self.draw_wost_pressure(TEMP_grad_p[TEMP_otherSample][:,1], TEMP_samples_arr[TEMP_otherSample])
                save_path_pressure = os.path.join(self.vis_pressure_dir, f'other_gradp_y_t{debugframe:03d}.png')
                save_figure_nopadding(fig, save_path_pressure)

                fig = self.draw_wost_pressure(TEMP_grad_p[TEMP_otherSample][:,0], TEMP_samples_arr[TEMP_otherSample])
                save_path_pressure = os.path.join(self.vis_pressure_dir, f'other_gradp_x_t{debugframe:03d}.png')
                save_figure_nopadding(fig, save_path_pressure)

            debugframe+=1
            
        debugcnt+=1
        debugcnt%=10000


        samples = self.pressure_samples[random_indices]


        if(bSampleBigger):
            random_indices=torch.cat([random_indices,otherSample],dim=0)
            samples= self.pressure_samples[random_indices]

        if(bOutBC):
            samples = outBCData(samples, onlyOutData = False)
               

        with torch.no_grad():
            prev_u = self.query_velocity(samples, use_prev=True).detach()

            # print('[prev_u2] ',prev_u.shape)

       

        newnum = prev_u.shape[0] - self.grad_p[random_indices].shape[0]

        if(newnum > 0):
            assert(bOutBC)
            temp=torch.zeros((newnum,2), device=prev_u.device, dtype=prev_u.dtype)
            target_u = prev_u - torch.cat([self.grad_p[random_indices],temp], dim = 0)
        else:
            if(bEnhanceGradP):
                self.grad_p = self.grad_p * 1.5

            
            # temp = temp1[random_indices]




            # print('[sampled gradp mean ]\t'+str(torch.mean(torch.sqrt(temp.square().sum(dim = 1)))))
            target_u = prev_u - self.grad_p[random_indices]
        

            # 但是,这里的采样必须分布在全局,否则,网络对于那些缺乏数据的区域,产生的编码是混乱的(毕竟,没有任何数据在那里),
            # 所以grad p很大和grad p为零其实一样重要
            #       w采用继承参数的机制,微调,并且把上一帧的训练数据仍然保持进来     X
                        # 实测不行

 
        assert(not target_u.requires_grad)
        assert(not self.grad_p.requires_grad)



        curr_u = self.query_velocity(samples)
          
        # debug
        assert(len(curr_u.shape)==2 and curr_u.shape[1]==2)
        assert(len(target_u.shape)==2 and target_u.shape[1]==2)
        
        loss = torch.mean((curr_u - target_u) ** 2) # Eqn 11 in Rundi's paper INSR
        loss_dict = {'main': loss}

        return loss_dict

    def project_velocity(self):
        print('[PROJ]')
        self._project_velocity()
        # only use
    
    ################# visualization #####################

    def draw_wost_pressure(self, p_arr, samples):

        if torch.is_tensor(p_arr):
            p_arr= p_arr.detach().cpu().numpy()

        if torch.is_tensor(samples):
            samples= samples.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=self.fig_size)
        mask=np.isnan(p_arr)
        mask=np.isinf(p_arr) | mask

        dotsize=800
        if("kar" in self.cfg.exp_name):
            dotsize=80
        else:
            assert(False)
        sc = ax.scatter(samples[mask, 0], samples[mask, 1], color='red', s=dotsize, alpha=0.5)

        # 可变数值上下限，但颜色的极限色彩不变。
        # 如果场上没有什么太大的数值，那么数值上下限也会很小（0.005）

        sc = ax.scatter(samples[:, 0], samples[:, 1], c=p_arr, cmap='viridis', s=dotsize, alpha=0.5)
        ax.set_axis_off()
        plt.colorbar(sc)
        
        return fig