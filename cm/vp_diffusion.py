import random

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import functools
import math
from piq import LPIPS
from torchvision.transforms import RandomCrop
from . import dist_util
from einops import rearrange, reduce, repeat
from tqdm.auto import tqdm

from .nn import mean_flat, append_dims, append_zero, right_pad_dims_to
from .random_util import get_generator
import torch.distributed as dist


# alpha schedules
def simple_linear_schedule(t, clip_min = 1e-9):
    return (1 - t).clamp(min = clip_min)

def beta_linear_schedule(t, clip_min = 1e-9):
    return th.exp(-1e-4 - 10 * (t ** 2)).clamp(min = clip_min, max = 1.)

def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = th.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    v_start = th.tensor(start / tau).sigmoid()
    v_end = th.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)

#snr and time to alpha funcs
def log_snr_to_alpha(log_snr):
    alpha = th.sigmoid(log_snr)
    return alpha

def alpha_to_shifted_log_snr(alpha, scale = 1):
    return th.log((alpha / (1 - alpha))).clamp(min=-15, max=15) + 2*np.log(scale).item()

def time_to_alpha(t, alpha_schedule, scale):
    alpha = alpha_schedule(t)
    shifted_log_snr = alpha_to_shifted_log_snr(alpha, scale = scale)
    return log_snr_to_alpha(shifted_log_snr)

def scaling_for_ddim_boundry(tk, t):
    c_skip = t//tk
    ratio = t/tk
    c_out = th.where(ratio<1, th.tensor(1, dtype=tk.dtype, device=tk.device), th.tensor(0, dtype=tk.dtype, device=tk.device))
    return c_skip, c_out

def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
    return c_skip, c_out

class ModelPrediction:
    def __init__(
        self,
        pred_noise=None,
        pred_x_start=None,
        pred_x_s=None,
        pred_v=None
    ):
        self.pred_noise = pred_noise
        self.pred_x_start = pred_x_start
        self.pred_x_s = pred_x_s
        self.pred_v = pred_v

class VPDenoiser:
    def __init__(
        self,
        alpha_schedule,
        distillation=False,
        loss_norm="lpips",
        objective="noise",
        scale=1,
        num_timesteps=100
    ):
        if loss_norm == "lpips":
            self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")
        self.distillation = distillation
        self.objective = objective
        if alpha_schedule == "simple_linear":
            alpha_schedule = simple_linear_schedule
        elif alpha_schedule == "beta_linear":
            alpha_schedule = beta_linear_schedule
        elif alpha_schedule == "cosine":
            alpha_schedule = cosine_schedule
        elif alpha_schedule == "sigmoid":
            alpha_schedule = sigmoid_schedule
        else:
            raise ValueError(f'invalid alpha schedule {alpha_schedule}')
        self.loss_norm = loss_norm
        self.alpha_schedule = functools.partial(time_to_alpha, alpha_schedule=alpha_schedule, scale=scale)
        self.num_timesteps = num_timesteps
        self.denoise = self.diffusion_model_output if not distillation else self.consistency_model_output
    
    def predict_start_from_noise(self, z_t, alpha_sq, noise):
        alpha_sq = right_pad_dims_to(z_t, alpha_sq)
        return (z_t - (1-alpha_sq).sqrt() * noise) / alpha_sq.sqrt().clamp(min = 1e-8)
        
    def predict_noise_from_start(self, z_t, alpha_sq, x0, sampling=False):
        alpha_sq = right_pad_dims_to(z_t, alpha_sq)
        return (z_t - alpha_sq.sqrt() * x0) / (1-alpha_sq).sqrt().clamp(min = 1e-8)
    
    def diffusion_model_output(self, model, x_t, t, alpha_sq=None, **model_kwargs):
        if alpha_sq is None:
            alpha_sq = self.alpha_schedule(t)
        model_output = model(x_t, alpha_sq, **model_kwargs)
        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x_t, alpha_sq, pred_noise)
        elif self.objective == "pred_x0":
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x_t, alpha_sq, x_start)
        else:
            raise NotImplementedError
        return ModelPrediction(pred_noise=pred_noise, pred_x_start=x_start)
    
    def consistency_model_output(self, model, x_t, t, alpha_sq=None, **model_kwargs):
        if alpha_sq is None:
            alpha_sq = self.alpha_schedule(t)
        model_output = model(x_t, alpha_sq, **model_kwargs)
        c_skip, c_out = scalings_for_boundary_conditions(t)
        c_skip, c_out = [append_dims(x, x_t.ndim) for x in [c_skip, c_out]]
        model_output = c_skip*x_t + c_out*model_output
        return model_output

    def get_sampling_timesteps(self, batch, *, device, invert = False):
        times = th.linspace(1., 0., self.num_timesteps + 1, device = device)
        if invert:
            times = times.flip(dims = (0,))
        times = repeat(times, 't -> b t', b = batch)
        times = th.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times
    
    def get_alpha_sigma(self, t, s, x):
        alpha_t = self.alpha_schedule(t)
        alpha_s = self.alpha_schedule(s)
        alpha_t, alpha_s = map(functools.partial(right_pad_dims_to, x), (alpha_t, alpha_s))
        sigma_t = (1-alpha_t)
        sigma_s = (1-alpha_s)
        return alpha_t.sqrt(), alpha_s.sqrt(), sigma_t.sqrt(), sigma_s.sqrt()
    
    def __ddim(self, x_t, pred_x, t, s):
        alpha_t, alpha_s, sigma_t, sigma_s = self.get_alpha_sigma(t, s, x_t)
        c_skip, c_out = scaling_for_ddim_boundry(t, s)
        c_skip = append_dims(c_skip, x_t.ndim)
        c_out = append_dims(c_out, x_t.ndim)
        out = alpha_s*pred_x + ((sigma_s/sigma_t)*(x_t-alpha_t*pred_x))
        return c_skip*x_t + c_out*out

    def __invDDIM(self, x_s, x_t, t, s):
        alpha_t, alpha_s, sigma_t, sigma_s = self.get_alpha_sigma(t, s, x_s)
        x = (x_s-((sigma_s/sigma_t)*x_t))/(alpha_s-(alpha_t*(sigma_s/sigma_t)))
        return x
    
    def __aDDIM(self, x_t, pred_x, ground_x, t, s, d=None, n=None):
        if d is None:
            d = x_t.ndim
        if n is None:
            n = 0.75
        alpha_t, alpha_s, sigma_t, sigma_s = self.get_alpha_sigma(t, s, x_t)
        x_var = n*(th.norm(pred_x-ground_x)**2)/d
        #x_var = 0.1/(2+(alpha_tk**2)/(sigma_tk**2))
        eps = (x_t-alpha_t*pred_x)/sigma_t
        x_s_var = (alpha_s-alpha_t*sigma_s/sigma_t)**2 * x_var #TODO: check if this formula is right
        var_scale_raw = (sigma_s**2)+(d/th.norm(eps)**2)*x_s_var
        var_scale = th.sqrt(var_scale_raw)
        x_s = alpha_s*pred_x+var_scale*eps
        return x_s

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        terms = {}
        
        alpha_sq = self.alpha_schedule(t)
        alpha_sq_append = append_dims(alpha_sq, x_start.ndim)

        x_t = alpha_sq_append.sqrt()*x_start + (1-alpha_sq_append).sqrt()*noise
        #consider writting a wrapper self.denoise that way it can be reused for SCM
        model_output = self.diffusion_model_output(model, x_t, t, alpha_sq, **model_kwargs)
        
        if self.objective == "pred_x0":
            target = x_start
            pred = model_output.pred_x_start
        elif self.objective == "pred_noise":
            target = noise
            pred = model_output.pred_noise#
        else:
            raise NotImplementedError
        
        terms["loss"] = mean_flat((pred - target) ** 2)
        return terms
    
    def cd_loss(self, model, target_model, diffusion_model, x_start, device, k=1, t=None, s=None, alpha_sq=None, noise=None,**model_kwargs):
        #t and s calculated in parent func
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        terms = {}

        if t is None or s is None:
            raw_s = th.randint(0,self.num_timesteps-k+1, (x_start.shape[0],), device=device).float()
            raw_t = raw_s + k
            s = raw_s/self.num_timesteps
            t = raw_t/self.num_timesteps
        
        alpha_t, alpha_s, sigma_t, sigma_s = self.get_alpha_sigma(t, s, x_start)  
        noise = th.randn_like(x_start)
        x_t = alpha_t*x_start+sigma_t*noise   

        with th.no_grad():
            x_diffusion = self.diffusion_model_output(diffusion_model, x_t=x_t, t=t).pred_x_start
            x_s = self.__ddim(x_t, x_diffusion, t, s)
            x_0_s = self.consistency_model_output(target_model, x_s, s) #TODO
        x_0_t = self.consistency_model_output(model, x_t, t)

        if self.loss_norm == "l1":
            diffs = th.abs(x_0_t - x_0_s)
            loss = mean_flat(diffs)
        elif self.loss_norm == "l2":
            diffs = (x_0_t - x_0_s) ** 2
            loss = mean_flat(diffs)
        elif self.loss_norm == "l2-32":
            x_0_t = F.interpolate(x_0_t, size=32, mode="bilinear")
            x_0_s = F.interpolate(
                x_0_s,
                size=32,
                mode="bilinear",
            )
            diffs = (x_0_t - x_0_s) ** 2
            loss = mean_flat(diffs)
        elif self.loss_norm == "lpips":
            if x_start.shape[-1] < 256:
                x_0_t = F.interpolate(x_0_t, size=224, mode="bilinear")
                x_0_s = F.interpolate(
                    x_0_s, size=224, mode="bilinear"
                )
            loss = (
                self.lpips_loss(
                    (x_0_t + 1) / 2.0,
                    (x_0_s + 1) / 2.0,
                )
            )
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")
        
        terms = {}
        terms["loss"] = loss

        return terms
    
    def scd_loss(self, model, target_model, diffusion_model, x_start, device, k=1, t=None, s=None, alpha_sq=None, noise=None, t_scalar=0.1, is_addim=False, steps=4, **model_kwargs):
        #t and s calculated in parent func
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        terms = {}

        #ni = self.n_scedule(train_i) #TODO
        ni = self.num_timesteps
        T_step = int(np.round(ni/steps))
        step = th.randint(0, steps, (x_start.shape[0],), device=device).float()
        i = th.randint(0, math.ceil(T_step*t_scalar), (x_start.shape[0],), device=device).float()
        stage_end = step/steps
        t = (stage_end + (i+1)/ni)
        s = stage_end
        alpha_t, alpha_s, sigma_t, sigma_s = self.get_alpha_sigma(t, s, x_start)

        noise = th.randn_like(x_start)
        x_t = alpha_t*x_start+sigma_t*noise  

        with th.no_grad():
            x_diffusion = self.diffusion_model_output(diffusion_model, x_t=x_t, t=t).pred_x_start
            #if is_addim:
            #    x_s = self.__aDDIM(x_t, x_diffusion, x_start, t, s, x_t.shape[-1]**2) #d is assumed to be width^2. Only square images
            #else:
            x_s = self.__ddim(x_t, x_diffusion, t, s)
        
        x_0_t = self.consistency_model_output(model, x_t, t)
        x_0_s = self.__invDDIM(x_s, x_t, t, s)

        if self.loss_norm == "l1":
            diffs = th.abs(x_0_t - x_0_s)
            loss = mean_flat(diffs)
        elif self.loss_norm == "l2":
            diffs = (x_0_t - x_0_s) ** 2
            loss = mean_flat(diffs)
        elif self.loss_norm == "l2-32":
            x_0_t = F.interpolate(x_0_t, size=32, mode="bilinear")
            x_0_s = F.interpolate(
                x_0_s,
                size=32,
                mode="bilinear",
            )
            diffs = (x_0_t - x_0_s) ** 2
            loss = mean_flat(diffs)
        elif self.loss_norm == "lpips":
            if x_start.shape[-1] < 256:
                x_0_t = F.interpolate(x_0_t, size=224, mode="bilinear")
                x_0_s = F.interpolate(
                    x_0_s, size=224, mode="bilinear"
                )
            loss = (
                self.lpips_loss(
                    (x_0_t + 1) / 2.0,
                    (x_0_s + 1) / 2.0,
                )
            )
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")
        
        terms = {}
        terms["loss"] = loss

        return terms

#TODO: refactor VPSampler to be part of VPDenoiser
class VPSampler:
    def __init__(
        self,
        diffusion: VPDenoiser,
        model,
        shape,
        sampler="ddim",
        num_timesteps=None,
        clip_denoised=True,
        progress=False,
        callback=None,
        model_kwargs=None,
        device=None,
        training_mode="edm",
        cm_steps=4,
    ):
        self.diffusion = diffusion
        self.model = model
        self.shape = shape
        if num_timesteps is None:
            num_timesteps = diffusion.num_timesteps
        self.num_timesteps = num_timesteps
        self.clip_denoised = clip_denoised
        self.progress = progress
        self.callback = callback
        self.model_kwargs = model_kwargs
        self.device = device
        self.sample = {
            "ddim": self.sample_ddim,
            "onestep": self.one_step,
            "scd": self.sample_scd
        }[sampler]
        self.training_mode=training_mode
        self.cm_steps=cm_steps
    
    def denoise(self, x_t, t, **model_kwargs):
        denoised = self.diffusion.denoise(self.model, x_t, t, **self.model_kwargs)
        if self.training_mode == "edm":
            denoised = denoised.pred_x_start
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    def get_sampling_timesteps(self, batch, *, device, invert = False):
        times = th.linspace(1., 0., self.num_timesteps + 1, device = device)
        if invert:
            times = times.flip(dims = (0,))
        times = repeat(times, 't -> b t', b = batch)
        times = th.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times
    
    def get_alpha_sigma(self, t, s, x):
        alpha_t = self.diffusion.alpha_schedule(t)
        alpha_s = self.diffusion.alpha_schedule(s)
        alpha_t, alpha_s = map(functools.partial(right_pad_dims_to, x), (alpha_t, alpha_s))
        sigma_t = (1-alpha_t)
        sigma_s = (1-alpha_s)
        return alpha_t.sqrt(), alpha_s.sqrt(), sigma_t.sqrt(), sigma_s.sqrt()
    
    def __ddim(self, x_t, pred_x, t, s):
        alpha_t, alpha_s, sigma_t, sigma_s = self.get_alpha_sigma(t, s, x_t)
        c_skip, c_out = scaling_for_ddim_boundry(t, s)
        c_skip = append_dims(c_skip, x_t.ndim)
        c_out = append_dims(c_out, x_t.ndim)
        out = alpha_s*pred_x + ((sigma_s/sigma_t)*(x_t-alpha_t*pred_x))
        return c_skip*x_t + c_out*out
    
    def __invDDIM(self, x_s, x_t, t, s):
        alpha_t, alpha_s, sigma_t, sigma_s = self.get_alpha_sigma(t, s, x_s)
        x = (x_s-((sigma_s/sigma_t)*x_t))/(alpha_s-(alpha_t*(sigma_s/sigma_t)))
        return x
    
    @th.no_grad()
    def sample_ddim(self):
        time_pairs = self.get_sampling_timesteps(self.shape[0], device = self.device) #TODO: shape[0] gives batch
        x_t = th.randn(self.shape, device=self.device)
        for t, s in tqdm(time_pairs, desc = 'sampling loop time step', total = self.num_timesteps):
            denoised = self.denoise(x_t=x_t, t=t)
            x_t = self.__ddim(x_t, denoised, t, s)
        return x_t
    
    @th.no_grad()
    def one_step(self):
        x_t = th.randn(self.shape, device=self.device)
        t = th.ones(self.shape[0], device=self.device)
        return self.denoise(x_t=x_t, t=t)
    
    @th.no_grad()
    def sample_scd(self):
        x_t = th.randn(self.shape, device=self.device)
        for i in range(self.cm_steps):
            t = 1-(i/self.cm_steps)
            s = t-(1/self.cm_steps)
            t = th.tensor(t, device=self.device).unsqueeze(0)
            s = th.tensor(s, device=self.device).unsqueeze(0)
            denoised = self.denoise(x_t=x_t, t=t)
            x_t = self.__ddim(x_t, denoised, t, s)
        return x_t



    
        
