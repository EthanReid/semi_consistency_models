"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image

from cm import dist_util, logger
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample
from cm.vp_diffusion import VPSampler


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.out_dir, args=args)

    if args.training_mode == "progdist" or args.training_mode == "edm":
        distillation = False
    else:
        distillation = True

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    all_images = []
    all_labels = []
    generator = get_generator(args.generator, args.num_samples, args.seed)

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        """x
        sample = karras_sample(
            diffusion,
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            steps=args.steps,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            s_churn=args.s_churn,
            s_tmin=args.s_tmin,
            s_tmax=args.s_tmax,
            s_noise=args.s_noise,
            generator=generator,
            ts=ts,
            vp=args.vp
        )
        """
        sampler = VPSampler(
            diffusion=diffusion,
            model=model,
            shape=(args.batch_size, 3, args.image_size, args.image_size),
            sampler=args.sampler,
            num_timesteps=args.num_timesteps,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            training_mode=args.training_mode,
            cm_steps=args.cm_steps
        )
        sample = sampler.sample()
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
           np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
        img_dir = os.path.join(logger.get_dir(), args.sampler, "images")
        os.makedirs(img_dir, exist_ok=True)
        for i, image_array in enumerate(arr):
            image = Image.fromarray(image_array)
            image_path = os.path.join(img_dir, f"sample_{i}.png")
            image.save(image_path)
            logger.log(f"saved image {i} to {image_path}")

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="",
        out_dir="",
        project_name="openai_cm",
        run_name="edm",
        alpha_schedule="cosine",
        objective="pred_noise",
        num_timesteps=100,
        cm_steps=4
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
