import os
from pathlib import Path

import numpy as np
import torch
from accelerate.logging import get_logger
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
from diffusers.utils.torch_utils import is_compiled_module
from omegaconf import OmegaConf
from PIL import Image

from models.utils.utils import filter_kwargs, get_image_to_video_latent_fromPIL
from training.data.dataset_cam import RealEstate10KCameraControlnetDataset
from training.pipeline_wan_fun_cam import WanFunCamControlPipeline
from training.utils import free_memory


logger = get_logger(__name__, log_level="INFO")


def stack_images_horizontally(image1: Image.Image, image2: Image.Image) -> Image.Image:
    # Ensure both images have the same height
    height = max(image1.height, image2.height)
    width = image1.width + image2.width

    # Create a new blank image with the combined width and the maximum height
    new_image = Image.new('RGB', (width, height))

    # Paste the images into the new image
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.width, 0))

    return new_image


def tensor_to_pil(img_t):
    """Convert a normalized RGB tensor, optionally with a batch dimension, to PIL."""
    if img_t.dim() == 4:
        img_t = img_t[0]

    img_t = img_t.detach().float().cpu().clamp(-1, 1)
    img_t = (img_t + 1) / 2        # [-1,1] -> [0,1]
    img_t = (img_t * 255.0).round().to(torch.uint8)

    img_np = img_t.permute(1, 2, 0).numpy()
    pil_img = Image.fromarray(img_np)
    return pil_img


def prepare_for_validation(args):

    dataset = RealEstate10KCameraControlnetDataset(
        video_root_dir=args.video_root_dir,
        npz_path=args.test_npz_path,
        image_size=(args.height, args.width),
        stride=(2,2),
        sample_n_frames=args.video_sample_n_frames,
        mode = "val"
    )

    num_samples = args.num_validation_samples
    samples = [dataset[i] for i in range(num_samples)]

    val_data = {
        "prompts": [sample["prompts"] for sample in samples],
        "clip_names": [sample["clip_name"] for sample in samples],
        "videos": [sample["videos"].unsqueeze(0) for sample in samples],
        "intrinsics": [sample["intrinsics"].unsqueeze(0) for sample in samples],
        "c2w": [sample["c2w"].unsqueeze(0) for sample in samples],
        "images": [sample["image"].unsqueeze(0) for sample in samples],
        "clip_image_pils": [sample["clip_image_pils"] for sample in samples],
    }
    return val_data


def validation_step_i2v(
        eval_data, pipe, accelerator,weight_dtype, args, config,
    ):

    video_length = args.video_sample_n_frames
    prompt, first_image = eval_data["prompts"], eval_data["image"]

    intrinsics, c2ws = eval_data["intrinsics"],  eval_data["c2w"]
    clip_image_pil = eval_data["clip_image_pil"]


    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    inpaint_video, inpaint_video_mask, _ = get_image_to_video_latent_fromPIL(
        tensor_to_pil(first_image),
        None,
        video_length=video_length,
        sample_size=[args.height, args.width],
    )

    video_generate = pipe(
        prompt,
        num_frames=video_length,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        generator=generator,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        video=inpaint_video,
        mask_video=inpaint_video_mask,
        intrinsics= intrinsics,
        c2ws = c2ws,
        clip_image=clip_image_pil,
        output_type="pil"
    ).videos[0]

    return [("generated_video", video_generate)]


def validate(
    global_step,
    accelerator,
    vae, text_encoder, tokenizer, transformer3d, controlnet,clip_image_encoder,
    args, config, weight_dtype,
    **kwargs,
):


    logger.info("Running validation... ")


    num_validation_samples = args.num_validation_samples

    if num_validation_samples == 0:
        logger.warning("No validation samples found. Skipping validation.")
        return


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    torch.set_grad_enabled(False)

    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )
    pipe = WanFunCamControlPipeline(
        transformer=unwrap_model(transformer3d),
        vae=unwrap_model(vae).to(weight_dtype),
        text_encoder=unwrap_model(text_encoder),
        controlnet=unwrap_model(controlnet),
        tokenizer=tokenizer,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    )
    pipe.enable_model_cpu_offload(device=accelerator.device)

    for i in range(accelerator.process_index, num_validation_samples, accelerator.num_processes):
        image_name = kwargs["clip_names"] [i]
        video = kwargs["videos"][i]

        sample_dict = {
            "prompts": kwargs["prompts"][i],
            "intrinsics": kwargs["intrinsics"][i],
            "c2w": kwargs["c2w"][i],
            "image": kwargs["images"][i],                     
            "clip_image_pil": kwargs["clip_image_pils"][i],
        }
        with torch.no_grad():
            validation_artifacts = validation_step_i2v(
                sample_dict,
                pipe,
                accelerator,
                weight_dtype,
                args,
                config,
            )


        video_tensor = video.squeeze(0).clamp(-1, 1)        # [T, 3, H, W]
        video_tensor = (video_tensor + 1.0) / 2.0           # [-1, 1] → [0, 1]
        video_tensor = (video_tensor * 255).to(torch.uint8)
        video_frames = [
            Image.fromarray(frame.permute(1, 2, 0).cpu().numpy())
            for frame in video_tensor
        ]
        artifacts = {
            "gt": {"type": "video", "value": video_frames},
        }
        for artifact_type, artifact_value in validation_artifacts:
            if artifact_type == "generated_video":
                artifacts["generated_video"] = {"type": "video", "value": artifact_value}


        output_dir = Path(args.output_dir)
        validation_path = output_dir / "val_results" / f"step_{global_step}"
        validation_path.mkdir(parents=True, exist_ok=True)

        if "gt" in artifacts and "generated_video" in artifacts:
            gt_frames = artifacts["gt"]["value"]
            generated_frames = artifacts["generated_video"]["value"]

            assert len(gt_frames) == len(generated_frames), "Input and generated videos must have the same frame count"

            concatenated_frames = [
                stack_images_horizontally(in_f, gen_f)
                for in_f, gen_f in zip(gt_frames, generated_frames)
            ]

            concat_filename = f"concat_step{global_step}_sample{i}_{image_name}.mp4"
            concat_filepath = str(validation_path / "cat_gt_out" / concat_filename)
            os.makedirs(os.path.dirname(concat_filepath), exist_ok=True)

            logger.info(f"Saving concatenated video to {concat_filepath}")
            export_to_video(concatenated_frames, concat_filepath, fps=args.gen_fps)

        for key, value in artifacts.items():
            artifact_type = value["type"]
            artifact_value = value["value"]
            if artifact_type not in ["image", "video"] or artifact_value is None:
                continue

            extension = "mp4"  
            filename = f"{key}_step{global_step}_sample{i}_{image_name}.{extension}"
            filename = str(validation_path / f'{key}'/filename)
            os.makedirs( os.path.dirname(filename), exist_ok=True)

            logger.info(f"Saving {key} to {filename}")
            export_to_video(artifact_value, filename, fps=args.gen_fps)



    del pipe
    del scheduler
    vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    transformer3d.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    free_memory()
    accelerator.wait_for_everyone()
    torch.set_grad_enabled(True)
