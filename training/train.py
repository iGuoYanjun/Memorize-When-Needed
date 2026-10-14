import argparse
import gc
import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_loss_weighting_for_sd3
from diffusers.utils import check_min_version
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import AutoTokenizer


project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.wan_modules import AutoencoderKLWan, CLIPModel, WanT5EncoderModel
from models.controlnets.wan_controlnet_camera import WanControlnet
from models.controlnets.wan_transformer_dual import CustomWanTransformer3DModel
from models.utils.utils import filter_kwargs
from training.data.collate import collate_fn
from training.data.dataset_cam import RealEstate10KCameraControlnetDataset
from training.utils import (
    free_memory,
    resize_mask,
    sample_high_noise_timesteps_keep_linear_time_dist,
)
from training.validation import prepare_for_validation, validate


check_min_version("0.18.0.dev0")
logger = get_logger(__name__, log_level="INFO")


def linear_decay(initial_value, final_value, total_steps, current_step):
    """Interpolate from the initial value to the final value over training steps."""
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value


def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
    """Compute weighted MSE while excluding residuals above the threshold."""
    noise_pred = noise_pred.float()
    target = target.float()
    diff = noise_pred - target
    mse_loss = F.mse_loss(noise_pred, target, reduction='none')
    mask = (diff.abs() <= threshold).float()
    masked_loss = mse_loss * mask
    if weighting is not None:
        masked_loss = masked_loss * weighting
    final_loss = masked_loss.mean()
    return final_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--video_root_dir",
        type=str,
        default=None,
        required=True,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        default=None,
        required=False,
        help=("Relative path to the npz file."),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="All input videos are resized to this height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="All input videos are resized to this width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=81,
        help="Num frame of video.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--vae_mini_batch", type=int, default=1, help="mini batch size for vae."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")##全参数1e-4,dmd的时候0.01
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save model weights every X optimization updates.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "The config of the model in training."
        ),
    )
    parser.add_argument(
        "--controlnet_config_path",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "controlnet.yaml"),
        help="Path to the configuration of the ControlNet to train.",
    )
    parser.add_argument(
        '--tokenizer_max_length',
        type=int,
        default=512,
        help='Max length of tokenizer'
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    parser.add_argument(
        "--abnormal_norm_clip_start",
        type=int,
        default=1000,
        help=(
            'When do we start doing additional processing on abnormal gradients. '
        ),
    )
    parser.add_argument(
        "--initial_grad_norm_ratio",
        type=int,
        default=5,
        help=(
            'The initial gradient is relative to the multiple of the max_grad_norm. '
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--do_validation",
        action="store_true",
        default=False
    )

    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走,场景突变",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--num_validation_samples",
        type=int,
        default=10,
        required=False,
        help=("Number of validation samples."),
    )
    parser.add_argument(
        "--controlnet_stride",
        type=int,
        default=1,
        help="Stride of the controlnet.",
    )

    parser.add_argument("--gen_fps", type=int, default=8, help="All generated videos will be used at this FPS.")
    parser.add_argument(
        "--test_npz_path",
        type=str,
        default=None,
        required=False,
        help=("Relative path to the npz file."),
    )
    parser.add_argument(
        "--multi_stream",
        action="store_true",
        help="whether to use cuda multi-stream",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def main():
    args = parse_args()

    config = OmegaConf.load(args.config_path)
    controlnet_kwargs = OmegaConf.to_container(
        OmegaConf.load(args.controlnet_config_path), resolve=True
    )

    from accelerate.utils import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
        # find_unused_parameters=True,#!!!!!!!!!!!!!!!!!!!
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None
    index_rng = np.random.default_rng(np.random.PCG64(43))
    print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Tokenizer

    accelerator.print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    accelerator.print("Loading text encoder...")
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path,
                    config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()

    accelerator.print("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path,
                    config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    )
    vae.eval()

    accelerator.print("Loading CLIP image encoder...")
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path,
                    config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    )
    clip_image_encoder = clip_image_encoder.eval()


    accelerator.print("Loading Wan transformer...")
    transformer3d = CustomWanTransformer3DModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage = True,
    ).to(weight_dtype)

    accelerator.print("Initializing camera ControlNet...")
    controlnet = WanControlnet(**controlnet_kwargs).to(weight_dtype)

    # Freeze vae and text_encoder and set transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)
    controlnet.requires_grad_(True)
    clip_image_encoder.requires_grad_(False)
    for name, p in controlnet.named_parameters():
        if "condition_embedder.text_embedder" in name:
            p.requires_grad_(False)

    accelerator.print("Moving models to training devices...")
    vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    clip_image_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    transformer3d.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)


    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()


    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, controlnet.parameters()))
    trainable_parameters_with_lr = {"params": trainable_params, "lr": args.learning_rate}
    trainable_params_optim = [trainable_parameters_with_lr]

    optimizer = optimizer_cls(
        trainable_params_optim,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    train_dataset = RealEstate10KCameraControlnetDataset(
        video_root_dir=args.video_root_dir,
        npz_path=args.npz_path,
        image_size=(args.height, args.width),
        stride=(2,2),
        sample_n_frames=args.video_sample_n_frames,
    )


    def worker_init_fn(_seed):
        _seed = _seed * 256
        def _worker_init_fn(worker_id):
            # print(f"worker_init_fn with {_seed + worker_id}")
            np.random.seed(_seed + worker_id)
            random.seed(_seed + worker_id)
            torch.manual_seed(_seed + worker_id)
        return _worker_init_fn

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        persistent_workers=True if args.dataloader_num_workers != 0 else False,
        num_workers=args.dataloader_num_workers,
        worker_init_fn=worker_init_fn(args.seed + accelerator.process_index),
        # pin_memory=True
    )


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )


    # Move text_encode and vae to gpu and cast to weight_dtype

        ##################=====================================#########

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    initial_global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if args.multi_stream :
        # create extra cuda streams to speedup inpaint vae computation
        vae_stream_1 = torch.cuda.Stream()
        vae_stream_2 = torch.cuda.Stream()
    else:
        vae_stream_1 = None
        vae_stream_2 = None


    for epoch in range(first_epoch, args.num_train_epochs):
        controlnet.train()
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [controlnet]
            with accelerator.accumulate(models_to_accumulate):
                # Convert images to latent space
                clip_image_pils = batch["clip_image_pils"]
                pixel_values = batch["videos"].to(weight_dtype)
                B, Frame, _, H, W = pixel_values.shape
                intrinsics = batch["intrinsics"]
                c2ws = batch["c2w"]

                mask = torch.ones_like(pixel_values[:, :, :1])
                mask[:, 0] = 0.0

                mask_pixel_values = pixel_values * (1 - mask)


                if args.low_vram:
                    torch.cuda.empty_cache()
                    vae.to(accelerator.device)
                    clip_image_encoder.to(accelerator.device)
                    text_encoder.to("cpu")

                with torch.no_grad():
                    # This way is quicker when batch grows up
                    def _batch_encode_vae(pixel_values):
                        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                        bs = args.vae_mini_batch
                        new_pixel_values = []
                        for i in range(0, pixel_values.shape[0], bs):
                            pixel_values_bs = pixel_values[i : i + bs]
                            pixel_values_bs = vae.encode(pixel_values_bs)[0]
                            pixel_values_bs = pixel_values_bs.sample()
                            new_pixel_values.append(pixel_values_bs)
                        return torch.cat(new_pixel_values, dim = 0)
                    if vae_stream_1 is not None:
                        vae_stream_1.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(vae_stream_1):
                            latents = _batch_encode_vae(pixel_values)
                    else:
                        latents = _batch_encode_vae(pixel_values)

                    mask = rearrange(mask, "b f c h w -> b c f h w")
                    mask = torch.concat(
                        [
                            torch.repeat_interleave(mask[:, :, 0:1], repeats=4, dim=2),
                            mask[:, :, 1:]
                        ], dim=2
                    )
                    mask = mask.view(mask.shape[0], mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4])
                    mask = mask.transpose(1, 2)

                    mask = resize_mask(1 - mask, latents.size())
                    # Encode inpaint latents.
                    mask_latents = _batch_encode_vae(mask_pixel_values)
                    if vae_stream_2 is not None:
                        torch.cuda.current_stream().wait_stream(vae_stream_2)
                    inpaint_latents = torch.concat([mask, mask_latents], dim=1)

                    clip_context = []
                    for clip_image_pil in clip_image_pils:
                        # clip_image = Image.fromarray(np.uint8(clip_pixel_value.float().cpu().numpy()))
                        clip_image = TF.to_tensor(clip_image_pil).sub_(0.5).div_(0.5).to(clip_image_encoder.device, weight_dtype)
                        _clip_context = clip_image_encoder([clip_image[:, None, :, :]])
                        clip_context.append(_clip_context)
                    clip_context = torch.cat(clip_context)


                # wait for latents = vae.encode(pixel_values) to complete
                if vae_stream_1 is not None:
                    torch.cuda.current_stream().wait_stream(vae_stream_1)

                if args.low_vram:
                    vae.to('cpu')
                    clip_image_encoder.to('cpu')
                    torch.cuda.empty_cache()
                    text_encoder.to(accelerator.device)

                with torch.no_grad():
                    prompts = batch["prompts"]
                    prompts = [
                        "" if random.random() < 0.1 else p
                        for p in prompts
                    ]
                    prompt_ids = tokenizer(
                        prompts,
                        padding="max_length",
                        max_length=args.tokenizer_max_length,
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors="pt"
                    )
                    text_input_ids = prompt_ids.input_ids
                    prompt_attention_mask = prompt_ids.attention_mask

                    seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                    prompt_embeds = text_encoder(text_input_ids.to(latents.device), attention_mask=prompt_attention_mask.to(latents.device))[0]
                    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

                if args.low_vram:
                    text_encoder.to('cpu')
                    torch.cuda.empty_cache()

                bsz, channel, num_frames, height, width = latents.size()
                noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)


                timesteps, sigmas = sample_high_noise_timesteps_keep_linear_time_dist(
                    noise_scheduler, bsz,
                    mean=0.95, std=0.1,
                    a=0.6,
                    b=1.0,
                    device=latents.device,
                    dtype=latents.dtype,
                    n_dim=latents.ndim,
                )

                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                # Add noise
                target = noise - latents

                target_shape = (vae.latent_channels, num_frames,  height,width)
                seq_len = math.ceil(
                    (target_shape[2] * target_shape[3]) /
                    (accelerator.unwrap_model(transformer3d).config.patch_size[1] * accelerator.unwrap_model(transformer3d).config.patch_size[2]) *
                    target_shape[1]
                )


                # Predict the noise residual
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                    controlnet_states = controlnet(
                        hidden_states=noisy_latents,
                        y=inpaint_latents,
                        encoder_hidden_states=prompt_embeds,
                        intrinsics=intrinsics,
                        c2ws=c2ws,
                        timestep=timesteps,
                        frame_num=Frame,
                        return_dict=False,
                    )[0]
                    if isinstance(controlnet_states, (tuple, list)):
                        controlnet_states = [x.to(dtype=weight_dtype) for x in controlnet_states]
                    else:
                        controlnet_states = controlnet_states.to(dtype=weight_dtype)


                    noise_pred = transformer3d(
                        x=noisy_latents,
                        context=prompt_embeds,
                        t=timesteps,
                        seq_len=seq_len,
                        y=inpaint_latents,
                        controlnet_states=controlnet_states,
                        controlnet_stride=args.controlnet_stride,
                        clip_fea=clip_context,
                    )


                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                loss = custom_mse_loss(noise_pred.float(), target.float(), weighting.float())
                loss = loss.mean()

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    trainable_params_grads = [p.grad for p in trainable_params if p.grad is not None]
                    trainable_params_total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in trainable_params_grads]), 2)
                    max_grad_norm = linear_decay(args.max_grad_norm * args.initial_grad_norm_ratio, args.max_grad_norm, args.abnormal_norm_clip_start, global_step)
                    if trainable_params_total_norm / max_grad_norm > 5 and global_step > args.abnormal_norm_clip_start:
                        actual_max_grad_norm = max_grad_norm / min((trainable_params_total_norm / max_grad_norm), 10)
                    else:
                        actual_max_grad_norm = max_grad_norm
                    norm_sum = accelerator.clip_grad_norm_(trainable_params, actual_max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # accelerator.save_state(save_path)
                        unwrap_model(controlnet).save_pretrained(save_path)
                        logger.info(f"Saved state to {save_path}")

                should_run_validation = args.do_validation and global_step % args.validation_steps == 0
                if should_run_validation:
                    del loss
                    free_memory()
                    val_data = prepare_for_validation(args)

                    validate(
                        global_step,
                        accelerator,
                        vae, text_encoder, tokenizer, transformer3d, controlnet,clip_image_encoder,
                        args,config, weight_dtype,
                        **val_data,
                    )

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        # accelerator.save_state(save_path)
        unwrap_model(controlnet).save_pretrained(save_path)
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
