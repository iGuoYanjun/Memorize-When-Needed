import argparse
import os
import random
import re
import shutil
import sys
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video, load_video
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(current_file_path),
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from dataload.datasets import MemDataset
from pipeline import WanMemPipeline
from models.controlnets.wan_controlnet_camera import WanControlnet
from models.controlnets.wan_controlnet_mem import WanHistoryControlnet
from models.controlnets.wan_transformer_dual import CustomWanTransformer3DModel
from models.wan_modules import AutoencoderKLWan, CLIPModel, WanT5EncoderModel
from tqdm import tqdm
from models.utils.utils import get_image_to_video_latent_fromPIL

def filter_kwargs(cls, kwargs):
    import inspect

    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "cls"}
    return {k: v for k, v in kwargs.items() if k in valid_params}


def build_reference_frames(control_video, frame_ids_image):
    if frame_ids_image is None or not control_video:
        return None

    frame_ids_list = frame_ids_image.squeeze(1).tolist()
    ref_img = control_video[0]
    width, height = ref_img.size
    black_frame = Image.new("RGB", (width, height), (0, 0, 0))

    selected_frames = []
    for frame_id in frame_ids_list:
        frame_id = int(frame_id)
        if frame_id == -1:
            selected_frames.append(black_frame)
        elif 0 <= frame_id < len(control_video):
            selected_frames.append(control_video[frame_id])
        else:
            selected_frames.append(black_frame)

    return selected_frames


@torch.no_grad()
def generate_video(args: argparse.Namespace, dtype: torch.dtype = torch.bfloat16):
    os.makedirs(args.output_path, exist_ok=True)
    weight_dtype = dtype
    config = OmegaConf.load(args.config_path)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            args.pretrained_model_name_or_path,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
        )
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            args.pretrained_model_name_or_path,
            config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).eval()

    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(
            args.pretrained_model_name_or_path,
            config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"),
        )
    ).eval()

    transformer3d = CustomWanTransformer3DModel.from_pretrained(
        os.path.join(
            args.pretrained_model_name_or_path,
            config["transformer_additional_kwargs"].get("transformer_subpath", "transformer"),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True,
    ).to(weight_dtype)
    transformer3d.eval()

    controlnet = WanControlnet.from_pretrained(args.controlnet_ckpt_dir, low_cpu_mem_usage=True).to(weight_dtype).eval()
    history_controlnet = (
        WanHistoryControlnet.from_pretrained(args.history_controlnet_ckpt, low_cpu_mem_usage=True)
        .to(weight_dtype)
        .eval()
    )

    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config["scheduler_kwargs"]))
    )

    pipe = WanMemPipeline(
        transformer=transformer3d,
        vae=vae.to(weight_dtype),
        text_encoder=text_encoder,
        controlnet=controlnet,
        history_controlnet=history_controlnet,
        tokenizer=tokenizer,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    )

    pipe = pipe.to(dtype=dtype)
    pipe.enable_model_cpu_offload(device="cuda")
    pipe.set_progress_bar_config()

    eval_dataset = MemDataset(
        metadata_path=args.metadata_path,
        image_size=(args.height, args.width),
        sample_n_frames=args.video_sample_n_frames,
    )
    num_samples = len(eval_dataset)
    if num_samples == 0:
        raise ValueError("The metadata file does not contain any samples.")

   

    temp_path = Path(args.output_path + "_temp_vae")
    temp_path.mkdir(parents=True, exist_ok=True)
    seed = args.seed if args.seed >= 0 else random.randint(0, sys.maxsize)

    for sample_idx in tqdm(range(num_samples), desc="Processing samples"):
        data_dict = eval_dataset[sample_idx]
        first_image_pil = to_pil_image(data_dict["image"])
        prompt = data_dict["prompt"]
        all_frame_ids_image = data_dict["frame_ids_image"]
        all_intrinsics = data_dict["intrinsics"]
        all_c2ws = data_dict["c2w"]
        clip_name = data_dict["clip_name"]

        generate_dir = os.path.join(args.output_path, f"{sample_idx:04d}_{clip_name}")
        os.makedirs(generate_dir, exist_ok=True)
        temp_path_video = temp_path / f"{sample_idx:04d}_{clip_name}"
        temp_path_video.mkdir(parents=True, exist_ok=True)

        long_videos = [
            filename
            for filename in os.listdir(generate_dir)
            if filename.startswith("long_video_turn_") and filename.endswith(".mp4")
        ]
        if long_videos:
            turns = []
            for filename in long_videos:
                match = re.search(r"long_video_turn_(\d+)\.mp4", filename)
                if match:
                    turns.append(int(match.group(1)))
            resume_turn = max(turns)
            resume_long_video_path = os.path.join(generate_dir, f"long_video_turn_{resume_turn}.mp4")
            print(f"[Resume] Found existing turn {resume_turn}: {resume_long_video_path}")
            start_turn = resume_turn + 1
            control_video = load_video(resume_long_video_path)
            first_image = control_video[-1]
        else:
            print("[Fresh] No existing long video, start from the input image.")
            start_turn = 0
            control_video = []
            first_image = first_image_pil

        for turn_idx in range(start_turn, len(all_c2ws)):
            if turn_idx == 0:
                controlnet_history_videos = None
                ref_image_tensor = None
                frame_ids_image = None
                c2ws = all_c2ws[turn_idx].unsqueeze(0)
                intrinsics = all_intrinsics[turn_idx].unsqueeze(0)
            else:
                c2ws = all_c2ws[turn_idx].unsqueeze(0)
                intrinsics = all_intrinsics[turn_idx].unsqueeze(0)
                current_frame_ids_image = all_frame_ids_image[turn_idx]
                frame_ids_image = (
                    current_frame_ids_image.unsqueeze(0).to("cuda") if current_frame_ids_image is not None else None
                )

                ref_image_mp4 = temp_path_video / f"turn_{turn_idx}_ref_image.mp4"
                if current_frame_ids_image is not None and ref_image_mp4.exists():
                    ref_image = load_video(str(ref_image_mp4))
                else:
                    ref_image = build_reference_frames(control_video, current_frame_ids_image)
                ref_image_tensor = (
                    pipe.video_processor.preprocess(ref_image).unsqueeze(0) if ref_image is not None else None
                )
                ref_image_mp4.unlink(missing_ok=True)

                controlnet_history_videos = pipe.video_processor.preprocess(control_video).unsqueeze(0)

            inpaint_video, inpaint_video_mask, _ = get_image_to_video_latent_fromPIL(
                first_image,
                None,
                video_length=args.video_sample_n_frames,
                sample_size=[args.height, args.width],
            )
            video_generate_frames = pipe(
                prompt=prompt,
                num_frames=args.video_sample_n_frames,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                generator=torch.Generator().manual_seed(seed),
                clip_image=first_image,
                video=inpaint_video,
                mask_video=inpaint_video_mask,
                intrinsics=intrinsics,
                c2ws=c2ws,
                frame_ids_image=frame_ids_image,
                cond_image=ref_image_tensor,
                history_video=controlnet_history_videos,
                shift=args.shift,
                output_type="pil",
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            ).videos[0]

            if turn_idx == 0:
                control_video = video_generate_frames
            else:
                control_video.extend(video_generate_frames[1:])

            export_to_video(
                control_video,
                os.path.join(generate_dir, f"long_video_turn_{turn_idx}.mp4"),
                fps=args.long_video_fps,
            )

            if (turn_idx + 1) < len(all_frame_ids_image):
                next_frame_ids_image = all_frame_ids_image[turn_idx + 1]
                ref_image = build_reference_frames(control_video, next_frame_ids_image)
                if ref_image is not None:
                    temp_video_path_turn = temp_path_video / f"turn_{turn_idx + 1}_ref_image.mp4"
                    export_to_video(ref_image, str(temp_video_path_turn), fps=args.gen_fps)

            first_image = video_generate_frames[-1]

        shutil.rmtree(temp_path_video, ignore_errors=True)
        print(f"finished: {sample_idx} ({clip_name})")

    shutil.rmtree(temp_path, ignore_errors=True)


if __name__ == "__main__":
    default_metadata_path = Path(__file__).resolve().parents[1] / "assets" / "prompt.json"

    parser = argparse.ArgumentParser(description="Open-source inference script for prompt.json + pose txt assets.")
    parser.add_argument("--metadata_path", type=str, default=str(default_metadata_path))
    parser.add_argument("--output_path", type=str, default="./output_open_assets")

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--history_controlnet_ckpt", type=str, required=True)
    parser.add_argument("--controlnet_ckpt_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)

    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--video_sample_n_frames", type=int, default=49)
    parser.add_argument("--gen_fps", type=int, default=8)
    parser.add_argument("--long_video_fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="镜头切换，出现人物，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的,静止不动的画面，杂乱的背景,场景突变",
    )

    args = parser.parse_args()
    generate_video(args, dtype=torch.bfloat16)
