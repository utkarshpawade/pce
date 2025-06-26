import math
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors
import mediapy
import rembg
from einops import rearrange, repeat

from kiui.op import recenter
import tyro
import sys
sys.path.append(r'C:\V3D')

from importlib.machinery import SourceFileLoader
DeepFloydModule = SourceFileLoader(
    "nsfw_and_watermark_detection",
    r"C:\V3D\scripts\util\detection\nsfw_and_watermark_dectection.py"
).load_module()
DeepFloydDataFiltering = DeepFloydModule.DeepFloydDataFiltering
from importlib.machinery import SourceFileLoader
HelpersModule = SourceFileLoader(
    "sgm_inference_helpers",
    r"C:\V3D\sgm\inference\helpers.py"
).load_module()
embed_watermark = HelpersModule.embed_watermark
from importlib.machinery import SourceFileLoader
UtilModule = SourceFileLoader(
    "sgm_util",
    r"C:\V3D\sgm\util.py"
).load_module()
default = UtilModule.default
instantiate_from_config = UtilModule.instantiate_from_config
repo_path = r"C:\V3D"                      # Path to cloned V3D repository
input_image_path = r"C:\V3D\baby_yoda.png"     # Input image path
output_folder = r"C:\V3D\outputs"              # Output directory
checkpoint_path = r"C:\V3D\ckpts\V3D_512.ckpt" # Fine-tuned V3D weights
base_model_path = r"C:\V3D\ckpts\svd_xt.safetensors"  # Base SVD weights
model_config = r"C:\V3D\scripts\pub\configs\V3D_512.yaml"  # V3D config
clip_model_config = r"C:\V3D\configs\embedder\clip_image.yaml"  # CLIP config
ae_model_config = r"C:\V3D\configs\ae\video.yaml"  # Autoencoder config
num_frames = 8                                 # Number of multi-view frames
num_steps = 25                                 # Number of denoising steps
fps_id = 1                                     # Frames per second ID
motion_bucket_id = 300                         # Motion bucket for SVD
cond_aug = 0.02                                # Conditional augmentation noise
seed = 23                                      # Random seed
decoding_t = 8                                 # Frames decoded at a time (adjust for VRAM)
device = "cuda"                                # Device (cuda or cpu)
border_ratio = 0.3                             # Border ratio for recentering
min_guidance_scale = 3.5                       # Minimum CFG scale
max_guidance_scale = 3.5                       # Maximum CFG scale
sigma_max = None                               # Optional sigma max override
save_video = True                              # Save output as video
ignore_alpha = False                           # Ignore alpha channel if True

# Utility functions (from V3D repository)
# def get_unique_embedder_keys_from_conditioner(conditioner):
#     return list(set([x.input_key for x in conditioner.embedders]))

def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}
    for key in keys:
        if key == "fps_id":
            batch[key] = torch.tensor([value_dict["fps_id"]]).to(device).repeat(int(math.prod(N)))
        elif key == "motion_bucket_id":
            batch[key] = torch.tensor([value_dict["motion_bucket_id"]]).to(device).repeat(int(math.prod(N)))
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]
    if T is not None:
        batch["num_video_frames"] = T
    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    ckpt_path: str,
    min_cfg: float,
    max_cfg: float,
    sigma_max: float,
):
    config = OmegaConf.load(config)
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = num_frames
    config.model.params.sampler_config.params.guider_config.params.min_scale = min_cfg
    config.model.params.sampler_config.params.guider_config.params.max_scale = max_cfg
    if sigma_max is not None:
        config.model.params.sampler_config.params.discretization_config.params.sigma_max = sigma_max
    config.model.params.from_scratch = False
    config.model.params.ckpt_path = ckpt_path
    with torch.device(device):
        model = instantiate_from_config(config.model).to(device).eval()
    return model, None

# Main generation function
def generate_multiview_images(
    input_path: str,
    checkpoint_path: str,
    base_model_path: str,
    model_config: str,
    clip_model_config: str,
    ae_model_config: str,
    num_frames: int,
    num_steps: int,
    fps_id: int,
    motion_bucket_id: int,
    cond_aug: float,
    seed: int,
    decoding_t: int,
    device: str,
    output_folder: str,
    border_ratio: float,
    min_guidance_scale: float,
    max_guidance_scale: float,
    sigma_max: float,
    save_video: bool,
    ignore_alpha: bool
):

    os.makedirs(output_folder, exist_ok=True)

    # Load base model weights (svd_xt.safetensors)
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model weights not found at {base_model_path}")
    sd = load_safetensors(base_model_path)

    # Load CLIP model for conditioning
    if not os.path.exists(clip_model_config):
        raise FileNotFoundError(f"CLIP config file not found at {clip_model_config}")
    clip_model = instantiate_from_config(OmegaConf.load(clip_model_config)).eval()
    clip_sd = {k.replace("conditioner.embedders.0.", ""): v for k, v in sd.items() if "conditioner.embedders.0" in k}
    clip_model.load_state_dict(clip_sd)
    clip_model = clip_model.to(device)

    # Load autoencoder (AE) model
    if not os.path.exists(ae_model_config):
        raise FileNotFoundError(f"AE config file not found at {ae_model_config}")
    ae_model = instantiate_from_config(OmegaConf.load(ae_model_config)).eval()
    encoder_sd = {k.replace("first_stage_model.", ""): v for k, v in sd.items() if "first_stage_model" in k}
    ae_model.load_state_dict(encoder_sd)
    ae_model = ae_model.to(device)

    # Load fine-tuned V3D model
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Fine-tuned checkpoint not found at {checkpoint_path}")
    if not os.path.exists(model_config):
        raise FileNotFoundError(f"Model config file not found at {model_config}")
    model, _ = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        checkpoint_path,
        min_guidance_scale,
        max_guidance_scale,
        sigma_max
    )

    # Set random seed
    torch.manual_seed(seed)

    # Load and preprocess input image
    path = Path(input_path)
    if not path.is_file() or not path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        raise ValueError("Input must be a valid image file (.jpg, .jpeg, .png)")

    with Image.open(input_path) as image:
        w, h = image.size
        if border_ratio > 0:
            if image.mode != "RGBA" or ignore_alpha:
                image = image.convert("RGB")
                image = np.asarray(image)
                carved_image = rembg.remove(image)  # [H, W, 4]
            else:
                image = np.asarray(image)
                carved_image = image
            mask = carved_image[..., -1] > 0
            image = recenter(carved_image, mask, border_ratio=border_ratio)
            image = image.astype(np.float32) / 255.0
            if image.shape[-1] == 4:
                image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
            image = Image.fromarray((image * 255).astype(np.uint8))
        else:
            print("Ignoring border ratio")
        image = image.resize((512, 512))

        image = ToTensor()(image)
        image = image * 2.0 - 1.0  # Normalize to [-1, 1]

    image = image.unsqueeze(0).to(device)
    H, W = image.shape[2:]
    assert image.shape[1] == 3
    F = 8
    C = 4
    shape = (num_frames, C, H // F, W // F)

    # Prepare conditioning
    value_dict = {
        "motion_bucket_id": motion_bucket_id,
        "fps_id": fps_id,
        "cond_aug": cond_aug,
        "cond_frames_without_noise": clip_model(image),
        "cond_frames": ae_model.encode(image) + cond_aug * torch.randn_like(ae_model.encode(image))
    }

    with torch.no_grad():
        with torch.autocast(device):
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                [1, num_frames],
                T=num_frames,
                device=device,
            )
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"],
            )

            for k in ["crossattn", "concat"]:
                uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

            # Generate samples
            randn = torch.randn(shape, device=device)
            additional_model_inputs = {
                "image_only_indicator": torch.zeros(2, num_frames).to(device),
                "num_video_frames": batch["num_video_frames"]
            }

            def denoiser(input, sigma, c):
                return model.denoiser(model.model, input, sigma, c, **additional_model_inputs)

            samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
            model.en_and_decode_n_samples_a_time = decoding_t
            samples_x = model.decode_first_stage(samples_z)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            # Convert samples to images
            frames = (rearrange(samples, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8)
            images = [Image.fromarray(frame) for frame in frames]

            # Save individual frames
            for i, img in enumerate(images):
                output_path = os.path.join(output_folder, f"view_{i:03d}.png")
                img.save(output_path)
                print(f"Saved view {i} to {output_path}")

            # Save video if requested
            if save_video:
                # Use a unique video filename based on the input image name
                base_name = os.path.splitext(os.path.basename(input_path))[0]
                video_path = os.path.join(output_folder, f"{base_name}.mp4")
                mediapy.write_video(video_path, frames, fps=fps_id + 1)
                print(f"Saved video to {video_path}")

    return images

if __name__ == "__main__":
    images = generate_multiview_images(
        input_path=input_image_path,
        checkpoint_path=checkpoint_path,
        base_model_path=base_model_path,
        model_config=model_config,
        clip_model_config=clip_model_config,
        ae_model_config=ae_model_config,
        num_frames=num_frames,
        num_steps=num_steps,
        fps_id=fps_id,
        motion_bucket_id=motion_bucket_id,
        cond_aug=cond_aug,
        seed=seed,
        decoding_t=decoding_t,
        device=device,
        output_folder=output_folder,
        border_ratio=border_ratio,
        min_guidance_scale=min_guidance_scale,
        max_guidance_scale=max_guidance_scale,
        sigma_max=sigma_max,
        save_video=save_video,
        ignore_alpha=ignore_alpha
    )
    print(f"Generated {len(images)} multi-view images in {output_folder}")
