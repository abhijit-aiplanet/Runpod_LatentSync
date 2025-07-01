import gradio as gr
import os
import sys
import shutil
import uuid
import subprocess
from glob import glob
from huggingface_hub import snapshot_download

# Download models
os.makedirs("checkpoints", exist_ok=True)

snapshot_download(
    repo_id = "chunyu-li/LatentSync",
    local_dir = "./checkpoints"  
)

import tempfile
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

import argparse
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature


def main(video_path, audio_path, progress=gr.Progress(track_tqdm=True)):
    """
    Perform lip-sync video generation using an input video and a separate audio track.
    
    This function takes an input video (usually a person speaking) and an audio file,
    and synchronizes the video frames so that the lips of the speaker match the audio content.
    It uses a latent diffusion model-based pipeline (LatentSync) for audio-conditioned lip synchronization.
    
    Args:
        video_path (str): File path to the input video in MP4 format.
        audio_path (str): File path to the input audio file (e.g., WAV or MP3).
        progress (gr.Progress, optional): Gradio progress tracker for UI feedback (auto-injected).
        
    Returns:
        str: File path to the generated output video with lip synchronization applied.
    """
    
    inference_ckpt_path = "checkpoints/latentsync_unet.pt"
    unet_config_path = "configs/unet/second_stage.yaml"
    config = OmegaConf.load(unet_config_path)
    
    print(f"Input video path: {video_path}")
    print(f"Input audio path: {audio_path}")
    print(f"Loaded checkpoint path: {inference_ckpt_path}")

    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        inference_ckpt_path,  # load checkpoint
        device="cpu",
    )

    unet = unet.to(dtype=torch.float16)

    # Enable xformers for better performance on A100
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
        print("xformers memory efficient attention enabled")

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    seed = -1
    if seed != -1:
        set_seed(seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")

    unique_id = str(uuid.uuid4())
    video_out_path = f"video_out{unique_id}.mp4"

    pipeline(
        video_path=video_path,
        audio_path=audio_path,
        video_out_path=video_out_path,
        video_mask_path=video_out_path.replace(".mp4", "_mask.mp4"),
        num_frames=config.data.num_frames,
        num_inference_steps=config.run.inference_steps,
        guidance_scale=1.0,
        weight_dtype=torch.float16,
        width=config.data.resolution,
        height=config.data.resolution,
    )

    return video_out_path


css="""
div#col-container{
    margin: 0 auto;
    max-width: 982px;
}
"""
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# LatentSync: Audio Conditioned Latent Diffusion Models for Lip Sync")
        gr.Markdown("LatentSync, an end-to-end lip sync framework based on audio conditioned latent diffusion models without any intermediate motion representation, diverging from previous diffusion-based lip sync methods based on pixel space diffusion or two-stage generation.")
        gr.HTML("""
        <div style="display:flex;column-gap:4px;">
            <a href="https://github.com/bytedance/LatentSync">
                <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
            </a> 
            <a href="https://arxiv.org/abs/2412.09262">
                <img src='https://img.shields.io/badge/ArXiv-Paper-red'>
            </a>
        </div>
        """)
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Video Control", format="mp4")
                audio_input = gr.Audio(label="Audio Input", type="filepath")
                submit_btn = gr.Button("Submit")
            with gr.Column():
                video_result = gr.Video(label="Result")

                gr.Examples(
                    examples = [
                        ["assets/demo1_video.mp4", "assets/demo1_audio.wav"],
                        ["assets/demo2_video.mp4", "assets/demo2_audio.wav"],
                        ["assets/demo3_video.mp4", "assets/demo3_audio.wav"],
                    ],
                    inputs = [video_input, audio_input]
                )

    submit_btn.click(
        fn = main,
        inputs = [video_input, audio_input],
        outputs = [video_result]
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=True, 
        show_error=True
    )
