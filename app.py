import gradio as gr
import spaces
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

def process_video(input_video_path, temp_dir="temp_dir"):
    """
    Crop a given MP4 video to a maximum duration of 10 seconds if it is longer than 10 seconds.
    Save the new video in the specified folder (default is temp_dir).
    
    Args:
        input_video_path (str): Path to the input video file.
        temp_dir (str): Directory where the processed video will be saved.
        
    Returns:
        str: Path to the cropped video file.
    """
    # Ensure the temp_dir exists
    os.makedirs(temp_dir, exist_ok=True)
    
    # Load the video
    video = VideoFileClip(input_video_path)
    
    # Determine the output path
    input_file_name = os.path.basename(input_video_path)
    output_video_path = os.path.join(temp_dir, f"cropped_{input_file_name}")
    
    # Crop the video to 10 seconds if necessary
    if video.duration > 10:
        video = video.subclip(0, 10)
    
    # Write the cropped video to the output path
    video.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    
    # Return the path to the cropped video
    return output_video_path

def process_audio(file_path, temp_dir):
    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    
    # Check and cut the audio if longer than 4 seconds
    max_duration = 8 * 1000  # 4 seconds in milliseconds
    if len(audio) > max_duration:
        audio = audio[:max_duration]
    
    # Save the processed audio in the temporary directory
    output_path = os.path.join(temp_dir, "trimmed_audio.wav")
    audio.export(output_path, format="wav")
    
    # Return the path to the trimmed file
    print(f"Processed audio saved at: {output_path}")
    return output_path

import argparse
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature


@spaces.GPU(duration=180)
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

    is_shared_ui = True if "fffiloni/LatentSync" in os.environ['SPACE_ID'] else False
    temp_dir = None
    if is_shared_ui:
        temp_dir = tempfile.mkdtemp()
        cropped_video_path = process_video(video_path)
        print(f"Cropped video saved to: {cropped_video_path}")
        video_path=cropped_video_path

        trimmed_audio_path = process_audio(audio_path, temp_dir)
        print(f"Processed file was stored temporarily at: {trimmed_audio_path}")
        audio_path=trimmed_audio_path

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

    # set xformers
    
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
    

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

    if is_shared_ui:
        # Clean up the temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Temporary directory {temp_dir} deleted.")

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
            <a href="https://huggingface.co/spaces/fffiloni/LatentSync?duplicate=true">
                <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/duplicate-this-space-sm.svg" alt="Duplicate this Space">
            </a>
            <a href="https://huggingface.co/fffiloni">
                <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-me-on-HF-sm-dark.svg" alt="Follow me on HF">
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

demo.queue().launch(show_api=True, show_error=True, ssr_mode=False, mcp_server=True)
