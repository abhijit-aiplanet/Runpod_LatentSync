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

def process_video(input_video_path, temp_dir="temp_dir"):
    """
    Trim video to max 10 seconds using ffmpeg without re-encoding.
    This avoids the 'crf' option error in minimal ffmpeg builds.
    """
    os.makedirs(temp_dir, exist_ok=True)

    # Determine output path
    input_file_name = os.path.basename(input_video_path)
    output_video_path = os.path.join(temp_dir, f"processed_{input_file_name}")

    # Build ffmpeg command
    cmd = [
        "ffmpeg", "-y", "-i", input_video_path,
        "-t", "10",            # max 10 seconds
        "-c:v", "copy",         # no re-encode (avoids crf)
        "-c:a", "copy",
        output_video_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("ffmpeg error while processing video:\n", e.stderr.decode())
        raise RuntimeError("Video preprocessing failed")

    if not os.path.exists(output_video_path):
        raise RuntimeError("Processed video not created")

    return output_video_path

def process_audio(file_path, temp_dir):
    """
    Process audio to ensure compatibility and reasonable duration.
    """
    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    
    # Check and cut the audio if longer than 8 seconds
    max_duration = 8 * 1000  # 8 seconds in milliseconds
    if len(audio) > max_duration:
        audio = audio[:max_duration]
    
    # Save the processed audio in the temporary directory
    output_path = os.path.join(temp_dir, "processed_audio.wav")
    audio.export(output_path, format="wav")
    
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
import latentsync.utils.util as ls_util
import latentsync.pipelines.lipsync_pipeline as ls_pipe
import latentsync.whisper.whisper.audio as wa
import numpy as np

orig_read_video = ls_util.read_video

def _read_video_no_change(video_path: str, change_fps=True, use_decord=True):
    return orig_read_video(video_path, change_fps=False, use_decord=True)

ls_util.read_video = _read_video_no_change
ls_pipe.read_video = _read_video_no_change

# Patch mel_filters to allow_pickle
_orig_mel_filters = wa.mel_filters

def _patched_mel_filters(device, n_mels=80):
    assert n_mels == 80
    with np.load(os.path.join(os.path.dirname(wa.__file__), "assets", "mel_filters.npz"), allow_pickle=True) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

wa.mel_filters.cache_clear()
wa.mel_filters = _patched_mel_filters

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
    # Create temporary directory for processed files
    temp_dir = tempfile.mkdtemp()
    
    try:
        print(f"Original video path: {video_path}")
        print(f"Original audio path: {audio_path}")
        
        # Process video and audio for better compatibility
        processed_video_path = process_video(video_path, temp_dir)
        processed_audio_path = process_audio(audio_path, temp_dir)
        
        print(f"Processed video path: {processed_video_path}")
        print(f"Processed audio path: {processed_audio_path}")
        
        # Use processed files
        video_path = processed_video_path
        audio_path = processed_audio_path
        
        inference_ckpt_path = "checkpoints/latentsync_unet.pt"
        unet_config_path = "configs/unet/second_stage.yaml"
        config = OmegaConf.load(unet_config_path)
        
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
            # Patch for custom UNet model signature mismatch
            patched_modules = []
            for name, module in unet.named_modules():
                if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                    # Store original method for this specific instance
                    original_method = module.set_use_memory_efficient_attention_xformers
                    
                    # Create a wrapper that handles extra arguments for this instance
                    def create_patched_method(orig_method):
                        def patched_method(valid, attention_op=None):
                            return orig_method(valid)
                        return patched_method
                    
                    # Replace the method on this specific instance
                    module.set_use_memory_efficient_attention_xformers = create_patched_method(original_method)
                    patched_modules.append(name)
            
            try:
                unet.enable_xformers_memory_efficient_attention()
                print(f"✅ xformers memory efficient attention enabled with custom patch on {len(patched_modules)} modules")
            except Exception as e:
                print(f"Failed to enable xformers: {e}")
                print("Continuing without xformers - performance may be slower")
        else:
            print("xformers not available - install with: pip install xformers")

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
    
    except Exception as e:
        print(f"Error during processing: {e}")
        raise e
    
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")


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
