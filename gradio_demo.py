import gradio as gr
import subprocess
import numpy as np

# Function to run text2world.py
def generate_text2world(
    prompt, model_size, offload_options, seed, negative_prompt, num_steps, guidance, num_video_frames, height, width, fps, disable_prompt_upsampler
):
    offload_prompt_upsampler = 'Offload Prompt Upsampler' in offload_options
    offload_guardrail_models = 'Offload Guardrail Models' in offload_options
    offload_tokenizer = 'Offload Tokenizer' in offload_options
    offload_diffusion_transformer = 'Offload Diffusion Transformer' in offload_options
    offload_text_encoder_model = 'Offload Text Encoder Model' in offload_options

    args = [
        'PYTHONPATH=$(pwd) python cosmos1/models/diffusion/inference/text2world.py',
        '--checkpoint_dir checkpoints',
        f'--diffusion_transformer_dir Cosmos-1.0-Diffusion-{model_size}-Text2World',
        f'--prompt "{prompt}"',
        '--video_save_name output_text2world',
        f'--seed {seed}',
        f'--negative_prompt "{negative_prompt}"',
        f'--num_steps {num_steps}',
        f'--guidance {guidance}',
        f'--num_video_frames {num_video_frames}',
        f'--height {height}',
        f'--width {width}',
        f'--fps {fps}'
    ]
    
    if disable_prompt_upsampler:
        args.append('--disable_prompt_upsampler')
        if prompt:
            args.extend([f'--prompt "{prompt}"'])
        else:
            raise ValueError("Prompt is required when prompt upsampler is disabled.")
    
    if offload_prompt_upsampler:
        args.append('--offload_prompt_upsampler')
    if offload_guardrail_models:
        args.append('--offload_guardrail_models')
    if offload_tokenizer:
        args.append('--offload_tokenizer')
    if offload_diffusion_transformer:
        args.append('--offload_diffusion_transformer')
    if offload_text_encoder_model:
        args.append('--offload_text_encoder_model')
    
    command = ' '.join(args)
    subprocess.run(command, shell=True)
    
    video_path = 'outputs/output_text2world.mp4'
    return video_path

# Function to run video2world.py
def generate_video2world(
    input_file, model_size, num_input_frames, prompt, disable_prompt_upsampler, offload_options, seed,
    negative_prompt, num_steps, guidance, num_video_frames, height, width, fps
):
    offload_prompt_upsampler = 'Offload Prompt Upsampler' in offload_options
    offload_guardrail_models = 'Offload Guardrail Models' in offload_options
    offload_tokenizer = 'Offload Tokenizer' in offload_options
    offload_diffusion_transformer = 'Offload Diffusion Transformer' in offload_options
    offload_text_encoder_model = 'Offload Text Encoder Model' in offload_options

    args = [
        'PYTHONPATH=$(pwd) python cosmos1/models/diffusion/inference/video2world.py',
        '--checkpoint_dir checkpoints',
        f'--diffusion_transformer_dir Cosmos-1.0-Diffusion-{model_size}-Video2World',
        f'--input_image_or_video_path {input_file}',
        '--video_save_name output_video2world',
        f'--seed {seed}',
        f'--num_input_frames {num_input_frames}',
        f'--negative_prompt "{negative_prompt}"',
        f'--num_steps {num_steps}',
        f'--guidance {guidance}',
        f'--num_video_frames {num_video_frames}',
        f'--height {height}',
        f'--width {width}',
        f'--fps {fps}'
    ]
    
    if disable_prompt_upsampler:
        args.append('--disable_prompt_upsampler')
        if prompt:
            args.extend([f'--prompt "{prompt}"'])
        else:
            raise ValueError("Prompt is required when prompt upsampler is disabled.")
    
    if offload_prompt_upsampler:
        args.append('--offload_prompt_upsampler')
    if offload_guardrail_models:
        args.append('--offload_guardrail_models')
    if offload_tokenizer:
        args.append('--offload_tokenizer')
    if offload_diffusion_transformer:
        args.append('--offload_diffusion_transformer')
    if offload_text_encoder_model:
        args.append('--offload_text_encoder_model')
    
    command = ' '.join(args)
    subprocess.run(command, shell=True)
    
    video_path = 'outputs/output_video2world.mp4'
    return video_path

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Cosmos Diffusion-based World Foundation Models Demo")
    
    with gr.Tab("Text2World"):
        text_prompt = gr.Textbox(label="Text Prompt", lines=5)
        model_size_text = gr.Radio(["7B", "14B"], label="Model Size", value="7B")
        offload_options_text = gr.CheckboxGroup(["Offload Prompt Upsampler", "Offload Guardrail Models", "Offload Tokenizer", "Offload Diffusion Transformer", "Offload Text Encoder Model"], label="Offload Options")
        seed_text = gr.Number(label="Seed", value=1)
        disable_prompt_upsampler_text = gr.Checkbox(label="Disable Prompt Upsampler")
        negative_prompt_text = gr.Textbox(label="Negative Prompt", value="""The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality.""", lines=5)
        num_steps_text = gr.Number(label="Number of Steps", value=35)
        guidance_text = gr.Number(label="Guidance Scale", value=7)
        num_video_frames_text = gr.Number(label="Number of Video Frames", value=121, info="Must be divisible by 121")
        height_text = gr.Number(label="Height", value=704)
        width_text = gr.Number(label="Width", value=1280)
        fps_text = gr.Number(label="FPS", value=24)
        generate_button_text = gr.Button("Generate Video")
        output_video_text = gr.Video(label="Generated Video")
        
        generate_button_text.click(
            generate_text2world,
            inputs=[
                text_prompt, model_size_text, offload_options_text, seed_text,
                negative_prompt_text, num_steps_text, guidance_text,
                num_video_frames_text, height_text, width_text, fps_text,
                disable_prompt_upsampler_text
            ],
            outputs=output_video_text
        )
    
    with gr.Tab("Video2World"):
        input_file = gr.File(label="Input Image/Video")
        model_size_video = gr.Radio(["7B", "14B"], label="Model Size", value="7B")
        num_input_frames = gr.Slider(1, 9, step=1, label="Number of Input Frames", value=1)
        text_prompt_video = gr.Textbox(label="Text Prompt (Optional)", lines=5)
        disable_prompt_upsampler_video = gr.Checkbox(label="Disable Prompt Upsampler")
        offload_options_video = gr.CheckboxGroup(["Offload Prompt Upsampler", "Offload Guardrail Models", "Offload Tokenizer", "Offload Diffusion Transformer", "Offload Text Encoder Model"], label="Offload Options")
        seed_video = gr.Number(label="Seed", value=1)
        negative_prompt_video = gr.Textbox(label="Negative Prompt", value="""The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality.""", lines=5)
        num_steps_video = gr.Number(label="Number of Steps", value=35)
        guidance_video = gr.Number(label="Guidance Scale", value=7)
        num_video_frames_video = gr.Number(label="Number of Video Frames", value=121, info="Must be divisible by 121")
        height_video = gr.Number(label="Height", value=704)
        width_video = gr.Number(label="Width", value=1280)
        fps_video = gr.Number(label="FPS", value=24)
        generate_button_video = gr.Button("Generate Video")
        output_video_video = gr.Video(label="Generated Video")
        
        generate_button_video.click(
            generate_video2world,
            inputs=[
                input_file, model_size_video, num_input_frames, text_prompt_video,
                disable_prompt_upsampler_video, offload_options_video, seed_video,
                negative_prompt_video, num_steps_video, guidance_video,
                num_video_frames_video, height_video, width_video, fps_video
            ],
            outputs=output_video_video
        )

demo.launch(share=True)