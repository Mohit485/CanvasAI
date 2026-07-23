import spaces
import torch
import gradio as gr
import numpy as np
import os
import urllib
import sys

from PIL import Image, ImageFilter, ImageDraw
from diffusers import StableDiffusionInpaintPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration

import torchvision.transforms.functional as F
sys.modules["torchvision.transforms.functional_tensor"] = F

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# LOAD MODELS AT STARTUP
# ── Real-ESRGAN ──────────────────────────────────────────────
print("Downloading Real-ESRGAN weights...")

weights_dir = "weights"
os.makedirs(weights_dir, exist_ok=True)
weights_path = f"{weights_dir}/RealESRGAN_x4plus.pth"
if not os.path.exists(weights_path):
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    urllib.request.urlretrieve(url, weights_path)
    print("Weights downloaded.")

esrgan_model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)

enhancer = RealESRGANer(
    scale=4,
    model_path=weights_path,
    model=esrgan_model,
    tile=512,
    tile_pad=10,
    pre_pad=0,
    half=True,
)
print("Real-ESRGAN ready.")

# ── SD Inpainting ────────────────────────────────────────────
print("Loading SD Inpainting...")

inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
)
print("SD Inpainting ready.")


#  BLIP 
print("Loading BLIP...")

blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    torch_dtype=torch.float16,
)
# Same — no .to("cuda") at load time for ZeroGPU
print("All models loaded.")


# HELPER FUNCTIONS
def is_greyscale(image):
    rgb = image.convert("RGB")
    r, g, b = rgb.split()
    r_arr = np.array(r, dtype=float)
    g_arr = np.array(g, dtype=float)
    b_arr = np.array(b, dtype=float)
    diff_rg = np.mean(np.abs(r_arr - g_arr))
    diff_rb = np.mean(np.abs(r_arr - b_arr))
    return (diff_rg < 10 and diff_rb < 10)

def get_caption(image):
    inputs = blip_processor(
        image.convert("RGB"),
        return_tensors="pt"
    ).to("cuda", torch.float16)
    # Inside @spaces.GPU function, cuda is available
    output = blip_model.generate(**inputs, max_new_tokens=60)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption

def build_prompt(caption, image):
    bw = is_greyscale(image)
    style_hint = (
        "black and white photography, monochrome, greyscale, "
        "vintage photograph, same tonal range"
        if bw else
        "same color palette, same lighting conditions, photorealistic"
    )
    return (
        f"seamless natural continuation of scene, {caption}, "
        f"{style_hint}, extending background only, "
        f"same atmosphere, high quality"
    )

def extend_one_side(image, direction, pixels, prompt, negative_prompt):
    orig_w = image.width
    orig_h = image.height

    new_w   = orig_w
    new_h   = orig_h
    paste_x = 0
    paste_y = 0

    if direction == "left":
        new_w   = orig_w + pixels
        paste_x = pixels
    elif direction == "right":
        new_w   = orig_w + pixels
        paste_x = 0
    elif direction == "top":
        new_h   = orig_h + pixels
        paste_y = pixels
    elif direction == "bottom":
        new_h   = orig_h + pixels
        paste_y = 0

    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8

    if direction in ["left", "right"]:
        pixels = new_w - orig_w
    else:
        pixels = new_h - orig_h

    if direction == "left":
        paste_x = pixels
    if direction == "top":
        paste_y = pixels

    canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    canvas.paste(image, (paste_x, paste_y))

    mask = Image.new("L", (new_w, new_h), 255)
    draw = ImageDraw.Draw(mask)
    feather = 30

    if direction == "left":
        draw.rectangle([paste_x + feather, feather, new_w - feather, new_h - feather], fill=0)
    elif direction == "right":
        draw.rectangle([feather, feather, orig_w - feather, new_h - feather], fill=0)
    elif direction == "top":
        draw.rectangle([feather, paste_y + feather, new_w - feather, new_h - feather], fill=0)
    elif direction == "bottom":
        draw.rectangle([feather, feather, new_w - feather, orig_h - feather], fill=0)

    mask = mask.filter(ImageFilter.GaussianBlur(radius=30))

    sd_size   = 512
    canvas_sd = canvas.resize((sd_size, sd_size), Image.LANCZOS)
    mask_sd   = mask.resize((sd_size, sd_size), Image.LANCZOS)

    result = inpaint(
        prompt              = prompt,
        image               = canvas_sd,
        mask_image          = mask_sd,
        height              = sd_size,
        width               = sd_size,
        num_inference_steps = 40,
        guidance_scale      = 7.0,
        negative_prompt     = negative_prompt,
    )

    generated_512  = result.images[0]
    generated_full = generated_512.resize((new_w, new_h), Image.LANCZOS)
    # No hard paste — GaussianBlur mask handles the boundary softly
    return generated_full

@spaces.GPU
def enhance_image(image, scale_factor):
    if image is None:
        raise gr.Error("Please upload an image first.")

    # Move models to GPU — happens inside the decorated function
    # because GPU is only available here
    enhancer.device = torch.device("cuda")
    enhancer.half   = True

    image_array = np.array(image)
    image_array = image_array[:, :, :3]
    outscale    = 4 if scale_factor == "4x" else 2

    try:
        output_array, _ = enhancer.enhance(image_array, outscale=outscale)
    except RuntimeError as e:
        raise gr.Error(f"Enhancement failed: {e}. Try a smaller image.")

    output_rgb   = output_array[:, :, ::-1]
    output_image = Image.fromarray(output_rgb)

    original_size = f"{image.width}×{image.height}"
    new_size      = f"{output_image.width}×{output_image.height}"

    return output_image, f"Original: {original_size} → Enhanced: {new_size}"

@spaces.GPU
def outpaint_image(image, direction, extend_percent, custom_prompt, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Please upload an image first.")

    # Move models to GPU inside the decorated function
    inpaint.to("cuda")
    blip_model.to("cuda")

    target_side = 512
    ratio       = min(target_side / image.width, target_side / image.height)
    new_size    = (int(image.width * ratio), int(image.height * ratio))
    image       = image.resize(new_size, Image.LANCZOS)

    progress(0.05, desc="Analyzing image with BLIP...")

    blip_caption = custom_prompt.strip() if custom_prompt.strip() else get_caption(image)
    prompt       = build_prompt(blip_caption, image)

    negative_prompt = (
        "blurry, bad quality, watermark, text, "
        "new person, new face, extra people, "
        "colorful, vibrant colors, color photography, "
        "duplicate, border, frame, seam, visible edge, "
        "distorted, inconsistent style"
    )

    STEP_PX = 64
    extend  = extend_percent / 100.0
    h_total = int(image.width  * extend)
    v_total = int(image.height * extend)

    def make_passes(side, total_px):
        passes    = []
        remaining = total_px
        while remaining > 0:
            step = min(STEP_PX, remaining)
            passes.append((side, step))
            remaining -= step
        return passes

    if direction == "Horizontal":
        passes = make_passes("right", h_total) + make_passes("left", h_total)
    elif direction == "Vertical":
        passes = make_passes("bottom", v_total) + make_passes("top", v_total)
    else:
        passes = (
            make_passes("bottom", v_total) +
            make_passes("top",    v_total) +
            make_passes("right",  h_total) +
            make_passes("left",   h_total)
        )

    total_passes  = len(passes)
    current_image = image

    for i, (side, px) in enumerate(passes):
        progress(
            0.1 + 0.85 * (i / total_passes),
            desc=f"Pass {i+1}/{total_passes} — extending {side} by {px}px"
        )
        current_image = extend_one_side(
            current_image, side, px, prompt, negative_prompt
        )

    progress(1.0, desc="Done!")

    bw_note = " [B&W detected]" if is_greyscale(image) else ""
    return (
        current_image,
        f"Caption{bw_note}:\n{blip_caption}\n\nPrompt:\n{prompt}"
    )

# GRADIO UI
with gr.Blocks(title="CanvasAI — Enhance & Outpaint") as demo:

    gr.Markdown("#  CanvasAI")
    gr.Markdown(
        "**Enhance** any image with Real-ESRGAN super resolution, "
        "or **Outpaint** to extend the scene in any direction using AI."
    )

    with gr.Tabs():

        with gr.Tab(" Enhance"):
            gr.Markdown("Upscale and sharpen any image 2x or 4x using Real-ESRGAN.")
            with gr.Row():
                with gr.Column():
                    enh_input = gr.Image(label="Upload Image", type="pil")
                    enh_scale = gr.Dropdown(
                        choices=["2x", "4x"],
                        value="4x",
                        label="Upscale Factor"
                    )
                    enh_btn = gr.Button("Enhance", variant="primary")
                with gr.Column():
                    enh_output = gr.Image(label="Result", type="pil", interactive=False)
                    enh_info   = gr.Textbox(label="Size Info", interactive=False)

            enh_btn.click(
                fn=enhance_image,
                inputs=[enh_input, enh_scale],
                outputs=[enh_output, enh_info]
            )

        with gr.Tab(" Outpaint"):
            gr.Markdown(
                "Upload an image and extend it in any direction. "
                "BLIP reads the scene automatically — no prompt needed."
            )
            with gr.Row():
                with gr.Column():
                    out_input = gr.Image(label="Upload Image", type="pil")
                    out_dir   = gr.Radio(
                        choices=["Horizontal", "Vertical", "Both"],
                        value="Horizontal",
                        label="Direction"
                    )
                    out_pct = gr.Slider(
                        minimum=10, maximum=50,
                        value=25, step=5,
                        label="Extend by (%)"
                    )
                    out_prompt = gr.Textbox(
                        label="Custom Prompt (optional)",
                        placeholder="Leave empty for auto-detection",
                        lines=2
                    )
                    out_btn = gr.Button("🔲 Outpaint", variant="primary")
                with gr.Column():
                    out_output  = gr.Image(label="Result", type="pil", interactive=False)
                    out_caption = gr.Textbox(label="Prompt Used", interactive=False, lines=4)

            out_btn.click(
                fn=outpaint_image,
                inputs=[out_input, out_dir, out_pct, out_prompt],
                outputs=[out_output, out_caption]
            )

#launch
demo.launch()