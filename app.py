import spaces
import torch
import gradio as gr
import numpy as np
import os
import urllib
import sys
import cv2 

from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
from diffusers import StableDiffusionInpaintPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

import torchvision.transforms.functional as F
sys.modules["torchvision.transforms.functional_tensor"] = F

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


# ================================================================
# LOAD REAL-ESRGAN
# ================================================================
print("Setting up Real-ESRGAN...")

weights_dir  = "weights"
os.makedirs(weights_dir, exist_ok=True)
weights_path = f"{weights_dir}/RealESRGAN_x4plus.pth"

if not os.path.exists(weights_path):
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    urllib.request.urlretrieve(url, weights_path)
    print("Weights downloaded.")

esrgan_model = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64,
    num_block=23, num_grow_ch=32, scale=4
)

enhancer = RealESRGANer(
    scale=4, model_path=weights_path, model=esrgan_model,
    tile=512, tile_pad=10, pre_pad=0, half=True,
)
enhancer.device = torch.device("cuda")
print("Real-ESRGAN ready.")


# ================================================================
# LOAD SD2 INPAINTING
# ================================================================
print("Loading SD2 Inpainting...")

inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "sd2-community/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
).to("cuda")

print("SD2 Inpainting ready.")


# ================================================================
# LOAD DDCOLOR
# ================================================================
print("Loading DDColor...")

colorizer = pipeline(
    Tasks.image_colorization,
    model="damo/cv_ddcolor_image-colorization"
)
print("DDColor ready.")


# ================================================================
# LOAD BLIP
# ================================================================
print("Loading BLIP...")

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    torch_dtype=torch.float16,
).to("cuda")
print("All models loaded.")


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def is_greyscale(image):
    # Returns True if image is effectively black and white
    rgb     = image.convert("RGB")
    r, g, b = rgb.split()
    r_arr   = np.array(r, dtype=float)
    g_arr   = np.array(g, dtype=float)
    b_arr   = np.array(b, dtype=float)
    diff_rg = np.mean(np.abs(r_arr - g_arr))
    diff_rb = np.mean(np.abs(r_arr - b_arr))
    return diff_rg < 10 and diff_rb < 10


def get_caption(image):
    inputs  = blip_processor(
        image.convert("RGB"), return_tensors="pt"
    ).to("cuda", torch.float16)
    output  = blip_model.generate(**inputs, max_new_tokens=60)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption


def build_outpaint_prompt(caption, image):
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
        f"same atmosphere, high quality, no new subjects"
    )


# ================================================================
# ENHANCEMENT
# ================================================================

@spaces.GPU
def enhance_image(image, scale_factor): #deblur
    if image is None:
        raise gr.Error("Please upload an image first.")
    #Enhance

    image_array = np.array(image)
    image_array = image_array[:, :, :3]
    # Keep only RGB — drop alpha channel if RGBA (transparent PNG)

    outscale = 4 if scale_factor == "4x" else 2

    try:
        output_array, _ = enhancer.enhance(image_array, outscale=outscale)
    except RuntimeError as e:
        raise gr.Error(f"Enhancement failed: {e}. Try a smaller image.")

    output_image = Image.fromarray(output_array)

    # Real-ESRGAN returns RGB — no channel flip needed
    return (
        output_image,
        f"Original: {image.width}x{image.height} -> "
        f"Enhanced: {output_image.width}x{output_image.height}"
    )



# ================================================================
# COLORIZATION
# ================================================================

@spaces.GPU
def colour_image(image, strength):
    if image is None:
        raise gr.Error("Please upload an image first.")


    original_size = (image.width, image.height)
    image = image.convert("RGB")
    

    target = 512
    ratio  = min(target / image.width, target / image.height)
    w = int(image.width  * ratio)
    h = int(image.height * ratio)
    image  = image.resize((w, h), Image.LANCZOS)
    #DDCOLOR
    # Convert PIL RGB -> numpy BGR for DDColor
    img_rgb = np.array(image)
    img_bgr = img_rgb[:, :, ::-1]
    # [:,:,::-1] reverses channel order: RGB -> BGR

    result     = colorizer(img_bgr)
    output_bgr = result["output_img"]
    # output_bgr = colorized BGR numpy array, same size as input
    output_rgb = output_bgr[:, :, ::-1]
    # Reverse back: BGR -> RGB for PIL

    # Lab Color Smoothing:LAB color space separates:
    #   L = Lightness (brightness) — we don't touch this
    #   A = green-red color axis
    #   B = blue-yellow color axis

    lab = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    # LAB values: L=0-100, A=-128 to 127, B=-128 to 127
    # After cv2 conversion with uint8: L=0-255, A=0-255, B=0-255
    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    a_smooth = cv2.GaussianBlur(a_channel, (31, 31), sigmaX=0)
    b_smooth = cv2.GaussianBlur(b_channel, (31, 31), sigmaX=0)

    # Merge smoothed color channels back with original lightness
    lab_smooth = np.stack([l_channel, a_smooth, b_smooth], axis=-1)
    lab_smooth = lab_smooth.astype(np.uint8)
    output_rgb = cv2.cvtColor(lab_smooth, cv2.COLOR_LAB2RGB)
    # Convert back LAB → RGB

    # Saturation Boost
    pil_result = Image.fromarray(output_rgb)
    enhancer_sat = ImageEnhance.Color(pil_result)
    pil_result = enhancer_sat.enhance(1.4)
    output_rgb = np.array(pil_result)

    #strength
    if strength < 1.0:
        original_rgb = np.array(image)
        grey     = np.array(image.convert("L"))
        grey_3ch = np.stack([grey, grey, grey], axis=-1)
        # Stack single greyscale channel 3x to match (h,w,3) shape
        output_rgb = (
            strength       * output_rgb.astype(float) +
            (1 - strength) * grey_3ch.astype(float)
        ).astype(np.uint8)

    final = Image.fromarray(output_rgb)
    final = final.resize(original_size, Image.LANCZOS)
    return final


# ================================================================
# OUTPAINTING CORE
# ================================================================

def extend_one_side(image, direction, pixels, prompt, negative_prompt):
    # Extends image by 'pixels' in 'direction'
    # Called in 64px steps — small steps give SD max context

    orig_w = image.width
    orig_h = image.height
    new_w  = orig_w
    new_h  = orig_h
    paste_x = 0
    paste_y = 0

    if direction == "left":
        new_w   = orig_w + pixels
        paste_x = pixels
    elif direction == "right":
        new_w   = orig_w + pixels
    elif direction == "top":
        new_h   = orig_h + pixels
        paste_y = pixels
    elif direction == "bottom":
        new_h   = orig_h + pixels

    # Round to multiple of 8 — SD UNet requirement
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8

    # Recalculate pixels and paste positions after rounding
    if direction in ["left", "right"]:
        pixels = new_w - orig_w
    else:
        pixels = new_h - orig_h

    if direction == "left":
        paste_x = pixels
    if direction == "top":
        paste_y = pixels

    # Black canvas with original pasted at correct position
    canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    canvas.paste(image, (paste_x, paste_y))

    # Mask: white=generate, black=keep original
    mask = Image.new("L", (new_w, new_h), 255)
    draw = ImageDraw.Draw(mask)
    # BUG 6 FIX: removed redundant "from PIL import ImageDraw" inside function
    # Already imported at top of file

    feather = 30
    # 30px margin gives GaussianBlur room to create soft gradient

    if direction == "left":
        draw.rectangle([paste_x + feather, feather, new_w - feather, new_h - feather], fill=0)
    elif direction == "right":
        draw.rectangle([feather, feather, orig_w - feather, new_h - feather], fill=0)
    elif direction == "top":
        draw.rectangle([feather, paste_y + feather, new_w - feather, new_h - feather], fill=0)
    elif direction == "bottom":
        draw.rectangle([feather, feather, new_w - feather, orig_h - feather], fill=0)

    # GaussianBlur creates real feathering — soft gradient at boundary
    # Hard edge = visible seam. Soft gradient = seamless blend.
    mask = mask.filter(ImageFilter.GaussianBlur(radius=30))

    # SD2 native resolution = 768x768
    sd_size   = 768
    canvas_sd = canvas.resize((sd_size, sd_size), Image.LANCZOS)
    mask_sd   = mask.resize((sd_size, sd_size), Image.LANCZOS)
    # LANCZOS preserves soft gradient (NEAREST would destroy it)

    result = inpaint(
        prompt=prompt, image=canvas_sd, mask_image=mask_sd,
        height=sd_size, width=sd_size,
        num_inference_steps=40, guidance_scale=7.0,
        negative_prompt=negative_prompt,
    )

    generated_full = result.images[0].resize((new_w, new_h), Image.LANCZOS)
    # No hard paste — feathered mask handles boundary softly
    return generated_full


# ================================================================
# OUTPAINTING MAIN
# ================================================================

@spaces.GPU
def outpaint_image(image, direction, extend_percent, custom_prompt, progress=gr.Progress()):

    if image is None:
        raise gr.Error("Please upload an image first.")
    
    custom_prompt = custom_prompt or ""

    max_side = 512
    ratio    = min(max_side / image.width, max_side / image.height)
    image    = image.resize(
        (int(image.width * ratio), int(image.height * ratio)),
        Image.LANCZOS
    )

    progress(0.05, desc="Analyzing image with BLIP...")

    blip_caption = custom_prompt.strip() if custom_prompt.strip() else get_caption(image)
    prompt = build_outpaint_prompt(blip_caption, image)

    base_negative = (
        "blurry, bad quality, watermark, text, "
        "new person, new face, new subject, extra people, "
        "duplicate, tiled, repeated pattern, border, frame, "
        "seam, visible edge, abrupt change, inconsistent, "
        "distorted, unnatural, different style, different era"
    )
    negative_prompt = (
        base_negative + ", colorful, vibrant colors, color photography"
        if is_greyscale(image) else base_negative
    )
    # B&W images get extra negative terms to prevent SD adding color

    STEP_PX = 64
    extend  = extend_percent / 100.0
    h_total = int(image.width  * extend)
    v_total = int(image.height * extend)

    def make_passes(side, total_px):
        passes, remaining = [], total_px
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
            make_passes("bottom", v_total) + make_passes("top",   v_total) +
            make_passes("right",  h_total) + make_passes("left",  h_total)
        )

    total_passes  = len(passes)
    current_image = image

    for i, (side, px) in enumerate(passes):
        progress(
            0.1 + 0.85 * (i / total_passes),
            desc=f"Pass {i+1}/{total_passes} — extending {side} by {px}px"
        )
        current_image = extend_one_side(current_image, side, px, prompt, negative_prompt)

    progress(1.0, desc="Done!")

    bw_note = " [B&W detected]" if is_greyscale(image) else ""
    return current_image, f"Caption{bw_note}:\n{blip_caption}\n\nPrompt:\n{prompt}"


# ================================================================
# GRADIO UI
# ================================================================

with gr.Blocks(
                title="CanvasAI",
                theme= gr.themes.Soft(
                    primary_hue=gr.themes.colors.violet,
                    secondary_hue=gr.themes.colors.purple,
                    neutral_hue=gr.themes.colors.slate,
                    radius_size=gr.themes.sizes.radius_sm,
                )) as demo:

    gr.Markdown(""" # CanvasAI
    *AI-powered image enhancement, colorization, and outpainting*""")
    gr.Markdown(
        "**Enhance** with Real-ESRGAN  |  "
        "**Colorize** B&W photos with DDColor  |  "
        "**Outpaint** to extend any scene with SD2"
    )
    gr.Markdown("---")

    with gr.Tabs():
        #---------Enhance----------------
        with gr.Tab(" ✨ Enhance"):
            gr.Markdown("## Upscale and sharpen any image 2x or 4x using Real-ESRGAN.")
            with gr.Row(equal_height= True):
                with gr.Column(scale= 1):
                    enh_input = gr.Image(label="Upload Image", type="pil", height= 350)
                    enh_scale = gr.Radio(
                        choices=["2x", "4x"],
                        value="4x",
                        label="Upscale Factor",
                        info="4x recommended for most images. Use 2x for very large inputs."
                    )

                    enh_btn = gr.Button("Enhance", variant="primary", size="lg")
                with gr.Column(scale=1):
                    enh_output = gr.Image(label="Result", type="pil", interactive=False, height=350,)
                    enh_info   = gr.Textbox(label="Size Info", interactive=False, container=True, lines=2,)

            enh_btn.click(fn=enhance_image, inputs=[enh_input, enh_scale], 
                          outputs=[enh_output, enh_info])
        #--------Colourize---------------------------
        with gr.Tab(" 🎨 Colorize"):
            gr.Markdown("""
                ## B&W Photo Colorization
                DDColor adds natural, realistic colors while preserving the original structure."""
            )
            with gr.Row(equal_height= True):
                with gr.Column(scale=1):
                    col_input    = gr.Image(label="Upload B&W Image", type="pil", height=350)
                    col_strength = gr.Slider(
                        minimum=0.5, maximum=1.0, value=0.9, step=0.05,
                        label="Color Strength",
                        info="0.5 = subtle tint  ·  1.0 = full vivid color"
                    )
                    col_btn = gr.Button("Colorize", variant="primary", size="lg")
                with gr.Column():
                    col_output = gr.Image(label="Colorized Result", type="pil", interactive=False, height=350,)

            col_btn.click(fn=colour_image, inputs=[col_input, col_strength], outputs=[col_output])
        #----------Outpaint-----------------------------
        with gr.Tab("🔲 Outpaint"):
            gr.Markdown("""## Extend Any Scene
            Upload an image and extend it in any direction. 
            BLIP reads the scene automatically — no prompt needed."""
            )
            with gr.Row(equal_height= True):
                with gr.Column(scale= 1):
                    out_input  = gr.Image(label="Upload Image", type="pil", height=350)
                    out_dir    = gr.Radio(
                        choices=["Horizontal", "Vertical", "Both"], value="Horizontal",
                        label="Extension Direction",
                        info="Horizontal = left & right | Vertical = top & bottom | Both = all sides"
                    )
                    out_pct    = gr.Slider(minimum=10, maximum=50, value=25, step=5, label="Extend by (%)")
                    with gr.Accordion("Advanced: Custom Prompt", open=False):
                    # gr.Accordion = collapsible section
                    # open=False = collapsed by default — keeps UI clean
                    # Users who want it can expand, others aren't distracted
                        out_prompt = gr.Textbox(
                        label="Custom Prompt (optional)",
                        placeholder="Leave empty for auto-detection",
                        lines=2
                    )
                    out_btn = gr.Button("Outpaint", variant="primary", size="lg")
                with gr.Column(scale=1):
                    out_output  = gr.Image(label="Result", type="pil", interactive=False,)
                    out_caption = gr.Textbox(label="Prompt Used", interactive=False, lines=4)

            out_btn.click(
                fn=outpaint_image,
                inputs=[out_input, out_dir, out_pct, out_prompt],
                outputs=[out_output, out_caption]
            )

    gr.Markdown("""
                    ---
                    **Privacy:** Images are processed in memory and not stored permanently.
                    Uploaded images are deleted when your session ends.
                    Do not upload sensitive or private images.
                    """)



demo.launch(show_error= True)