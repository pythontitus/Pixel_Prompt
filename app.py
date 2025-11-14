import os
import uuid
from datetime import datetime
from huggingface_hub import login
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import gradio as gr

# ---------------------------------------------
# 🔑 Hugging Face Login (replace input if preferred)
# ---------------------------------------------
hf_token = input("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").strip()
login(token=hf_token)

# ---------------------------------------------
# ⚙️ Model Loading
# ---------------------------------------------
model_id = "runwayml/stable-diffusion-v1-5"

if torch.cuda.is_available():
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

print("Loading Stable Diffusion model (this may take a minute)...")

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    safety_checker=None,
    requires_safety_checker=False
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    print("✅ Using GPU")
else:
    pipe = pipe.to("cpu")
    print("⚠️ Using CPU")

print("✅ Model loaded successfully!")

# ---------------------------------------------
# 🖼️ Image Generation Function
# ---------------------------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_image(prompt: str, width: int = 512, height: int = 512, steps: int = 25, guidance: float = 7.5):
    """Generates an image using Stable Diffusion and saves it."""
    width = int(width)
    height = int(height)
    result = pipe(prompt, height=height, width=width, num_inference_steps=int(steps), guidance_scale=float(guidance))
    image = result.images[0]
    filename = f"{uuid.uuid4().hex}.png"
    out_path = os.path.join(OUTPUT_DIR, filename)
    image.save(out_path)
    return out_path

# ---------------------------------------------
# 🌐 Gradio Web UI
# ---------------------------------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
body {
  background: linear-gradient(145deg, #0a0a0a, #1a1a1a);
  color: #fff !important;
  font-family: 'Inter', sans-serif !important;
  overflow-x: hidden;
}
.gradio-container {
  font-family: 'Inter', sans-serif !important;
}
#prompt-box {
  backdrop-filter: blur(20px);
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 20px;
  padding: 20px;
  box-shadow: 0 0 20px rgba(255,255,255,0.05);
  transition: 0.3s ease;
}
#prompt-box:hover {
  box-shadow: 0 0 40px rgba(255,255,255,0.1);
  transform: scale(1.01);
}
button.primary {
  background: linear-gradient(90deg, #007aff, #6e5ae2) !important;
  color: white !important;
  font-weight: 600;
  border-radius: 16px !important;
  padding: 10px 0 !important;
  transition: 0.3s ease;
  border: none !important;
}
button.primary:hover {
  transform: scale(1.05);
  box-shadow: 0 0 15px rgba(110, 90, 226, 0.6);
}
input, textarea {
  background: rgba(255, 255, 255, 0.08) !important;
  border: 1px solid rgba(255, 255, 255, 0.15) !important;
  color: #fff !important;
  border-radius: 12px !important;
}
footer { display: none !important; }
#credits {
  text-align: center;
  margin-top: 30px;
  font-size: 0.8em;
  color: #888;
  letter-spacing: 0.5px;
}
"""

def generate_and_download(prompt, width, height, steps, guidance):
    if not prompt.strip():
        return None, "⚠️ Please enter a prompt."
    img_path = generate_image(prompt, width, height, steps, guidance)
    timestamp = datetime.now().strftime("%H:%M:%S")
    return img_path, img_path, f"✅ Image generated at {timestamp}"

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.HTML("""
        <div style="text-align:center; padding: 50px 0 20px 0;">
            <h1 style="font-size:3em; font-weight:800; background:linear-gradient(90deg,#007aff,#6e5ae2); -webkit-background-clip:text; color:transparent;">
                Pixel Prompt
            </h1>
            <p style="font-size:1.2em; color:#aaa;">Transform your imagination into visuals.</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1, min_width=300, elem_id="prompt-box"):
            prompt = gr.Textbox(label="✨ Your Imagination", placeholder="e.g. A futuristic city in clouds", lines=3)
            width = gr.Slider(256, 1024, value=512, step=64, label="Width (px)")
            height = gr.Slider(256, 1024, value=512, step=64, label="Height (px)")
            steps = gr.Slider(5, 50, value=25, step=1, label="Inference Steps")
            guidance = gr.Slider(1.0, 20.0, value=7.5, step=0.1, label="Guidance Scale (CFG)")
            generate_btn = gr.Button("Generate ✨", variant="primary")
            status = gr.Markdown("", elem_id="status-text")

        with gr.Column(scale=2, min_width=600):
            output_image = gr.Image(label="Generated Output", show_label=False, interactive=False)
            download_btn = gr.File(label="⬇️ Download Image")

    generate_btn.click(
        fn=generate_and_download,
        inputs=[prompt, width, height, steps, guidance],
        outputs=[output_image, download_btn, status]
    )

    gr.HTML("""
        <div id="credits">
            <p>© 2025 Pixel Prompt — Crafted by <b>Aditya Panwar</b> & <b>Raghav Mathur</b></p>
        </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
