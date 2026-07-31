import os
import uuid
from datetime import datetime

import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub import login

def get_hf_token():
    try:
        if "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass
    return os.environ.get("HF_TOKEN")


hf_token = get_hf_token()
if hf_token:
    login(token=hf_token)


model_id = "runwayml/stable-diffusion-v1-5"


@st.cache_resource(show_spinner="Loading Stable Diffusion model (this may take a minute)...")
def load_pipeline():
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    return pipe


pipe = load_pipeline()

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

st.set_page_config(page_title="Pixel Prompt", page_icon="✨", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background: linear-gradient(145deg, #0a0a0a, #1a1a1a);
        color: #fff;
    }
    .pixel-title {
        text-align: center;
        padding: 30px 0 10px 0;
    }
    .pixel-title h1 {
        font-size: 3em;
        font-weight: 800;
        background: linear-gradient(90deg, #007aff, #6e5ae2);
        -webkit-background-clip: text;
        color: transparent;
        margin-bottom: 0;
    }
    .pixel-title p {
        font-size: 1.1em;
        color: #aaa;
    }
    .prompt-box {
        backdrop-filter: blur(20px);
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 0 20px rgba(255,255,255,0.05);
    }
    div.stButton > button {
        background: linear-gradient(90deg, #007aff, #6e5ae2) !important;
        color: white !important;
        font-weight: 600;
        border-radius: 16px !important;
        border: none !important;
        padding: 0.6em 0 !important;
        width: 100%;
        transition: 0.3s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(110, 90, 226, 0.6);
    }
    #credits {
        text-align: center;
        margin-top: 30px;
        font-size: 0.8em;
        color: #888;
        letter-spacing: 0.5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="pixel-title">
        <h1>Pixel Prompt</h1>
        <p>Transform your imagination into visuals.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown('<div class="prompt-box">', unsafe_allow_html=True)
    prompt = st.text_area(
        "✨ Your Imagination",
        placeholder="e.g. A futuristic city in the clouds",
        height=100,
    )
    width = st.slider("Width (px)", 256, 1024, value=512, step=64)
    height = st.slider("Height (px)", 256, 1024, value=512, step=64)
    steps = st.slider("Inference Steps", 5, 50, value=25)
    guidance = st.slider("Guidance Scale (CFG)", 1.0, 20.0, value=7.5, step=0.1)
    generate_clicked = st.button("✨")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if generate_clicked:
        if not prompt.strip():
            st.warning("⚠️ Please enter a prompt.")
        else:
            with st.spinner("Generating your image... this can take a while on CPU."):
                try:
                    img_path = generate_image(prompt, width, height, steps, guidance)
                    st.session_state["last_image_path"] = img_path
                    st.session_state["last_timestamp"] = datetime.now().strftime("%H:%M:%S")
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    if "last_image_path" in st.session_state:
        st.image(st.session_state["last_image_path"], use_container_width=True)
        st.success(f"✅ Image generated at {st.session_state['last_timestamp']}")
        with open(st.session_state["last_image_path"], "rb") as f:
            st.download_button(
                "⬇️ Download Image",
                data=f.read(),
                file_name=os.path.basename(st.session_state["last_image_path"]),
                mime="image/png",
            )
    else:
        st.info("Your generated image will appear here.")

st.markdown(
    """
    <div id="credits">
        <p>© 2026 Pixel Prompt — Crafted by <b>Aditya Panwar</b> & <b>Raghav Mathur</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)
