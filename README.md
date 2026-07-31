# ✨ Pixel Prompt (Streamlit, local Stable Diffusion model)

Direct Streamlit port of the original Gradio app. Loads Stable Diffusion 1.5
**locally** via `diffusers` + `torch` -- same behavior as the original
script, just running inside Streamlit instead of Gradio.

> ⚠️ **Security note:** the original script had a real Hugging Face token
> hardcoded as the argument to `input()`. That's been removed. If that
> token is still active, revoke it now at
> https://huggingface.co/settings/tokens.

## ⚠️ Free-tier hosting: read this first

This app needs to download and hold a multi-GB model in memory, and without
a GPU, generates images slowly (potentially several minutes each). Streamlit
Community Cloud's free tier gives roughly **1GB of RAM** and no GPU. In
practice, deploying this exact app there is likely to hit one of these:

- **Build/runtime crash** (`OOM` / "app has exceeded its resource limits")
  when the model loads into memory
- **The first request timing out** while the model downloads and loads
- **Extremely slow generation** (minutes per image) even if it does load,
  since it's running on CPU

You asked to try anyway, so this is set up to give it the best possible
shot on free hosting:
- `requirements.txt` installs the **CPU-only** build of `torch` (via
  `--extra-index-url https://download.pytorch.org/whl/cpu`), which is much
  smaller than the default CUDA-enabled wheel and more likely to fit within
  free-tier build size limits.
- The model is loaded once via `@st.cache_resource`, not on every rerun.

If it does crash or time out on Streamlit Cloud, that's the free tier's
memory limit, not a bug in the code. Your realistic next steps at that point
are: (a) run it locally instead (works fine on a normal laptop, just slow
without a GPU), or (b) move to a paid/GPU-backed host (Hugging Face Spaces
with a paid GPU tier, Render's paid plans, RunPod, etc.) -- happy to help
set either of those up if you get there.

## 🖥️ Run locally in VS Code

1. Open this folder in VS Code.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependencies (this will take a while -- torch + diffusers are
   large):
   ```bash
   pip install -r requirements.txt
   ```
4. If you need a Hugging Face token (only required for gated model repos),
   copy the secrets template and fill it in:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   Edit `.streamlit/secrets.toml` and paste your real token. This file is
   gitignored and will never be committed.
5. Run the app:
   ```bash
   streamlit run app.py
   ```
   Opens at `http://localhost:8501`. First run will download the ~4GB model
   -- subsequent runs use the cached copy.

## ☁️ Deploy on Streamlit Community Cloud (free tier)

1. Push this project to GitHub (see below).
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub.
3. **New app** → select your repo, branch `main`, main file `app.py`.
4. If you use a token, add it under **Settings → Secrets**:
   ```toml
   HF_TOKEN = "hf_your_real_token_here"
   ```
5. Deploy, and watch the build logs closely -- this is where you'll see if
   it hits the memory/size limits described above.

## 📦 Push this project to GitHub

```bash
cd pixel-prompt-streamlit-local
git init
git add .
git commit -m "Initial commit: Pixel Prompt (Streamlit, local SD model)"
git branch -M main
git remote add origin https://github.com/<your-username>/pixel-prompt.git
git push -u origin main
```

Before pushing, double-check no real token snuck into any file:
```bash
grep -r "hf_" . --include="*.py" --include="*.toml" | grep -v "hf_your_token_here"
```
This should print nothing.

## 🗂️ Project structure

```
pixel-prompt-streamlit-local/
├── app.py                          # Main Streamlit app (local SD model)
├── requirements.txt                 # Deps, pinned to CPU-only torch
├── .gitignore                       # Excludes secrets.toml, venv, outputs/
└── .streamlit/
    └── secrets.toml.example         # Template -- copy to secrets.toml locally
```
