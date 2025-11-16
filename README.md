# AI Clothing Designer 

- Single-purpose GenAI lab that stylizes garments with Indian heritage art using Python, TensorFlow, and Streamlit.
- `ML/application.py` hosts the neural style-transfer + background removal pipeline with configurable loss weights and color blending.
- `ML/streamlit_app.py` offers a minimal UI to upload garments/art, tune iterations/colormaps, preview logs, and download the final PNG.
- Assets live under `ML/inputs/` (base garments), `ML/examples/` (reference outputs), and optional `ML/designs/<artform>` folders for preset styles.
- Works on CPU out of the box; optional CUDA/cuDNN + `onnxruntime-gpu` accelerate generation on NVIDIA hardware.

## Repository Layout

- `ML/application.py` – Flask API exposing `/style_transfer` and `/generate_style` endpoints with tunable style/content/TV weights, Adam optimizer, and color blending.
- `ML/streamlit_app.py` – Streamlit client with garment/design uploaders, sliders for iterations/blend ratio, advanced weight controls, and log streaming.
- `ML/requirements.txt` – Dependencies (TensorFlow, rembg, onnxruntime, Streamlit, Pillow, etc.).
- `ML/inputs/` – Starter garment assets (`base_tshirt.png`).
- `ML/examples/` – Sample generations for lab documentation.

## Quick Start (Windows PowerShell)

```powershell
cd "C:\Users\Daanish Mittal\OneDrive\Desktop\GenAI-heritage\heritage-buzz-main\ML"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Terminal 1 – run the ML API
python application.py

# Terminal 2 – launch the Streamlit UI
cd "C:\Users\Daanish Mittal\OneDrive\Desktop\GenAI-heritage\heritage-buzz-main\ML"
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

## Running Experiments

- Upload a garment photo plus an art texture, or choose an artform backed by files in `ML/designs/<name>`.
- Adjust iterations (detail), color treatment, blend ratio (styled vs garment colors), and loss weights from the sidebar.
- Review the “Generation Logs” section to monitor iteration-by-iteration loss values.
- Download the PNG output and optionally archive it under `ML/examples/` for reports.

## Customizing Assets

- Add new garment bases to `ML/inputs/` and select them via the Streamlit uploader.
- Populate `ML/designs/<Artform>` with multiple PNG/JPG motifs so `/generate_style` can randomly sample them.
- Extend the `ARTFORMS` list in `streamlit_app.py` whenever you add new heritage categories.
- Tweak default weights or iteration counts via environment variables (`HB_STYLE_WEIGHT`, `HB_STYLE_ITERATIONS`, etc.) for lab presets.

The repo now acts as a focused AI Clothing Designer lab: a reproducible Python backend plus a simple Streamlit surface for showcasing GenAI-driven fashion blends.

