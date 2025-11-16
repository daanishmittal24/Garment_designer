# Heritage Buzz · GenAI Lab Demo

- Neural style-transfer lab that blends Indian heritage art with apparel mockups entirely in Python/Streamlit.
- Single ML service (`application.py`) exposes `/style_transfer` and `/generate_style` endpoints for high-quality blends.
- Lightweight Streamlit UI (`streamlit_app.py`) lets you upload garments/designs, tweak iterations/colormaps, and download results.
- Example outputs live in `ML/examples/` while starter assets (base garment, future designs) live under `ML/inputs/`.
- Optional GPU acceleration via TensorFlow + CUDA or ONNX Runtime CUDA providers for faster experimentation.

## Repository Layout

- `ML/application.py` – Flask API running neural style transfer, background removal, and optional color treatments.
- `ML/streamlit_app.py` – Streamlit client for lab demos (upload garment + art, control iterations, preview/download result).
- `ML/requirements.txt` – Python dependencies (TensorFlow, rembg/onnxruntime, Streamlit, etc.).
- `ML/inputs/` – Sample garment (`base_tshirt.png`) plus space for your own cloth/heritage assets.
- `ML/examples/` – Curated PNGs showing what successful generations look like for reporting.

## Quick Start (Windows PowerShell)

```powershell
cd "C:\Users\Daanish Mittal\OneDrive\Desktop\GenAI-heritage\heritage-buzz-main\ML"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Terminal 1 – run the ML API
python application.py

# Terminal 2 – launch the Streamlit lab UI
cd "C:\Users\Daanish Mittal\OneDrive\Desktop\GenAI-heritage\heritage-buzz-main\ML"
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

## Running Experiments

- Upload a garment photo + art texture (PNG/JPG) or choose an artform to sample from `./designs/<name>`.
- Use the sidebar slider to increase style iterations for richer detail; pick a colormap or keep original colors.
- Download each PNG for lab reports and drop noteworthy results into `ML/examples/` for documentation.
- For GPU speed-ups install CUDA 12.x + cuDNN and swap to `onnxruntime-gpu`; otherwise the CPU path still works.

## Adding Your Own Assets

- Place additional garment bases inside `ML/inputs/` and point the Streamlit upload to them.
- Create folders such as `ML/designs/Madhubani` or `ML/designs/Warli` with multiple PNG/JPG motifs for `/generate_style`.
- Update the artform list in `streamlit_app.py` if you introduce new heritage styles.

This trimmed-down repository now focuses solely on the GenAI workflow required for lab submissions—no Node/React code, just the Python service, Streamlit front-end, and curated artefacts to showcase results.

