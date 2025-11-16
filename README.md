# AI Clothing Designer 

AI Clothing Designer is a focused GenAI project that fuses heritage art motifs with modern garments. A TensorFlow-based neural style-transfer core blends textures and silhouettes, while a Streamlit front-end exposes every control needed for lab experiments, demos, or rapid iteration.

## Why It’s GenAI
- **Neural Style Transfer (NST)**: Uses VGG19 feature extraction plus tunable style/content/TV losses to synthesize new apparel textures that never existed before.
- **Adaptive Optimization Controls**: Iterations, learning-rate decay, color blending, and per-loss weights expose creative freedom typical of GenAI workflows.
- **Live Feedback Loop**: Streamlit UI streams iteration logs and a progress bar so users see how the model evolves each run—mirroring modern GenAI “prompt → preview → refine” cycles.
- **Asset Remixing**: Any garment base + cultural art texture combo can be recomputed without retraining, enabling human-in-the-loop co-creation.

## System Architecture
1. **Streamlit UI (`ML/streamlit_app.py`)**
	- Garment and design uploaders or artform dropdown.
	- Sidebar sliders for iterations, color maps, blend ratio, and advanced loss weights.
	- Progress bar that polls the backend’s `/progress/<job_id>` endpoint and displays iteration/loss info.
	- Final image preview, download button, and log viewer.
2. **Flask ML Service (`ML/application.py`)**
	- `/style_transfer` and `/generate_style` endpoints orchestrate NST, color treatment, and background removal.
	- Adam optimizer with gradient clipping, learning-rate decay, and optional GPU acceleration (TensorFlow + `onnxruntime` providers).
	- Background removal via `rembg` (ONNX Runtime) for clean, transparent outputs.
	- Thread-safe progress tracker to support the Streamlit progress bar.
3. **Assets & Storage**
	- `ML/inputs/`: Base garment(s) such as `base_tshirt.png`.
	- `ML/examples/`: Curated PNG outputs for portfolio/lab submissions.
	- `ML/designs/<Artform>` (optional): Heritage motifs automatically sampled when using the “Surprise me” mode.

## Quick Start (Windows PowerShell)

```powershell
cd "C:\Users\Daanish Mittal\OneDrive\Desktop\GenAI-heritage\heritage-buzz-main\ML"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Terminal 1 – run the Flask/TensorFlow service
python application.py

# Terminal 2 – launch the Streamlit designer
cd "C:\Users\Daanish Mittal\OneDrive\Desktop\GenAI-heritage\heritage-buzz-main\ML"
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

> **GPU acceleration**: Install CUDA 12.x + cuDNN and `onnxruntime-gpu` for faster style transfer. Set env vars like `HB_ONNX_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider"` to force GPU usage.

## Experiment Workflow
1. **Select Inputs**: Upload your own garment + art image or pick an artform backed by `ML/designs/<name>`.
2. **Dial Parameters**:
	- `Iterations`: Governs texture richness (20–300).
	- `Color treatment`: Preserve original colors or apply heatmaps.
	- `Blend ratio`: Mix neural output with the garment’s original luminance.
	- `Style/Content/TV weights`: Fine-tune fidelity vs. creativity.
3. **Run & Observe**:
	- Progress bar shows `Iteration X/Y · loss …` updates.
	- Generation logs record optimizer checkpoints.
4. **Review & Export**:
	- Inspect the generated PNG.
	- Download or archive under `ML/examples/`.
	- Adjust knobs and rerun to explore the design space.

## Key Files & Directories
- `ML/application.py` – NST engine, Flask routes, progress tracker, rembg integration.
- `ML/streamlit_app.py` – UI/UX layer with progress polling, log viewer, and advanced controls.
- `ML/requirements.txt` – Reproducible dependency list (TensorFlow, rembg/onnxruntime, Streamlit, Pillow, etc.).
- `ML/inputs/` – Base garment assets.
- `ML/examples/` – Saved results for documentation.
- `ML/designs/` (optional) – Folder hierarchy for artform-specific motif libraries.

## Extending the Lab
- **New Artforms**: Drop PNG/JPG motifs into `ML/designs/<NewArt>` and add the name to the `ARTFORMS` list.
- **Preset Parameters**: Set env vars (`HB_STYLE_ITERATIONS`, `HB_STYLE_WEIGHT`, `HB_COLOR_BLEND`, etc.) to define defaults for demos.
- **Model Upgrades**: Swap out VGG19 NST for diffusion/LoRA pipelines or plug in segmentation masks for garment-only styling.
- **Data Logging**: Extend Streamlit to store user ratings or metadata in a CSV/Firebase for lab reports.

## Tech Stack
- **TensorFlow + Keras**: Feature extraction (VGG19) and iterative NST optimization.
- **ONNX Runtime + rembg**: Background matting with CPU/GPU provider selection.
- **Streamlit**: Rapid GenAI UI with real-time progress feedback.
- **Python (Flask, Pillow, NumPy)**: API transport, data wrangling, and image IO.

This README now captures the full GenAI journey—from architecture and experimentation to extensibility—so you can present AI Clothing Designer as a complete, reproducible generative-fashion lab.

