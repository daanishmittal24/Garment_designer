import base64
import io
import os
import threading
import time
import uuid
from typing import Optional, List

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Heritage Buzz Studio", layout="centered")

st.title("🧵 Heritage Buzz – Style Studio")
st.write(
    "This lightweight Streamlit UI talks to the existing Flask style-transfer API so you can demo the GenAI pipeline without the React web app."
)

# Allow overriding the ML server location without touching code.
def _default_backend_url() -> str:
    return os.environ.get("HERITAGE_ML_URL", "http://localhost:5000")

backend_url = st.sidebar.text_input("ML API base URL", value=_default_backend_url())
st.sidebar.info(
    "Run `python application.py` inside the ML folder first so the `/style_transfer` and `/generate_style` endpoints are available."
)


def execute_with_progress(request_callable, job_id: str, total_iterations: int):
    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()
    response_holder: dict[str, requests.Response] = {}
    error_holder: dict[str, Exception] = {}

    def _worker():
        try:
            response_holder['response'] = request_callable()
        except Exception as exc:  # pylint: disable=broad-except
            error_holder['error'] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    while thread.is_alive():
        try:
            progress_resp = requests.get(f"{backend_url}/progress/{job_id}", timeout=5)
            if progress_resp.status_code == 200:
                payload = progress_resp.json()
                total = payload.get('total') or total_iterations
                current = payload.get('current') or 0
                loss_value = payload.get('loss')
                ratio = 0.0 if not total else min(max(current / total, 0.0), 1.0)
                label = f"Iteration {current}/{total}"
                if loss_value is not None:
                    label += f" · loss {loss_value:.2f}"
                status_placeholder.text(label)
                progress_bar.progress(ratio)
            else:
                status_placeholder.text("Waiting for progress …")
        except requests.exceptions.RequestException:
            status_placeholder.text("Waiting for progress …")
        time.sleep(0.8)

    thread.join()
    progress_bar.progress(1.0)
    status_placeholder.text("Completed.")

    if 'error' in error_holder:
        raise error_holder['error']
    return response_holder.get('response')

COLORMAP_OPTIONS = {
    "Preserve original colors": "original",
    "Jet heatmap": "jet",
    "Cool heatmap": "cool",
    "Ocean heatmap": "ocean",
    "Rainbow heatmap": "rainbow",
    "Summer heatmap": "summer",
}

iterations = st.sidebar.slider("Style iterations", min_value=20, max_value=300, value=40, step=10)
colormap_label = st.sidebar.selectbox("Color treatment", list(COLORMAP_OPTIONS.keys()))
colormap_value = COLORMAP_OPTIONS[colormap_label]
st.sidebar.caption("Higher iterations run longer but produce richer textures.")
color_blend = st.sidebar.slider(
    "Styled vs garment blend",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05,
    help="1.0 = keep neural colors, 0.0 = original garment colors"
)

with st.sidebar.expander("Advanced loss weights", expanded=False):
    style_weight_input = st.number_input(
        "Style weight",
        value=1e-6,
        min_value=1e-9,
        max_value=1e-3,
        format="%.1e",
        step=1e-7,
    )
    content_weight_input = st.number_input(
        "Content weight",
        value=2.5e-8,
        min_value=1e-10,
        max_value=1e-4,
        format="%.1e",
        step=1e-9,
    )
    tv_weight_input = st.number_input(
        "Total variation weight",
        value=1e-6,
        min_value=1e-9,
        max_value=1e-3,
        format="%.1e",
        step=1e-7,
    )

ARTFORMS = [
    "Madhubani",
    "Gond",
    "Warli",
    "Kalamkari",
    "Pattachitra",
]

session_image: Optional[Image.Image] = None
session_logs: Optional[List[str]] = None

mode = st.radio("Choose a generation mode", ["Upload custom art", "Surprise me with an artform"], horizontal=True)

if mode == "Upload custom art":
    base_upload = st.file_uploader("Optional: upload a garment/clothing photo", type=["png", "jpg", "jpeg"], key="base_uploader")
    design_upload = st.file_uploader("Upload a PNG/JPG heritage design", type=["png", "jpg", "jpeg"], key="design_uploader")
    if st.button("Blend art onto garment", disabled=design_upload is None):
        session_logs = None
        design_bytes = design_upload.getvalue()
        job_id = str(uuid.uuid4())

        files = {
            "design": (design_upload.name or "design.png", design_bytes, design_upload.type or "image/png"),
        }
        if base_upload is not None:
            files["base"] = (
                base_upload.name or "base.png",
                base_upload.getvalue(),
                base_upload.type or "image/png",
            )

        form_data = {
            "iterations": str(iterations),
            "colormap": colormap_value,
            "return_logs": "true",
            "style_weight": f"{style_weight_input}",
            "content_weight": f"{content_weight_input}",
            "tv_weight": f"{tv_weight_input}",
            "color_blend": f"{color_blend}",
            "request_id": job_id,
        }

        try:
            with st.spinner("Generating garment …"):
                response = execute_with_progress(
                    lambda: requests.post(
                        f"{backend_url}/style_transfer",
                        files=files,
                        data=form_data,
                        timeout=300,
                    ),
                    job_id,
                    iterations,
                )
        except requests.exceptions.RequestException as exc:
            st.error(f"Failed to reach ML server: {exc}")
            response = None
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Generation failed: {exc}")
            response = None
        if response and response.status_code == 200:
            if "application/json" in response.headers.get("Content-Type", ""):
                payload = response.json()
                image_bytes = base64.b64decode(payload.get("image_base64", ""))
                session_image = Image.open(io.BytesIO(image_bytes))
                session_logs = payload.get("logs", []) or None
            else:
                session_image = Image.open(io.BytesIO(response.content))
                session_logs = None
        elif response is not None:
            st.error(f"ML server error {response.status_code}: {response.text}")
else:
    artform = st.selectbox("Pick a heritage artform", ARTFORMS)
    if st.button("Generate with artform"):
        session_logs = None
        job_id = str(uuid.uuid4())
        payload = {
            "artform": artform,
            "iterations": iterations,
            "colormap": colormap_value,
            "return_logs": True,
            "style_weight": style_weight_input,
            "content_weight": content_weight_input,
            "tv_weight": tv_weight_input,
            "color_blend": color_blend,
            "request_id": job_id,
        }

        try:
            with st.spinner("Generating garment …"):
                response = execute_with_progress(
                    lambda: requests.post(
                        f"{backend_url}/generate_style",
                        json=payload,
                        timeout=300,
                    ),
                    job_id,
                    iterations,
                )
        except requests.exceptions.RequestException as exc:
            st.error(f"Failed to reach ML server: {exc}")
            response = None
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Generation failed: {exc}")
            response = None
        if response and response.status_code == 200:
            if "application/json" in response.headers.get("Content-Type", ""):
                payload = response.json()
                image_bytes = base64.b64decode(payload.get("image_base64", ""))
                session_image = Image.open(io.BytesIO(image_bytes))
                session_logs = payload.get("logs", []) or None
            else:
                session_image = Image.open(io.BytesIO(response.content))
                session_logs = None
        elif response is not None:
            st.error(f"ML server error {response.status_code}: {response.text}")

if session_image:
    st.success("Here is your GenAI garment!")
    st.image(session_image, caption="Streamlit preview", use_column_width=True)
    buf = io.BytesIO()
    session_image.save(buf, format="PNG")
    st.download_button(
        "Download PNG",
        data=buf.getvalue(),
        file_name="heritage_style.png",
        mime="image/png",
    )

if session_logs:
    st.subheader("Generation Logs")
    st.code("\n".join(session_logs))
