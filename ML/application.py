import base64
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
from PIL import Image
from rembg import remove
import onnxruntime as ort
import cv2
from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
import os
import random
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple, List

try:
    for _gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

# List of available colormaps
COLORMAP_CHOICES = {
    "original": None,
    "jet": cv2.COLORMAP_JET,
    "hot": cv2.COLORMAP_HOT,
    "cool": cv2.COLORMAP_COOL,
    "rainbow": cv2.COLORMAP_RAINBOW,
    "ocean": cv2.COLORMAP_OCEAN,
    "summer": cv2.COLORMAP_SUMMER,
}

DEFAULT_ITERATIONS = int(os.getenv("HB_STYLE_ITERATIONS", "30"))
DEFAULT_COLORMAP = os.getenv("HB_STYLE_COLORMAP", "original").lower()
BASE_IMAGE_FALLBACK = Path("./inputs/base_tshirt.png")
DESIGNS_ROOT = Path("./designs")
DEFAULT_STYLE_WEIGHT = float(os.getenv("HB_STYLE_WEIGHT", "1e-6"))
DEFAULT_CONTENT_WEIGHT = float(os.getenv("HB_CONTENT_WEIGHT", "2.5e-8"))
DEFAULT_TV_WEIGHT = float(os.getenv("HB_TV_WEIGHT", "1e-6"))
DEFAULT_COLOR_BLEND = float(os.getenv("HB_COLOR_BLEND", "0.7"))
INITIAL_LR = float(os.getenv("HB_INITIAL_LR", "8.0"))
LR_DECAY_EVERY = int(os.getenv("HB_LR_DECAY_EVERY", "25"))
LR_DECAY_RATE = float(os.getenv("HB_LR_DECAY_RATE", "0.8"))


def _as_bool(value) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

# Open, resize, and format picture into tensors
def preprocess_image(image_path, img_nrows, img_ncols):
    img = keras.preprocessing.image.load_img(image_path, target_size=(img_nrows, img_ncols))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

# Tensor to image
def deprocess_image(x, img_nrows, img_ncols):
    x = x.reshape((img_nrows, img_ncols, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Gram matrix of image tensor (feature-wise outer product)
def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_loss(style, combination, img_nrows, img_ncols):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))

def total_variation_loss(x, img_nrows, img_ncols):
    a = tf.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = tf.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))

def compute_loss(combination_image, base_image, style_reference_image, feature_extractor, content_layer_name, content_weight, style_layer_names, style_weight, total_variation_weight, img_nrows, img_ncols):
    input_tensor = tf.concat([base_image, style_reference_image, combination_image], axis=0)
    features = feature_extractor(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())

    # Add content loss
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(base_image_features, combination_features)
    
    # Add style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features, img_nrows, img_ncols)
        loss += (style_weight / len(style_layer_names)) * sl

    # Add total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image, img_nrows, img_ncols)
    return loss

@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image, feature_extractor, content_layer_name, content_weight, style_layer_names, style_weight, total_variation_weight, img_nrows, img_ncols):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image, feature_extractor, content_layer_name, content_weight, style_layer_names, style_weight, total_variation_weight, img_nrows, img_ncols)
    grads = tape.gradient(loss, combination_image)
    return loss, grads

def _normalize_colormap(name: Optional[str]) -> Optional[int]:
    if not name:
        return COLORMAP_CHOICES.get(DEFAULT_COLORMAP, None)
    return COLORMAP_CHOICES.get(name.lower(), None)


def _apply_colormap(image_path: Path, colormap: Optional[int]) -> Path:
    if colormap is None:
        return image_path
    image = cv2.imread(str(image_path))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scaled_gray_image = cv2.convertScaleAbs(gray_image)
    heatmap = cv2.applyColorMap(scaled_gray_image, colormap)
    heatmap_path = Path(f"heatmap_{uuid.uuid4().hex}.png")
    cv2.imwrite(str(heatmap_path), heatmap)
    return heatmap_path


def _onnx_providers() -> list[str]:
    requested = os.getenv("HB_ONNX_PROVIDERS")
    if requested:
        return [provider.strip() for provider in requested.split(',') if provider.strip()]

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        # Prefer CUDA but keep CPU as fallback
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _remove_background(image_path: Path) -> Path:
    output_path = Path(f"output_{uuid.uuid4().hex}.png")
    sess_opts = ort.SessionOptions()
    providers = _onnx_providers()

    with open(image_path, 'rb') as src, open(output_path, 'wb') as dst:
        input_bytes = src.read()
        output_bytes = remove(input_bytes, session_options=sess_opts, providers=providers)
        dst.write(output_bytes)

    return output_path


def neural_style_transfer(
    base_image_path,
    style_reference_image_path,
    iterations=DEFAULT_ITERATIONS,
    colormap_name: Optional[str] = None,
    capture_logs: bool = False,
    style_weight: Optional[float] = None,
    content_weight: Optional[float] = None,
    total_variation_weight: Optional[float] = None,
    color_blend_ratio: Optional[float] = None,
) -> Tuple[str, List[str]]:
    total_variation_weight = total_variation_weight if total_variation_weight is not None else DEFAULT_TV_WEIGHT
    style_weight = style_weight if style_weight is not None else DEFAULT_STYLE_WEIGHT
    content_weight = content_weight if content_weight is not None else DEFAULT_CONTENT_WEIGHT
    color_blend_ratio = DEFAULT_COLOR_BLEND if color_blend_ratio is None else max(0.0, min(1.0, color_blend_ratio))
    logs: List[str] = []

    def _log(message: str):
        print(message)
        if capture_logs:
            logs.append(message)
    
    width, height = keras.preprocessing.image.load_img(base_image_path).size
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)

    model = vgg19.VGG19(weights='imagenet', include_top=False)
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)
    
    style_layer_names = [ 'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer_name = 'block5_conv2'
    optimizer = keras.optimizers.Adam(learning_rate=INITIAL_LR)
    base_image = preprocess_image(base_image_path, img_nrows, img_ncols)
    style_reference_image = preprocess_image(style_reference_image_path, img_nrows, img_ncols)
    combination_image = tf.Variable(preprocess_image(base_image_path, img_nrows, img_ncols))
    
    result_path = Path(f"styled_{uuid.uuid4().hex}.png")

    for i in range(int(iterations)):
        loss, grads = compute_loss_and_grads(
            combination_image,
            base_image,
            style_reference_image,
            feature_extractor,
            content_layer_name,
            content_weight,
            style_layer_names,
            style_weight,
            total_variation_weight,
            img_nrows,
            img_ncols,
        )
        grads = tf.clip_by_value(grads, -1.0, 1.0)
        optimizer.apply_gradients([(grads, combination_image)])
        if i % 10 == 0 or i == iterations - 1:
            _log(f'Iteration {i+1}/{iterations}: loss={loss:.2f}')
        if LR_DECAY_EVERY > 0 and (i + 1) % LR_DECAY_EVERY == 0:
            new_lr = float(optimizer.learning_rate.numpy()) * LR_DECAY_RATE
            optimizer.learning_rate.assign(new_lr)
            _log(f'Adjusted learning rate to {new_lr:.4f}')

    final_img = deprocess_image(combination_image.numpy(), img_nrows, img_ncols)

    if 0.0 <= color_blend_ratio < 1.0:
        base_for_blend = keras.preprocessing.image.load_img(base_image_path, target_size=(img_nrows, img_ncols))
        base_arr = keras.preprocessing.image.img_to_array(base_for_blend)
        final_img = (color_blend_ratio * final_img + (1.0 - color_blend_ratio) * base_arr).astype('uint8')

    keras.preprocessing.image.save_img(str(result_path), final_img)

    colormap_code = _normalize_colormap(colormap_name)
    processed_path = _apply_colormap(result_path, colormap_code)
    output_path = _remove_background(processed_path)

    try:
        if processed_path != result_path and processed_path.exists():
            processed_path.unlink()
        if result_path.exists():
            result_path.unlink()
    except OSError:
        pass

    return str(output_path), logs

base_image_path = './input_image.png'

app = Flask(__name__)
CORS(app)

@app.route('/style_transfer', methods=['POST'])
def style_transfer():
    design = request.files.get('design')
    if not design:
        return jsonify({'error': 'Design file is required'}), 400

    iterations = max(1, int(request.form.get('iterations') or DEFAULT_ITERATIONS))
    colormap_name = request.form.get('colormap') or DEFAULT_COLORMAP
    style_weight_val = _as_float(request.form.get('style_weight'), DEFAULT_STYLE_WEIGHT)
    content_weight_val = _as_float(request.form.get('content_weight'), DEFAULT_CONTENT_WEIGHT)
    tv_weight_val = _as_float(request.form.get('tv_weight'), DEFAULT_TV_WEIGHT)
    color_blend_val = _as_float(request.form.get('color_blend'), DEFAULT_COLOR_BLEND)
    return_logs = _as_bool(request.form.get('return_logs'))

    temp_design = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(design.filename or 'design.png')[1])
    temp_design.close()
    design.save(temp_design.name)

    base_file = request.files.get('base')
    if base_file:
        temp_base = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(base_file.filename or 'base.png')[1])
        temp_base.close()
        base_file.save(temp_base.name)
        base_image_path = temp_base.name
    else:
        base_image_path = str(BASE_IMAGE_FALLBACK)
        temp_base = None

    try:
        output_img, logs = neural_style_transfer(
            base_image_path,
            temp_design.name,
            iterations=iterations,
            colormap_name=colormap_name,
            capture_logs=return_logs,
            style_weight=style_weight_val,
            content_weight=content_weight_val,
            total_variation_weight=tv_weight_val,
            color_blend_ratio=color_blend_val,
        )
    finally:
        os.unlink(temp_design.name)
        if temp_base:
            os.unlink(temp_base.name)

    if return_logs:
        with open(output_img, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        return jsonify({'image_base64': encoded, 'logs': logs})

    return send_file(output_img, mimetype='image/png')

@app.route('/generate_style', methods=['POST'])
def generate_style():
    data = request.get_json(force=True)
    artform = data.get('artform')
    if not artform:
        return jsonify({'error': 'artform is required'}), 400

    iterations = max(1, int(data.get('iterations') or DEFAULT_ITERATIONS))
    colormap_name = data.get('colormap') or DEFAULT_COLORMAP
    style_weight_val = _as_float(data.get('style_weight'), DEFAULT_STYLE_WEIGHT)
    content_weight_val = _as_float(data.get('content_weight'), DEFAULT_CONTENT_WEIGHT)
    tv_weight_val = _as_float(data.get('tv_weight'), DEFAULT_TV_WEIGHT)
    color_blend_val = _as_float(data.get('color_blend'), DEFAULT_COLOR_BLEND)
    return_logs = _as_bool(data.get('return_logs'))

    artform_folder = DESIGNS_ROOT / artform
    if not artform_folder.exists() or not artform_folder.is_dir():
        return jsonify({'error': f'Artform folder "{artform}" not found under {DESIGNS_ROOT.resolve()}.'}), 400

    design_files = list(artform_folder.glob('*.*'))
    if not design_files:
        return jsonify({'error': f'No design assets found for artform "{artform}".'}), 400

    design_path = random.choice(design_files)
    output_img, logs = neural_style_transfer(
        str(BASE_IMAGE_FALLBACK),
        str(design_path),
        iterations=iterations,
        colormap_name=colormap_name,
        capture_logs=return_logs,
        style_weight=style_weight_val,
        content_weight=content_weight_val,
        total_variation_weight=tv_weight_val,
        color_blend_ratio=color_blend_val,
    )

    if return_logs:
        with open(output_img, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        return jsonify({'image_base64': encoded, 'logs': logs})
    
    return send_file(output_img, mimetype='image/png')

if __name__ == '__main__':
    app.run()
