"""
Aplikasi web klasifikasi buah segar dan busuk (MobileNetV2 transfer learning).

Fitur:
- Prediksi 6 kelas (apel/pisang/jeruk, segar/busuk) beserta probabilitas tiap kelas
- Confidence threshold: prediksi di bawah ambang ditolak sebagai "tidak yakin / bukan buah yang dikenali"
- Grad-CAM: heatmap area gambar yang paling memengaruhi keputusan model
- (Proyek-1) Model TensorFlow Lite terkompresi sebagai alternatif model Keras
- (Proyek-1) Test-Time Augmentation: prediksi dirata-rata dari 4 versi gambar

Jalankan lokal:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""
import glob
import json
import os
import time

import keras
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Klasifikasi Buah Segar & Busuk", page_icon="🍎", layout="wide")


@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    with open(os.path.join(HERE, "config.json")) as f:
        cfg = json.load(f)
    model = keras.models.load_model(os.path.join(HERE, cfg["model_file"]))
    parts = {
        "base": next(l for l in model.layers if l.name.startswith("mobilenetv2")),
        "rescale": model.get_layer("rescale"),
        "gap": model.get_layer("gap"),
        "pred": model.get_layer("pred"),
    }
    return cfg, model, parts


@st.cache_resource(show_spinner="Memuat model TFLite...")
def load_tflite(path):
    interp = tf.lite.Interpreter(model_path=path, num_threads=2)
    inp = interp.get_input_details()[0]["index"]
    out = interp.get_output_details()[0]["index"]
    interp.resize_tensor_input(inp, [1, 224, 224, 3])
    interp.allocate_tensors()
    return interp, inp, out


CFG, MODEL, PARTS = load_model()
TFLITE_FILES = {
    "TFLite " + os.path.basename(p).replace("fruit_mobilenetv2_", "").replace(".tflite", ""): p
    for p in sorted(glob.glob(os.path.join(HERE, "fruit_mobilenetv2_*.tflite")))
}
IMG_SIZE = CFG["img_size"]
CLASSES = CFG["classes"]
NAMES = CFG["display_names"]
DEFAULT_THRESHOLD = round(float(CFG["confidence_threshold"]), 2)


def preprocess(pil_img):
    """Sama dengan pipeline training: RGB -> resize bilinear 224x224 -> uint8 0-255."""
    arr = np.asarray(pil_img.convert("RGB"))
    arr = tf.image.resize(arr, (IMG_SIZE, IMG_SIZE)).numpy()
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def grad_cam(img_uint8):
    """Mengembalikan heatmap Grad-CAM (7x7) dan probabilitas 6 kelas."""
    x = tf.cast(img_uint8[None], tf.float32)
    with tf.GradientTape() as tape:
        conv = PARTS["base"](PARTS["rescale"](x), training=False)
        tape.watch(conv)
        probs = PARTS["pred"](PARTS["gap"](conv))
        idx = int(tf.argmax(probs[0]))
        score = probs[:, idx]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * weights[:, None, None, :], axis=-1))[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), probs[0].numpy()


def tta_views(img_uint8):
    """Asli, flip horizontal, flip vertikal, rotasi 180 derajat."""
    return [img_uint8, img_uint8[:, ::-1], img_uint8[::-1, :], img_uint8[::-1, ::-1]]


def predict_probs(img_uint8, engine, use_tta):
    """Probabilitas 6 kelas dari model Keras atau TFLite, dengan atau tanpa TTA."""
    views = tta_views(img_uint8) if use_tta else [img_uint8]
    batch = np.stack([np.ascontiguousarray(v) for v in views]).astype(np.float32)
    if engine == "Keras":
        probs = MODEL(batch, training=False).numpy()
    else:
        interp, inp, out = load_tflite(TFLITE_FILES[engine])
        probs = []
        for x in batch:
            interp.set_tensor(inp, x[None])
            interp.invoke()
            probs.append(interp.get_tensor(out)[0].copy())
        probs = np.array(probs)
    return probs.mean(axis=0)


def overlay(img_uint8, cam, alpha=0.45):
    heat = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE)).numpy()[..., 0]
    color = matplotlib.colormaps["jet"](heat)[..., :3]
    out = (1 - alpha) * img_uint8 / 255.0 + alpha * color
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


# ---------------------------- Tampilan ----------------------------
st.title("🍎🍌🍊 Klasifikasi Buah Segar dan Busuk")
st.markdown(
    "Unggah foto **apel, pisang, atau jeruk**. Model CNN **MobileNetV2** (transfer learning) "
    "akan menentukan jenis buah dan apakah buah tersebut **segar** atau **busuk**."
)

with st.sidebar:
    st.header("Pengaturan")
    threshold = st.slider("Confidence threshold", 0.40, 0.99, DEFAULT_THRESHOLD, 0.01,
                          help=f"Default {DEFAULT_THRESHOLD:.2f}, dipilih dari data validation saat training. "
                               "Prediksi dengan keyakinan di bawah ambang ini ditolak.")
    show_cam = st.checkbox("Tampilkan Grad-CAM", value=True)
    engines = ["Keras"] + list(TFLITE_FILES)
    engine = st.selectbox("Model", engines, index=0,
                          help="Keras = model asli (Tugas-1). TFLite = model terkompresi (Proyek-1), "
                               "ukuran jauh lebih kecil dengan akurasi yang hampir sama.")
    use_tta = st.checkbox("Test-Time Augmentation (TTA)", value=False,
                          help="Prediksi dirata-rata dari 4 versi gambar (asli, flip horizontal, "
                               "flip vertikal, rotasi 180°) agar lebih stabil.")
    st.divider()
    st.subheader("Cara penggunaan")
    st.markdown(
        "1. Unggah foto buah (JPG/PNG) atau ambil foto dengan kamera.\n"
        "2. Lihat hasil prediksi dan probabilitas setiap kelas.\n"
        "3. Grad-CAM menunjukkan area yang paling memengaruhi keputusan model "
        "(merah/kuning = paling berpengaruh).\n\n"
        "Gunakan foto **satu buah** dengan pencahayaan yang cukup."
    )
    st.divider()
    st.caption("Dataset: Fruits fresh and rotten for classification (Kaggle). "
               "Model: MobileNetV2 (ImageNet) + fine-tuning, akurasi test 99,74%.")

tab_upload, tab_camera = st.tabs(["📁 Unggah gambar", "📷 Kamera"])
with tab_upload:
    uploaded = st.file_uploader("Pilih foto buah", type=["jpg", "jpeg", "png", "webp"])
with tab_camera:
    captured = st.camera_input("Ambil foto buah")

source = uploaded or captured
if source is None:
    st.info("Silakan unggah atau ambil foto buah untuk memulai.")
    st.stop()

try:
    pil_img = Image.open(source)
except Exception:
    st.error("File tidak dapat dibaca sebagai gambar. Coba file JPG atau PNG lain.")
    st.stop()

t0 = time.time()
img = preprocess(pil_img)
probs = predict_probs(img, engine, use_tta)
elapsed = (time.time() - t0) * 1000
cam, _ = grad_cam(img) if show_cam else (None, None)

idx = int(probs.argmax())
conf = float(probs[idx])
top_name = NAMES[CLASSES[idx]]

col_img, col_res = st.columns(2)
with col_img:
    st.image(pil_img.convert("RGB"), caption="Gambar yang diunggah", width="stretch")
    if show_cam:
        st.image(overlay(img, cam), caption="Grad-CAM (dihitung dari model Keras pada gambar asli)", width="stretch")

with col_res:
    if conf >= threshold:
        condition = "segar" if CLASSES[idx].startswith("fresh") else "busuk"
        box = st.success if condition == "segar" else st.warning
        box(f"### {'✅' if condition == 'segar' else '⚠️'} {top_name}\n"
            f"Buah ini diprediksi **{condition}** dengan confidence **{conf:.2%}** (ambang {threshold:.2f}).")
    else:
        st.error(f"### ❓ Tidak yakin / bukan buah yang dikenali\n"
                 f"Confidence tertinggi hanya **{conf:.2%}** ({top_name}), di bawah ambang **{threshold:.2f}**. "
                 "Model hanya mengenali **apel, pisang, dan jeruk**.")

    st.subheader("Probabilitas per kelas")
    df = (pd.DataFrame({"Kelas": [NAMES[c] for c in CLASSES], "Probabilitas": probs.astype(float)})
            .sort_values("Probabilitas", ascending=False).reset_index(drop=True))
    for _, row in df.iterrows():
        st.progress(row["Probabilitas"], text=f"{row['Kelas']}: {row['Probabilitas']:.2%}")
    st.caption(f"Model: {engine} | TTA: {'aktif (4 versi gambar)' if use_tta else 'nonaktif'} | "
               f"Waktu inferensi: {elapsed:.0f} ms")
