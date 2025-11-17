import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageStat
from math import log

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model("model_sorter_hb_om_huggingface.h5")

class_names = [
    'Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash',
    'Paper', 'Plastic', 'Textile Trash', 'Vegetation'
]

# Quality reference points
AVG_BRIGHTNESS = 130
AVG_CONTRAST = 50
AVG_SHARPNESS = 3.5


# -----------------------------
# IMAGE QUALITY METRICS
# -----------------------------
def image_stats(img):
    stat = ImageStat.Stat(img)
    brightness = sum(stat.mean) / 3
    contrast = sum(stat.stddev) / 3

    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float32)
    lap = np.abs(arr[:-1, :-1] - arr[1:, :-1] - arr[:-1, 1:] + arr[1:, 1:])
    sharpness = float(np.mean(lap))

    return brightness, contrast, sharpness


def rating(value, avg):
    if value >= avg * 1.1:
        return "Above Average ✔"
    elif value >= avg * 0.9:
        return "Normal ~"
    else:
        return "Below Average ✖"


# -----------------------------
# MAIN PREDICTION FUNCTION
# -----------------------------
def analyze(img):

    # Resize + preprocess
    img_resized = img.resize((224, 224))
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img_resized))
    arr = np.expand_dims(arr, 0)

    # Predict
    preds = model.predict(arr)[0]
    idx = np.argmax(preds)
    confidence = float(preds[idx])

    # Top 3
    top3_idx = preds.argsort()[-3:][::-1]
    top3 = [(class_names[i], float(preds[i])) for i in top3_idx]

    # Certainty metrics
    entropy = -sum(p * log(p + 1e-12) for p in preds)
    margin = preds[top3_idx[0]] - preds[top3_idx[1]]

    if margin > 0.30 and entropy < 0.40:
        certainty = "High Confidence"
    elif margin > 0.18 and entropy < 0.80:
        certainty = "Medium Confidence"
    else:
        certainty = "Low Confidence"

    # Image quality
    brightness, contrast, sharpness = image_stats(img)

    # Build readable output
    result = {
        "Prediction": class_names[idx],
        "Confidence (%)": round(confidence * 100, 2),
        "Certainty Level": certainty,
        "Entropy": float(entropy),
        "Margin (Top1 - Top2)": float(margin),

        "Top 3 Predictions": {
            top3[0][0]: round(top3[0][1], 4),
            top3[1][0]: round(top3[1][1], 4),
            top3[2][0]: round(top3[2][1], 4)
        },

        "Image Quality": {
            "Brightness": f"{brightness:.1f} — {rating(brightness, AVG_BRIGHTNESS)}",
            "Contrast": f"{contrast:.1f} — {rating(contrast, AVG_CONTRAST)}",
            "Sharpness": f"{sharpness:.2f} — {rating(sharpness, AVG_SHARPNESS)}"
        }
    }

    return result, preds.tolist()


# -----------------------------
# GRADIO UI
# -----------------------------
with gr.Blocks(title="Waste Sorter AI") as demo:

    gr.Markdown("""
    # ♻ Waste Classification AI  
    Upload an image to classify waste + analyze image quality + check model certainty.
    """)

    with gr.Row():
        inp = gr.Image(type="pil", label="Upload Image")
        out_json = gr.JSON(label="Analysis")
        chart = gr.BarPlot(
            label="Prediction Distribution",
            x="class",
            y="probability",
            x_title="Class",
            y_title="Probability",
        )

    def format_chart(preds):
        return {
            "class": class_names,
            "probability": preds
        }

    inp.change(
        lambda img: (*analyze(img),),
        inputs=inp,
        outputs=[out_json, chart],
        postprocess=False
    )

    demo.launch(server_name="0.0.0.0", server_port=10000)

app = demo
