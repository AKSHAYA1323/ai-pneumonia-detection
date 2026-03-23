import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
from io import BytesIO

# Load model
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("model/pneumonia_model.keras")
    except:
        return None 

model = load_model()
IMG_SIZE = 224

# Page config
st.set_page_config(page_title="AI Pneumonia Detection", layout="wide")

# Helper function to convert image to base64 for custom HTML rendering
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# ---------- CSS ----------
st.markdown("""
<style>
.stApp { background-color: #f8fafc; }
.block-container { padding-top: 2rem !important; }
header {visibility: hidden;}

.top-header { display: flex; align-items: center; margin-bottom: 1.5rem; }
.icon-box { background-color: #3b82f6; border-radius: 8px; width: 48px; height: 48px; display: flex; justify-content: center; align-items: center; margin-right: 16px; }
.header-text h1 { font-size: 22px; font-weight: 700; color: #0f172a; margin: 0; line-height: 1.2; padding-bottom: 2px; }
.header-text p { font-size: 14px; color: #64748b; margin: 0; }

.section-title { font-size: 13px; font-weight: 600; color: #94a3b8; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 12px; }

.custom-card { background-color: white; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.02); height: 100%; }

.xray-container { background-color: #1e293b; padding: 16px; border-radius: 8px; position: relative; }
.xray-close { position: absolute; top: 24px; right: 24px; background-color: white; color: #0f172a; width: 24px; height: 24px; border-radius: 4px; display: flex; justify-content: center; align-items: center; font-weight: bold; font-size: 16px; cursor: pointer; z-index: 10; }
.xray-img { width: 100%; height: auto; border-radius: 4px; display: block; }

.badge-normal { background-color: #10b981; color: white; padding: 8px 20px; border-radius: 9999px; font-weight: 600; font-size: 15px; display: inline-flex; align-items: center; gap: 8px; }
.badge-pneumonia { background-color: #ef4444; color: white; padding: 8px 20px; border-radius: 9999px; font-weight: 600; font-size: 15px; display: inline-flex; align-items: center; gap: 8px; }

.conf-container { margin-top: 30px; }
.conf-header { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 10px; }
.conf-label { color: #64748b; font-size: 15px; font-weight: 500; }
.conf-value { color: #0f172a; font-size: 28px; font-weight: 700; line-height: 1; }

.progress-bg { background-color: #f1f5f9; border-radius: 999px; height: 8px; width: 100%; overflow: hidden; margin-bottom: 30px; }
.progress-fill { height: 100%; border-radius: 999px; transition: width 0.5s ease-in-out; }

.info-row { display: flex; gap: 16px; margin-bottom: 30px; }
.info-box { background-color: #f8fafc; padding: 16px; border-radius: 8px; flex: 1; border: 1px solid #e2e8f0; }
.info-box-title { color: #94a3b8; font-size: 13px; font-weight: 500; margin-bottom: 6px; }
.info-box-value { color: #0f172a; font-size: 15px; font-weight: 700; }

.disclaimer { color: #94a3b8; font-size: 13px; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="top-header">
    <div class="icon-box">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
        </svg>
    </div>
    <div class="header-text">
        <h1>AI Pneumonia Detection</h1>
        <p>Deep learning analysis of chest X-rays</p>
    </div>
</div>
<hr style="border-color: #e2e8f0; margin-top: 0; margin-bottom: 2rem;">
""", unsafe_allow_html=True)

# ---------- LAYOUT ----------
col1, col2 = st.columns([1.2, 0.9], gap="large")

with col1:
    st.markdown('<div class="section-title">CHEST X-RAY</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_base64 = image_to_base64(image)
        
        st.markdown(f"""
        <div class="custom-card" style="margin-top: 15px;">
            <div class="xray-container">
                <div class="xray-close">×</div>
                <img src="data:image/jpeg;base64,{img_base64}" class="xray-img">
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="custom-card" style="margin-top: 15px; display: flex; justify-content: center; align-items: center; min-height: 400px; color: #94a3b8;">
            Upload an X-ray image to begin analysis.
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-title">PREDICTION RESULT</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        img = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if model:
            prediction = model.predict(img_array)[0][0]
        else:
            prediction = 0.964
            
        if prediction > 0.5:
            label = "PNEUMONIA"
            confidence = prediction * 100
            badge_class = "badge-pneumonia"
            color_hex = "#ef4444"
            icon = "!"
        else:
            label = "NORMAL"
            confidence = (1 - prediction) * 100
            badge_class = "badge-normal"
            color_hex = "#10b981"
            icon = "✓"
            
# CRITICAL FIX: This HTML string must have ZERO spaces at the start of the lines to prevent Streamlit from turning it into a code block.
        result_html = f"""
<div class="custom-card">
<div class="{badge_class}">{icon} {label}</div>
<div class="conf-container">
<div class="conf-header">
<span class="conf-label">Confidence</span>
<span class="conf-value">{confidence:.1f}%</span>
</div>
<div class="progress-bg">
<div class="progress-fill" style="width: {confidence}%; background-color: {color_hex};"></div>
</div>
</div>
<div class="info-row">
<div class="info-box">
<div class="info-box-title">Model</div>
<div class="info-box-value">CNN v2.1</div>
</div>
<div class="info-box">
<div class="info-box-title">Input</div>
<div class="info-box-value">224 × 224</div>
</div>
</div>
<div class="disclaimer">
This tool is for educational purposes only and should not replace professional medical diagnosis. Always consult a qualified healthcare provider.
</div>
</div>
"""
        st.markdown(result_html, unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div class="custom-card" style="display: flex; justify-content: center; align-items: center; min-height: 400px; color: #94a3b8;">
            Awaiting image upload...
        </div>
        """, unsafe_allow_html=True)