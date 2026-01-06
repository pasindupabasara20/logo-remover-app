import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Logo Remover", layout="wide")
st.title("🖼️ Nano Banana Image Logo Remover")
st.write("පින්තූරය Upload කර ලෝගෝ එක මත Brush එකෙන් පාට කරන්න. පසුව 'Remove Logo' ක්ලික් කරන්න.")

# Sidebar Settings
st.sidebar.header("Settings")
brush_width = st.sidebar.slider("Brush Width:", 1, 50, 15)

# 1. Image Upload
uploaded_file = st.file_uploader("ඔබේ පින්තූරය මෙතැනට දාන්න...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load Image
    original_image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(original_image)
    
    # Canvas එක පෙන්වීම (මෙහිදී පරිශීලකයා ලෝගෝ එක පාට කළ යුතුය)
    st.subheader("ලෝගෝ එක ඇති තැන පාට කරන්න (Masking)")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1.0)",  # Mask color (White)
        stroke_width=brush_width,
        stroke_color="#FFFFFF",
        background_image=original_image,
        update_streamlit=True,
        height=img_array.shape[0] * (600 / img_array.shape[1]) if img_array.shape[1] > 600 else img_array.shape[0],
        width=600 if img_array.shape[1] > 600 else img_array.shape[1],
        drawing_mode="freedraw",
        key="canvas",
    )

    # 2. Process Button
    if st.button("Remove Logo"):
        if canvas_result.image_data is not None:
            # Mask එක සකසා ගැනීම
            mask = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
            mask = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]))
            
            # Inpainting (ලෝගෝ එක අයින් කිරීම)
            # මුල් පින්තූරය OpenCV format (BGR) එකට හැරවීම
            bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            result_bgr = cv2.inpaint(bgr_img, mask, 3, cv2.INPAINT_TELEA)
            
            # නැවත RGB වලට හැරවීම
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            result_img = Image.fromarray(result_rgb)
            
            # ප්‍රතිඵලය පෙන්වීම
            st.subheader("ප්‍රතිඵලය (Cleaned Image)")
            st.image(result_img)
            
            # Download Button
            st.download_button(
                label="පින්තූරය Download කරගන්න",
                data=cv2.imencode('.jpg', result_bgr)[1].tobytes(),
                file_name="cleaned_image.jpg",
                mime="image/jpeg"
            )
        else:
            st.warning("කරුණාකර ලෝගෝ එක මත brush එකෙන් පාට කරන්න.")