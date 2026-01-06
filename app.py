import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Auto Logo Remover", layout="wide")
st.title("🤖 AI Auto Logo Remover")

uploaded_file = st.file_uploader("Nano Banana පින්තූරය Upload කරන්න...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    original_image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(original_image)
    bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("මුල් පින්තූරය")
        st.image(original_image)

    if st.button("Auto Detect & Remove Logo"):
        with st.spinner('ලෝගෝ එක සොයමින් පවතී...'):
            # 1. Image එක Grayscale කිරීම (කළු සුදු)
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            
            # 2. Thresholding (දීප්තිමත් සුදු පාට ලෝගෝ එක වෙන් කර ගැනීම)
            # ලෝගෝ එක සුදු පාට නම් මෙය හොඳින් වැඩ කරයි
            _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            # 3. පින්තූරයේ මැද කොටස අතහැර කොන් වල ඇති දේවල් පමණක් තෝරා ගැනීම
            # (ලෝගෝ සාමාන්‍යයෙන් මැද නොමැති නිසා)
            h, w = mask.shape
            mask[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)] = 0

            # 4. Inpainting (ලෝගෝ එක ඉවත් කිරීම)
            # Mask එක ටිකක් ඝනකම් කිරීම (Dilation) ලෝගෝ එක වටේ ඇති ඉරි මැකීමට උදව් වේ
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            result_bgr = cv2.inpaint(bgr_img, mask, 7, cv2.INPAINT_TELEA)
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("ලෝගෝ එක ඉවත් කළ පසු")
            st.image(result_rgb)
            
            # Download Button
            st.download_button(
                label="Download Cleaned Image",
                data=cv2.imencode('.jpg', result_bgr)[1].tobytes(),
                file_name="auto_cleaned.jpg",
                mime="image/jpeg"
            )
