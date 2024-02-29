# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st
import torch
# Local Modules
import settings
import helper
import streamlit as st
import google.generativeai as genai


# Setting page layout
st.set_page_config(
    page_title="You Are Hungry",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Main page heading
st.title("AI Refrigerator🐷")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation', 'EMOTION'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)
elif model_type == 'EMOTION':
    model_path = Path(settings.EMOTION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
if source_radio == settings.IMAGE:
    source_img = st.camera_input("Take a picture")

    with st.spinner("Processing image..."):
        try:
            uploaded_image = PIL.Image.open(source_img)
            res = model.predict(uploaded_image, conf=confidence)
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Image', use_column_width=True)
            try:
                for box in boxes:
                    # Extract predicted class names
                    predicted_class_names = res[0].names
                    # Get the class name corresponding to the index 1
                    class_label = box.data[0][5].item()
                    predicted_class = predicted_class_names[class_label]
                    
                    # Modify prompt generation here
                    prompt_with_food = predicted_class + "할 때 먹는 음식"
                    
                    # Generate response with modified prompt
                    model = genai.GenerativeModel("gemini-pro")
                    response = model.generate_content(prompt_with_food)

                    # Display response
                    st.write("")
                    st.header(":blue[Just Try It!]")
                    st.write("")
                    st.write(predicted_class,'하시군요!')
                    st.markdown(response.text)
                    
            except Exception as ex:
                st.write("Error occurred while processing detection results.")
                st.error(ex)
        except Exception as ex:
            st.error("Error occurred while processing image.")
            st.error(ex)

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")

# st.image("/Users/chaewooklee/Desktop/streamlit/work/streamlit/images/Google-Gemini-AI-Logo.png", width=200)
# Sidebar에 API 키 입력
API_KEY = st.sidebar.text_input("Enter Google API Key", type="password")
genai.configure(api_key=API_KEY)
st.write("for your food")

gemini_pro = st.container()

def main():
    with gemini_pro:
        st.header("Interact with Gemini Pro")
        st.write("")

        # 사용자로부터 prompt를 입력받는 텍스트 상자
        prompt = st.text_input("prompt please...", placeholder="Prompt", label_visibility="visible")

        # 사용자가 SEND 버튼을 눌렀을 때의 동작
        if st.button("SEND", use_container_width=True):
            # 입력된 prompt 뒤에 "할 때 먹는 음식"을 추가"
            # Gemini Pro 모델 생성
            model = genai.GenerativeModel("gemini-pro")

            # 추가된 텍스트를 포함한 prompt로 content를 생성하고 응답을 받음
            response = model.generate_content(prompt)

            # 응답을 출력
            st.write("")
            st.header(":blue[Response]")
            st.write("")

            st.markdown(response.text)
if __name__ == "__main__":
    main()
    

