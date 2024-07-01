import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import av
import cv2
import streamlit_webrtc as webrtc
import pickle
import moviepy.editor as mp

# Load your model
model = load_model('my_model.keras')


with open('content_dict.pickle', 'rb') as f:
    loaded_dict = pickle.load(f)

# Function to preprocess image
def preprocess_image(img, target_size):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  
    return img_array

# Extract frames from video
def extract_frames(video_path, target_size):
    clips = []
    clip = mp.VideoFileClip(video_path)
    for frame in clip.iter_frames():
        img = Image.fromarray(frame)
        img = img.resize(target_size)
        clips.append(img)
    return clips

# Predict image function
def predict_image(model, img, target_size):
    preprocessed_image = preprocess_image(img, target_size)
    predictions = model.predict(preprocessed_image)
    return predictions

# Capture video from webcam
def get_video_capture():
    return webrtc.get_video_capture()

# class VideoTransformer(webrtc.Streamer):
#     def transform(self, frame):
#         img = Image.fromarray(cv2.cvtColor(frame.to_ndarray(), cv2.COLOR_RGB2BGR))
#         target_size = (200, 200)
#         img = img.resize(target_size)
#         predictions = predict_image(model, img, target_size)
#         predicted_class = np.argmax(predictions, axis=1)[0]
#         return img, f'Prediction: {predicted_class}'


# Main function for Streamlit app
def main():
    st.title('Sign To Voice Communication')
    
    # Option to choose input source
    input_type = st.selectbox(
        "Select Input Type",
        ("Image", "Video", "Webcam")
    )
    
    if input_type == "Image":
        # Image uploader widget
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"], key="image")
        if uploaded_file is not None:
            # Convert the uploaded file to a PIL Image
            pil_image = Image.open(io.BytesIO(uploaded_file.read()))
            target_size = (200, 200)
            predictions = predict_image(model, pil_image, target_size)
            predicted_class = np.argmax(predictions, axis=1)[0]
            st.text(f'Predicted class: {loaded_dict[predicted_class]}')

    elif input_type == "Video":
        # Video uploader widget
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi"], key="video")
        if uploaded_video is not None:
            st.video(uploaded_video, caption='Uploaded Video')
        
        # Convert the uploaded video to a list of PIL Images
        target_size = (200, 200)
        frames = extract_frames(uploaded_video.name, target_size)
        
        # Process each frame
        for i, frame in enumerate(frames):
            # Predict the frame
            predictions = predict_image(model, frame, target_size)
            
            # Get the predicted class
            predicted_class = np.argmax(predictions, axis=1)[0]
            
            # Optionally, display the prediction for each frame
            st.text(f'Frame {i+1} Prediction: {loaded_dict[predicted_class]}')
            # Or save the predictions to a file or database for further analysis

    elif input_type == "Webcam":
        # Capture video from webcam
        video_capture = get_video_capture()
        if video_capture is not None:
            # Display the video feed
            st.video(video_capture, caption='Webcam Feed')
            
            # Process each frame from the webcam feed
            for frame in video_capture:
                # Convert the frame to a PIL Image
                img = Image.fromarray(cv2.cvtColor(frame.to_ndarray(), cv2.COLOR_RGB2BGR))
                
                # Resize the image to the target size
                target_size = (200, 200)
                img = img.resize(target_size)
                
                # Predict the image
                predictions = predict_image(model, img, target_size)
                
                # Get the predicted class
                predicted_class = np.argmax(predictions, axis=1)[0]
                
                # Optionally, display the prediction for each frame
                st.text(f'Prediction: {loaded_dict[predicted_class]}')
                # Or save the predictions to a file or database for further analysis

# Run the main function
if __name__ == "__main__":
    main()
