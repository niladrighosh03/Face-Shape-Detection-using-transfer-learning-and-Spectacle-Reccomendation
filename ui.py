import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import mediapipe as mp
import os
import time

from predict import predict_face_shape, load_model  # Assuming you have a function to predict face shape

BASE_PATH = r"C:\Users\nilad\Documents\SEM 2\Computer Vision\draft cv project\Hey\Sunglass"
shape_to_folder = {
    "Heart": "Heart",
    "Oblong": "Oblong",
    "Oval": "Oval",
    "Round": "Round",
    "Square": "Square",
}

# Helper function to capture image
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam")
        return None
        
    st.info('Press "c" to capture your image')

    captured_image = None

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error('Failed to grab frame')
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow('Capture Your Face (Press c)', frame)

        key = cv2.waitKey(1)
        if key == ord('c'):
            captured_image = frame.copy()
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_image

# Load the model at the start to avoid reloading
@st.cache_resource
def get_model():
    return load_model(r"C:\Users\nilad\Documents\SEM 2\Computer Vision\draft cv project\ui\vgg16_face_transfer_final.pth", strict=False)

# Helper function to predict face shape
def predict_shape(image):
    img = image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for processing

    # Predict face shape
    model, device = get_model()
    shape, confidence, processed_img = predict_face_shape(img_rgb, model, device)
    return shape

# Function to rotate an image
def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

# Load sunglasses based on face shape
def load_sunglasses(face_shape):
    folder = shape_to_folder.get(face_shape, None)
    if not folder:
        return []
    folder_path = os.path.join(BASE_PATH, folder)
    if not os.path.exists(folder_path):
        st.error(f"Directory not found: {folder_path}")
        return []
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.png')]

# Streamlit UI
st.set_page_config(page_title="Face Shape Detector & Sunglass Recommender", page_icon="😎", layout="centered")

st.title("😎 Face Shape Detection and Sunglass Recommendation")

# Initialize session state variables if they don't exist
if 'face_shape' not in st.session_state:
    st.session_state.face_shape = None
if 'captured_img' not in st.session_state:
    st.session_state.captured_img = None
if 'selected_glass' not in st.session_state:
    st.session_state.selected_glass = None
if 'try_on_active' not in st.session_state:
    st.session_state.try_on_active = False

# Step 1: Capture Image
if st.button('📸 Capture Face'):
    captured_img = capture_image()

    if captured_img is not None:
        st.session_state.captured_img = captured_img
        
        # Step 2: Predict Face Shape
        try:
            face_shape = predict_shape(captured_img)
            st.session_state.face_shape = face_shape
        except Exception as e:
            st.error(f"Error predicting face shape: {e}")
            st.session_state.face_shape = None

# Display captured image and face shape
if st.session_state.captured_img is not None:
    st.success("Image Captured!")
    st.image(st.session_state.captured_img, channels="BGR", caption="Captured Face")
    
    if st.session_state.face_shape:
        st.subheader(f"🧠 Detected Face Shape: {st.session_state.face_shape.capitalize()}")
    else:
        st.warning("Could not detect face shape. Please try again.")

# Step 3: Recommend Sunglasses based on detected face shape
if st.session_state.face_shape:
    # Step 3: Recommend Sunglasses
    sunglass_paths = load_sunglasses(st.session_state.face_shape)
    
    if not sunglass_paths:
        st.warning(f"No sunglasses found for {st.session_state.face_shape} face shape")
    else:
        # Show Sunglasses
        st.sidebar.write(f"### Recommended Sunglasses for {st.session_state.face_shape} Face:")
        cols = st.sidebar.columns(2)

        for idx, glass_path in enumerate(sunglass_paths):
            with cols[idx % 2]:
                if st.button(f"Select {os.path.basename(glass_path)}", key=glass_path):
                    st.session_state.selected_glass = glass_path

        # Show Selected Sunglass
        if st.session_state.selected_glass:
            st.sidebar.image(st.session_state.selected_glass, caption="Selected Sunglass", width=150)
            
            # Option to try on the selected sunglasses
            if st.sidebar.button("Try On Sunglasses"):
                st.session_state.try_on_active = True
            
            if st.sidebar.button("Stop Try On"):
                st.session_state.try_on_active = False
        else:
            st.sidebar.info("Please select a sunglass!")

# Try on sunglasses using webcam
if st.session_state.try_on_active and st.session_state.selected_glass:
    st.subheader("Virtual Try-On")
    
    # Initialize webcam placeholder
    frame_placeholder = st.empty()
    
    # Setup MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1)
    
    # Load sunglass image
    try:
        sunglass_png = cv2.imread(st.session_state.selected_glass, cv2.IMREAD_UNCHANGED)
        if sunglass_png is None:
            st.error(f"Failed to load the sunglass image: {st.session_state.selected_glass}")
            st.session_state.try_on_active = False
    except Exception as e:
        st.error(f"Error loading sunglass image: {e}")
        st.session_state.try_on_active = False
        
    # Start webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam for try-on")
        st.session_state.try_on_active = False
    
    # Try-on loop
    try:
        while st.session_state.try_on_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame from webcam")
                break
                
            frame = cv2.flip(frame, 1)  # Mirror image
            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process face landmarks
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks and sunglass_png is not None:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Get eye landmarks (left and right outer corners)
                lx, ly = int(landmarks[33].x * w), int(landmarks[33].y * h)
                rx, ry = int(landmarks[263].x * w), int(landmarks[263].y * h)
                
                # Calculate center point and angle between eyes
                eye_center = ((lx + rx) // 2, (ly + ry) // 2)
                angle = np.degrees(np.arctan2(ry - ly, rx - lx))
                
                # Calculate sunglass size based on eye distance
                eye_width = int(1.9 * abs(rx - lx))
                aspect_ratio = sunglass_png.shape[0] / sunglass_png.shape[1]
                glass_height = int(eye_width * aspect_ratio)
                
                # Resize sunglass image
                try:
                    resized_glass = cv2.resize(sunglass_png, (eye_width, glass_height), interpolation=cv2.INTER_AREA)
                except Exception as e:
                    st.error(f"Error resizing sunglass: {e}")
                    continue
                
                # Rotate sunglass to match face angle
                rotated_glass = rotate_image(resized_glass, -angle)
                
                # Position sunglass on face
                y_offset = int(eye_center[1] - rotated_glass.shape[0] * 0.5)  # Adjust vertical position
                x1 = eye_center[0] - rotated_glass.shape[1] // 2
                y1 = y_offset
                
                # Overlay sunglass on face with alpha blending
                for i in range(rotated_glass.shape[0]):
                    for j in range(rotated_glass.shape[1]):
                        if y1 + i >= h or x1 + j >= w or y1 + i < 0 or x1 + j < 0:
                            continue
                        
                        alpha = rotated_glass[i, j, 3] / 255.0
                        if alpha > 0:
                            for c in range(3):
                                frame[y1 + i, x1 + j, c] = (
                                    alpha * rotated_glass[i, j, c] +
                                    (1 - alpha) * frame[y1 + i, x1 + j, c]
                                )
            
            # Display the frame
            frame_placeholder.image(frame, channels="BGR", caption="Virtual Try-On")
            
            # Add a small delay to reduce CPU usage
            time.sleep(0.01)
    
    except Exception as e:
        st.error(f"Try-on error: {e}")
    
    finally:
        # Clean up resources
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'face_mesh' in locals():
            face_mesh.close()
        st.session_state.try_on_active = False
else:
    st.write("Click 'Capture Face' to start detecting your face shape and getting sunglass recommendations!")