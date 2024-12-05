#!/usr/bin/env python
# coding: utf-8

# In[110]:


from tensorflow.keras.models import load_model
import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from datetime import datetime

loaded_model = tf.keras.models.load_model(r"C:\Users\Asus\Downloads\Projet\emotion_model93.keras")


# In[111]:


def predict_emotion(image):
    # Convert the PIL Image to a numpy array
    image = np.array(image)

    # Check if the image is not grayscale
    if len(image.shape) > 2 and image.shape[2] == 3:
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize the image to 48x48
    image = cv2.resize(image, (48, 48))

    # Normalize the pixel values
    image = image / 255.0

    # Add an extra dimension for the batch size
    image = np.expand_dims(image, axis=0)

    # Add an extra dimension for the color channel
    image = np.expand_dims(image, axis=-1)

    # Use the model to predict the emotion
    prediction = loaded_model.predict(image)

    # The prediction is an array of probabilities for each class
    # Use argmax to get the class with the highest probability
    emotion_class = np.argmax(prediction)

    # Map the class to the corresponding emotion
    # This depends on how your model is trained
    emotion_classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    emotion = emotion_classes[emotion_class]

    #return emotion
    return prediction[0]


# In[112]:


# Load the pre-trained cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_facei(image):
    # Convert the uploaded image to a NumPy array
    np_image = np.array(image)

    # Convert the image to grayscale if it's not already in grayscale
    if len(np_image.shape) > 2:
        gray_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = np_image

    # Perform face detection on the grayscale image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    # Initialize extracted_face with None
    extracted_face = None

    # Iterate over detected faces and extract them from the original image
    for (x, y, w, h) in faces:
        extracted_face = np_image[y:y+h, x:x+w]
        # You can further process or analyze the extracted face here
        break  # Stop after extracting the first detected face

    return extracted_face

 


# In[113]:


# Function to preprocess your images
def preprocess(image):
    preprocessed_frame=extract_face(image)
    preprocessed_frame = cv2.resize(preprocessed_frame, (48, 48))
    preprocessed_frame = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2GRAY)
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=(0, -1))
    return preprocessed_frame


# In[114]:


import cv2

def extract_face(image):
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5) #badelt gray b image

    # If a face is detected, return the first detected face
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = image[y:y+h, x:x+w]
        return face
    else:
        return None


# In[115]:


def camera():
    # Define the emotion labels
    emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

    # Start video capture
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break
    
        # Extract face
        extracted_face = extract_face(frame)
    
        # Check if a face is detected
        if extracted_face is not None:
            # Preprocess the frame
            preprocessed_frame = cv2.resize(extracted_face, (48, 48))
            preprocessed_frame = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2GRAY)
            preprocessed_frame = np.expand_dims(preprocessed_frame, axis=(0, -1))

            # Make predictions
            emotion_predictions = loaded_model.predict(preprocessed_frame)[0]

            # Get the top two predicted emotion labels and their probabilities
            top_indices = emotion_predictions.argsort()[-2:][::-1]
            top_emotions = [emotion_labels[i] for i in top_indices]
            top_probabilities = [emotion_predictions[i] for i in top_indices]

            # Define text position dynamically based on frame size
            text_x = 10
            text_y = frame.shape[0] - 20
            line_height = 30

            # Display the top two predicted emotion labels and their probabilities on separate lines
            for i, (emotion, prob) in enumerate(zip(top_emotions, top_probabilities), start=1):
                text = f'{i}st Prediction: {emotion} (Accuracy: {prob:.2f})'
                cv2.putText(frame, text, (text_x, text_y - i * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Emotion Recognition', frame)

        # Take screenshot on 's' key press
        key = cv2.waitKey(1)
        if key == ord('s'):
            # Define the filename with timestamp
            filename = f'screenshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    
            # Specify the directory path for saving the screenshots
            save_dir = r"D:\Desktop\screenshots"
    
            # Save the frame as a screenshot with full path
            cv2.imwrite(save_dir + '\\' + filename, frame)  # Add directory separator
            print(f'Screenshot saved as {filename}')

        # Break the loop on 'q' key press
        if key == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


# In[117]:


import streamlit as st


# Page 1: Login Page
def login_page():
    st.title("Login Page")
    
    # Add login form inputs (e.g., username, password)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    # Login button
    if st.button("Login"):
        # Check login credentials and redirect to the appropriate page
        if username == "firas" and password == "123":
            st.success("Login Successful!")
            # Redirect to the predictions page
            navigate("Image Selection")
        else:
            st.error("Invalid Username or Password")
    
    # Inscription button
    if st.button("Inscription"):
        # Handle inscription process here, like redirecting to inscription page
        navigate("inscription_page")


def inscription_page():
    import streamlit as st
    
    st.title("Inscription Page")
    
    # Add inscription form inputs (e.g., username, password, email)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    email = st.text_input("Email")
    
    # Inscription button
    if st.button("Register"):
        # Handle registration process here, like storing user details in a database
        st.success("Registration Successful!")
        # Optionally, redirect the user to another page after registration
        # navigate("Another Page")



# Page 2: Image Selection and Emotion Prediction
def image_selection_page():
    st.title('Emotion Prediction App')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
    
        if st.button('Predict'):
            st.write("Classifying...")
            # Predict the emotion
            probabilities = predict_emotion(image)
        
            # Reshape probabilities to a 2D array
            probabilities = probabilities.reshape(1, -1)
        
            # Display the prediction
            emotion_classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
            predicted_emotion = emotion_classes[np.argmax(probabilities)]
            st.write(f'Predicted Emotion: {predicted_emotion}')

            # Create a DataFrame for the probabilities
            df = pd.DataFrame(probabilities, columns=emotion_classes)

            # Display a bar chart of the probabilities
            st.bar_chart(df)

        
        
        
      
# Page 3: Live Camera Emotion Prediction

def live_camera_page():
    st.title("Live Camera Emotion Prediction")
    
    camera()
    # Add live camera feed and display emotion prediction
    # Replace this with your live camera and emotion prediction logic
    st.write("Live Camera Feed")
    st.write("Emotion: Sad")

# Navigation function
def navigate(page):
    if page == "Login":
        login_page()
    elif page == "Image Selection":
        image_selection_page()
    elif page == "Live Camera":
        live_camera_page()
    elif page =="inscription_page":
        inscription_page()

# Sidebar navigation
page_options = ["Login","inscription_page", "Image Selection", "Live Camera"]
selected_page = st.sidebar.radio("Navigation", page_options)

# Display the selected page
navigate(selected_page)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




