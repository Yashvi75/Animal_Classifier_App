Animal Image Classifier

A deep learning web app that classifies images of animals into 15 different categories using Transfer Learning (MobileNetV2). Upload a photo of an animal, and the model will tell you which animal it is. 

What It Does?

- Accepts image uploads via a simple web interface
- Preprocesses the image (resize, normalize)
- Uses a trained MobileNetV2 model to classify the image
- Displays the predicted animal class with a clean UI


How It Works

This project uses **Transfer Learning** with the **MobileNetV2** architecture, fine-tuned on a custom dataset of 15 animal categories. It was trained using TensorFlow and deployed using Streamlit Cloud for easy accessibility.

Supported Animal Classes

- Bear
- Bird
- Cat
- Cow
- Deer
- Dog
- Dolphin
- Elephant
- Giraffe
- Horse
- Kangaroo
- Lion
- Panda
- Tiger
- Zebra

Try It Out

Visit the live app here (once deployed):  
**[🔗 Your-App-Link-Here](https://your-username-animal-classifier-app.streamlit.app/)**



Built With

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow (PIL)

Files Included

- `app.py` - Streamlit app source code
- `animal_classifier_model.keras` - Trained model file
- `requirements.txt` - Python dependencies
- `README.md` - This file
