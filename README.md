# 🍅 Tomato Leaf Disease Detection using Deep Learning

A FastAPI-based web application that detects tomato leaf diseases using a trained TensorFlow/Keras deep learning model. Users can upload an image of a tomato leaf through a web interface, and the application predicts the disease class along with the confidence score.

---

## Features

- Upload tomato leaf images through a web interface.
- Predicts one of the supported tomato leaf diseases.
- Displays the uploaded image along with the prediction result.
- Shows prediction confidence.
- FastAPI backend for efficient inference.
- TensorFlow/Keras model for disease classification.

---

## Project Structure

```
Tomato/
│
├── API/
│   ├── main.py
│   ├── 3.h5
│   ├── requirements.txt
│   ├── static/
│   │   └── uploads/
│   └── templates/        (Optional if using Jinja2)
│
├── Train/
│   ├── Train.ipynb
│   └── dataset/
│
├── README.md
└── .gitignore
```

---

## Supported Disease Classes

The model can classify the following tomato leaf conditions:

- Tomato Bacterial Spot
- Tomato Early Blight
- Tomato Healthy
- Tomato Late Blight
- Tomato Leaf Mold
- Tomato Septoria Leaf Spot
- Tomato Spider Mites (Two-Spotted Spider Mite)
- Tomato Target Spot
- Tomato Tomato Mosaic Virus
- Tomato Tomato Yellow Leaf Curl Virus

---

## Dataset

This project uses the **Tomato Leaf Disease Dataset (Segmented)** available on Kaggle.

Dataset Link: https://www.kaggle.com/datasets/ahmadzargar/tomato-leaf-disease-dataset-segmented

Download the dataset and place it inside the **Train/dataset/** directory before training.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/Tomato-Leaf-Disease-Detection.git

cd Tomato-Leaf-Disease-Detection/API
```

---

### Create a Virtual Environment

**Windows**

```bash
python -m venv venv
```

Activate it

```bash
venv\Scripts\activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Model

Place the trained TensorFlow model inside the API folder.

```
API/
│
├── 3.h5
```

If your model has a different name, update this line inside `main.py`.

```python
MODEL = tf.keras.models.load_model("3.h5")
```

---

## Running the Application

Start the FastAPI server.

```bash
python main.py
```

or

```bash
uvicorn main:app --reload
```

The application will be available at

```
http://localhost:8008
```

---

## How to Use

1. Open the application in your browser.
2. Upload a tomato leaf image.
3. Click **Upload & Predict**.
4. View the uploaded image.
5. Check the predicted disease and confidence score.

---

## Technologies Used

- Python
- FastAPI
- TensorFlow
- NumPy
- Pillow (PIL)
- Uvicorn
- HTML
- CSS

---

## Requirements

Install all required packages using

```bash
pip install -r requirements.txt
```

---

## Example Output

- Uploaded Tomato Leaf Image
- Predicted Disease
- Prediction Confidence

Example:

```
Prediction:
Tomato Late Blight

Confidence:
98.63%
```

---

## Future Improvements

- Deploy the application on Render or Railway.
- Add disease treatment recommendations.
- Improve user interface using Bootstrap.
- Support multiple image uploads.
- Add prediction history.
- Integrate a database for storing prediction records.

---
