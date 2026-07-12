from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import shutil
import os
app = FastAPI()
MODEL = tf.keras.models.load_model(r"D:\Tomato\API\3.h5")
CLASS_NAMES = [
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data))
        image = image.convert("RGB")  
        image = image.resize((256, 256))  
        return np.array(image)
    except Exception as e:
        print(f"Error reading image: {e}")
        raise ValueError("Invalid image file")
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Tomato Leaf Disease Detection</title>
        <style>
            body { font-family: 'Times New Roman', serif; text-align: center; padding: 20px; background-color: #f4f4f4; color: #333; }
            h1 { color: #2c3e50; }
            form { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2); display: inline-block; }
            input[type="file"] { padding: 10px; border: 1px solid #ddd; border-radius: 5px; background: #fff; }
            button { padding: 10px 15px; margin-top: 10px; background-color: #27ae60; color: #fff; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background-color: #219150; }
            img { width: 250px; margin-top: 15px; border-radius: 5px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2); }
            h2 { color: #34495e; }
        </style>
    </head>
    <body>
        <h1>Tomato Leaf Disease Detection</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Upload & Predict</button>
        </form>
    </body>
    </html>
    """
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        file_path = f"{UPLOAD_DIR}/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        with open(file_path, "rb") as image_file:
            image = read_file_as_image(image_file.read())
        img_batch = np.expand_dims(image, 0)  
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100  
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Tomato Leaf Disease Detection</title>
            <style>
                body {{ font-family: 'Times New Roman', serif; text-align: center; padding: 20px; background-color: #f4f4f4; color: #333; }}
                h1 {{ color: #2c3e50; }}
                img {{ width: 250px; margin-top: 15px; border-radius: 5px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2); }}
                h2 {{ color: #34495e; }}
            </style>
        </head>
        <body>
            <h1>Tomato Leaf Disease Detection</h1>
            <h2>Uploaded Image:</h2>
            <img src="/static/uploads/{file.filename}" alt="Uploaded Image">
            <h2>Prediction: {predicted_class} (Confidence: {confidence:.2f}%)</h2>
            <br>
            <a href="/">Upload Another Image</a>
        </body>
        </html>
        """
    except Exception as e:
        print(f"Prediction error: {e}")
        return HTMLResponse(content=f"<h2>Error: {str(e)}</h2>", status_code=500)
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8008)
