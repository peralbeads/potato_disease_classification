import numpy as np
from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image  # used to  read image in python
import tensorflow as tf
import requests

app = FastAPI()

endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"

MODEL = tf.keras.models.load_model("../models/1.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)

):

    image = read_file_as_image(await file.read())   # convert the file in to numpy array
    image_batch = np.expand_dims(image, 0)  # changing 1d array dimensions to 2d array so that our predict function

    # does not have any trouble as it only accepts 2d array, or you can say batch of images
    json_data = {
        "instances": image_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions[0]"])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    pass

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)