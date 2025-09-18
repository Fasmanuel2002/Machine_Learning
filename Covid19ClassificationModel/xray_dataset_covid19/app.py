from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from PIL import Image
from tensorflow.keras.preprocessing import image # type: ignore
import io
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

MODEL_IMAGE_PATH = "CovidDetector.h5"
model = load_model(MODEL_IMAGE_PATH)


@app.post("/CovidPredictor")
async def uploadCovidPhoto(file : UploadFile = File(...)):
    data = await file.read()

    img = Image.open(io.BytesIO(data)).convert("RGB")
    reshapedImage = cv2.resize(img, (64,64))
    imageArray = image.img_to_array(reshapedImage)
    imageArray = np.expand_dims(imageArray, axis=0)
    img_array = imageArray / 255
    prediction = model.predict(img_array)[0]
     
    return prediction

