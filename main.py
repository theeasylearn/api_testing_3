from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI(title="Watermark Removal API")

# Load model (done once at startup)
print("Loading U-Net model...")
model = tf.keras.models.load_model("watermark_removal_model_final.h5",compile=False)
print("✅ Model loaded successfully!")

IMG_SIZE = 256

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array, img

def postprocess_image(input_array, pred):
    pred = pred[0]  # Remove batch dimension
    
    # === MOST COMMON FIX FOR BLANK IMAGE ===
    # If model predicts watermark, subtract it from original
    cleaned = input_array[0] - pred          # Subtract predicted watermark
    cleaned = np.clip(cleaned, 0, 1)         # Keep values between 0 and 1
    cleaned = (cleaned * 255).astype(np.uint8)
    
    return Image.fromarray(cleaned)

@app.post("/remove-watermark")
async def remove_watermark(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        input_array, original_pil = preprocess_image(image_bytes)

        # Inference
        prediction = model.predict(input_array, verbose=0)

        cleaned_pil = postprocess_image(input_array, prediction)

        # Return both images as base64 or raw
        output_buffer = io.BytesIO()
        cleaned_pil.save(output_buffer, format="PNG")
        output_buffer.seek(0)

        return StreamingResponse(output_buffer, media_type="image/png")

    except Exception as e:
        raise HTTPException(500, detail=str(e))

# Optional: Health check
@app.get("/health")
def health():
    return {"status": "healthy", "model": "Watermark Removal U-Net"}