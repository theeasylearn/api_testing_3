from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import asyncio

app = FastAPI(title="Watermark Removal API")

# Force CPU and limit memory
tf.config.set_visible_devices([], 'GPU')

print("Loading U-Net model...")
model = tf.keras.models.load_model(
    "watermark_removal_model_final.h5",
    compile=False
)
print("✅ Model loaded successfully!")

IMG_SIZE = 256   # Must stay 256 to match model

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array, img

def postprocess_image(input_array, pred):
    pred = pred[0]
    # Subtraction method
    cleaned = input_array[0] - pred
    cleaned = np.clip(cleaned, 0.0, 1.0)
    cleaned = (cleaned * 255).astype(np.uint8)
    return Image.fromarray(cleaned)

@app.post("/remove-watermark")
async def remove_watermark(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image")
    
    try:
        image_bytes = await file.read()
        
        async def run_inference():
            input_array, _ = preprocess_image(image_bytes)
            prediction = model.predict(input_array, verbose=0)
            
            print(f"Prediction min/max: {prediction.min():.4f} / {prediction.max():.4f}")
            print(f"Input min/max: {input_array.min():.4f} / {input_array.max():.4f}")
            
            cleaned_pil = postprocess_image(input_array, prediction)
            return cleaned_pil
        
        cleaned_pil = await asyncio.wait_for(run_inference(), timeout=25.0)
        
        output_buffer = io.BytesIO()
        cleaned_pil.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        
        return StreamingResponse(output_buffer, media_type="image/png")
    
    except asyncio.TimeoutError:
        raise HTTPException(504, detail="Processing took too long. Try smaller image.")
    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}