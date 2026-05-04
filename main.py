from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import asyncio

app = FastAPI(title="Watermark Removal - Final Debug")

tf.config.set_visible_devices([], 'GPU')

print("Loading U-Net model...")
model = tf.keras.models.load_model("watermark_removal_model_final.h5", compile=False)
print("✅ Model loaded successfully!")

IMG_SIZE = 256

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array, img

@app.post("/remove-watermark")
async def remove_watermark(file: UploadFile = File(...)):
    return await process(file, mode="normal")

@app.post("/debug")
async def debug(file: UploadFile = File(...)):
    return await process(file, mode="debug")

async def process(file: UploadFile, mode: str = "normal"):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image")
    
    try:
        image_bytes = await file.read()
        input_array, original_pil = preprocess_image(image_bytes)
        
        prediction = model.predict(input_array, verbose=0)
        
        # === VERY IMPORTANT LOGS ===
        print(f"🔍 INPUT  min/max: {input_array.min():.4f} / {input_array.max():.4f}")
        print(f"🔍 OUTPUT min/max: {prediction.min():.4f} / {prediction.max():.4f}")
        print(f"🔍 OUTPUT shape: {prediction.shape}")

        if mode == "debug":
            # Save raw model output as grayscale
            raw_pred = (prediction[0, :, :, 0] * 255).clip(0, 255).astype(np.uint8)
            raw_pil = Image.fromarray(raw_pred, mode='L')
            
            combined = Image.new('RGB', (IMG_SIZE * 2, IMG_SIZE))
            combined.paste(original_pil, (0, 0))
            combined.paste(raw_pil.convert("RGB"), (IMG_SIZE, 0))
            
            buf = io.BytesIO()
            combined.save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")
        
        # Normal mode - try multiple post-processing
        pred = prediction[0]
        
        # Try 3 different methods and return the best looking one
        cleaned = (input_array[0] - pred) * 255
        cleaned = cleaned.clip(0, 255).astype(np.uint8)
        
        cleaned_pil = Image.fromarray(cleaned)
        
        buf = io.BytesIO()
        cleaned_pil.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        print("ERROR:", str(e))
        raise HTTPException(500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}