from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import asyncio

app = FastAPI(title="Watermark Removal API")

# Load model (done once at startup)
print("Loading U-Net model...")
model = tf.keras.models.load_model(
    "watermark_removal_model_final.h5",
    compile=False   # Important for compatibility
)
print("✅ Model loaded successfully!")

IMG_SIZE = 256   # You can reduce to 192 or 128 if still slow

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array, img

def postprocess_image(input_array, pred):
    """Improved post-processing - Try subtraction first"""
    pred = pred[0]  # Remove batch dimension
    
    # Option 1: Subtract predicted watermark (most common fix for blank images)
    cleaned = input_array[0] - pred
    cleaned = np.clip(cleaned, 0.0, 1.0)
    cleaned = (cleaned * 255).astype(np.uint8)
    
    # Alternative (if subtraction gives bad results): Direct prediction
    # cleaned = (pred * 255).clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(cleaned)

@app.post("/remove-watermark")
async def remove_watermark(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image")
    
    try:
        image_bytes = await file.read()
        
        # Run inference with timeout
        async def run_inference():
            input_array, _ = preprocess_image(image_bytes)
            prediction = model.predict(input_array, verbose=0)
            
            # Debug info (check Render Logs)
            print(f"Prediction min/max: {prediction.min():.4f} / {prediction.max():.4f}")
            print(f"Input min/max: {input_array.min():.4f} / {input_array.max():.4f}")
            
            cleaned_pil = postprocess_image(input_array, prediction)
            return cleaned_pil
        
        # 20 seconds timeout (adjust if needed)
        cleaned_pil = await asyncio.wait_for(run_inference(), timeout=20.0)
        
        output_buffer = io.BytesIO()
        cleaned_pil.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        
        return StreamingResponse(output_buffer, media_type="image/png")
    
    except asyncio.TimeoutError:
        raise HTTPException(504, detail="Processing took too long. Try a smaller image.")
    except Exception as e:
        print("Error processing image:", str(e))
        raise HTTPException(500, detail=f"Failed to process image: {str(e)}")

@app.get("/health")
def health():
    return {"status": "healthy", "model": "Watermark Removal U-Net"}