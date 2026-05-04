from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import asyncio

app = FastAPI(title="Watermark Removal API - Debug Mode")

# Force CPU
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
    return await process_image(file, debug=False)

@app.post("/debug")
async def debug_watermark(file: UploadFile = File(...)):
    """Returns side-by-side: Original | Raw Model Output"""
    return await process_image(file, debug=True)

async def process_image(file: UploadFile, debug: bool = False):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image")
    
    try:
        image_bytes = await file.read()
        
        async def run_inference():
            input_array, original_pil = preprocess_image(image_bytes)
            prediction = model.predict(input_array, verbose=0)
            
            # CRITICAL DEBUG INFO
            print(f"🔍 Prediction min/max: {prediction.min():.4f} / {prediction.max():.4f}")
            print(f"🔍 Input min/max: {input_array.min():.4f} / {input_array.max():.4f}")
            
            return input_array, prediction, original_pil
        
        input_array, prediction, original_pil = await asyncio.wait_for(run_inference(), timeout=25.0)
        
        # Debug mode: return side-by-side images
        if debug:
            pred_img = (prediction[0] * 255).clip(0, 255).astype(np.uint8)
            if pred_img.shape[-1] == 1 or len(pred_img.shape) == 2:
                pred_img = np.squeeze(pred_img)
                pred_pil = Image.fromarray(pred_img, mode='L')
            else:
                pred_pil = Image.fromarray(pred_img)
            
            # Create side-by-side image
            combined = Image.new('RGB', (IMG_SIZE*2, IMG_SIZE))
            combined.paste(original_pil, (0, 0))
            combined.paste(pred_pil.convert("RGB"), (IMG_SIZE, 0))
            
            output_buffer = io.BytesIO()
            combined.save(output_buffer, format="PNG")
            output_buffer.seek(0)
            return StreamingResponse(output_buffer, media_type="image/png")
        
        # Normal mode - try subtraction again
        cleaned = input_array[0] - prediction[0]
        cleaned = np.clip(cleaned, 0.0, 1.0)
        cleaned = (cleaned * 255).astype(np.uint8)
        cleaned_pil = Image.fromarray(cleaned)
        
        output_buffer = io.BytesIO()
        cleaned_pil.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        return StreamingResponse(output_buffer, media_type="image/png")
    
    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}