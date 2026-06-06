import os
import base64
import numpy as np
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except Exception:
    load_model = None
    TENSORFLOW_AVAILABLE = False
import logging
import traceback
from utils.helpers import EMOTIONS, preprocess_face, EMOTION_COLORS
import ctypes

# Initialize FastAPI App
app = FastAPI(title="Live Face Emotion Recognition Server")

# Define request schema
class FrameData(BaseModel):
    image: str  # Base64 string from browser canvas

# Global variables for model and face detector
model = None
face_cascade = None

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

# Mount static folder
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates setting
templates = Jinja2Templates(directory=templates_dir)

@app.on_event("startup")
def startup_event():
    global model, face_cascade
    
    # 1. Load trained Keras model
    model = None
    model_path = os.path.join(BASE_DIR, "models", "best_model.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, "models", "best_model.h5")

    if TENSORFLOW_AVAILABLE and os.path.exists(model_path):
        try:
            print(f"Loading trained emotion classifier model from: {model_path}")
            model = load_model(model_path)
        except Exception as e:
            print(f"[WARNING] Failed to load model: {e}\nStarting in DEMO mode.")
            model = None
    else:
        print("[WARNING] TensorFlow or model not available. Starting server in DEMO/MOCK mode.")
        model = None
        
    # 2. Load Haar Cascade Face Detector (optional)
    if CV2_AVAILABLE:
        def get_short_path(p):
            try:
                buf = ctypes.create_unicode_buffer(260)
                res = ctypes.windll.kernel32.GetShortPathNameW(p, buf, 260)
                if res > 0:
                    return buf.value
            except Exception:
                pass
            return p

        # Priority 1: use the local XML bundled in the project root (avoids non-ASCII path issues)
        local_cascade = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
        if os.path.exists(local_cascade):
            cascade_path = local_cascade
        else:
            cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

        # Try loading with short (8.3) path to avoid Unicode issues on Windows
        short_path = get_short_path(cascade_path)
        face_cascade = cv2.CascadeClassifier(short_path)

        if face_cascade.empty():
            # Try without short-path conversion as a safety net
            face_cascade = cv2.CascadeClassifier(cascade_path)

        if face_cascade.empty():
            print("[ERROR] Failed to load Haar Cascade xml file. Running in DEMO mode (no face detection).")
            face_cascade = None
        else:
            print(f"[INFO] Haar Cascade loaded from: {short_path}")
    else:
        print("[WARNING] OpenCV (cv2) not available — running in DEMO mode (no face detection).")
        face_cascade = None

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Render and serve the live camera web application interface."""
    return templates.TemplateResponse(request, "index.html", {"emotions": EMOTIONS})

@app.post("/predict")
async def predict_emotion(data: FrameData):
    """
    Inference endpoint:
    - Decodes Base64 encoded web frame
    - Detects faces
    - Normalizes face region of interest (ROI)
    - Predicts emotion using Keras CNN
    - Returns JSON results
    """
    # If OpenCV isn't available or cascade isn't initialized, return demo/no-face response
    if not CV2_AVAILABLE or face_cascade is None:
        # Respond with no faces detected so client stays functional in demo mode
        return {"faces": [], "count": 0}
        
    try:
        # 1. Parse base64 image data URI (e.g. "data:image/jpeg;base64,...")
        header, encoded = data.image.split(",", 1) if "," in data.image else ("", data.image)
        decoded_bytes = base64.b64decode(encoded)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(decoded_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image payload received.")
            
        # Convert to Grayscale for face detection and CNN input
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Face Detection
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        results = []
        
        # 3. Model Inference on detected face coordinates
        for (x, y, w, h) in faces:
            # Crop Grayscale Face ROI
            face_roi_gray = gray_frame[y:y+h, x:x+w]
            
            # Predict
            if model is not None:
                # Preprocess cropped face to matching model input dimensions (1, 48, 48, 1)
                preprocessed = preprocess_face(face_roi_gray)
                
                if preprocessed is not None:
                    preds = model.predict(preprocessed)[0]
                    max_idx = np.argmax(preds)
                    emotion_label = EMOTIONS[max_idx]
                    confidence = float(preds[max_idx])
                    
                    # Complete list of emotion probabilities for graphs
                    probabilities = {EMOTIONS[i]: float(preds[i]) for i in range(len(EMOTIONS))}
                else:
                    emotion_label = "Error"
                    confidence = 0.0
                    probabilities = {}
            else:
                # Demo fallback simulation mode if model has not been trained yet
                emotion_label = "Neutral (Demo)"
                confidence = 1.0
                probabilities = {e: 0.14 for e in EMOTIONS}
                probabilities["Neutral"] = 1.0
                
            results.append({
                "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "emotion": emotion_label,
                "confidence": confidence,
                "color": EMOTION_COLORS.get(emotion_label.replace(" (Demo)", ""), "#6b7280"),
                "probabilities": probabilities
            })
            
        return {"faces": results, "count": len(results)}
        
    except Exception as e:
        tb = traceback.format_exc()
        logging.error("Server prediction failure:\n%s", tb)
        # Return a generic message to the client; details are in server logs
        raise HTTPException(status_code=500, detail="Internal prediction error. Check server logs for details.")

if __name__ == "__main__":
    import uvicorn
    # Start ASGI dev server on localhost:8000
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
