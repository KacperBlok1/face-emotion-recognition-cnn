// DOM elements lookup
const webcam = document.getElementById('webcam');
const canvasOverlay = document.getElementById('canvasOverlay');
const ctx = canvasOverlay.getContext('2d');
const noFacePlaceholder = document.getElementById('noFacePlaceholder');
const btnStartCamera = document.getElementById('btnStartCamera');
const btnStopCamera = document.getElementById('btnStopCamera');
const cameraStatus = document.getElementById('cameraStatus');
const indicatorText = cameraStatus.querySelector('.indicator-text');
const intervalSelect = document.getElementById('intervalSelect');

// Dominant analytics elements
const dominantEmotion = document.getElementById('dominantEmotion');
const dominantConfidenceValue = document.getElementById('dominantConfidenceValue');
const dominantConfidenceFill = document.getElementById('dominantConfidenceFill');

// Local application state
let stream = null;
let captureInterval = null;
let isRunning = false;

// Event listeners setup
btnStartCamera.addEventListener('click', startCamera);
btnStopCamera.addEventListener('click', stopCamera);
intervalSelect.addEventListener('change', () => {
    if (isRunning) {
        // Dynamic reload interval on user selection change
        clearInterval(captureInterval);
        startPredictionLoop();
    }
});

// Configure responsive canvas dimensions to match active streaming dimensions
function resizeCanvas() {
    canvasOverlay.width = webcam.videoWidth || 640;
    canvasOverlay.height = webcam.videoHeight || 480;
}

// Request and mount webcam media stream
async function startCamera() {
    try {
        indicatorText.textContent = "Uruchamianie...";
        
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        };
        
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        webcam.srcObject = stream;
        
        webcam.onloadedmetadata = () => {
            webcam.play();
            resizeCanvas();
            noFacePlaceholder.classList.remove('active');
            btnStartCamera.disabled = true;
            btnStopCamera.disabled = false;
            cameraStatus.classList.add('active');
            indicatorText.textContent = "Działa";
            isRunning = true;
            
            // Start the prediction loop
            startPredictionLoop();
        };
    } catch (err) {
        console.error("Camera access failed: ", err);
        indicatorText.textContent = "Błąd kamery";
        alert("Nie można uzyskać dostępu do kamery. Upewnij się, że przyznano odpowiednie uprawnienia w przeglądarce.");
    }
}

// Release media streams and stop processing loop
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    
    clearInterval(captureInterval);
    webcam.srcObject = null;
    isRunning = false;
    
    // Reset UI states
    btnStartCamera.disabled = false;
    btnStopCamera.disabled = true;
    cameraStatus.classList.remove('active');
    indicatorText.textContent = "Zatrzymana";
    noFacePlaceholder.classList.add('active');
    
    // Clear overlay canvas
    ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
    
    // Reset indicators
    resetEmotionBars();
    dominantEmotion.textContent = "Brak Twarzy";
    dominantConfidenceValue.textContent = "0%";
    dominantConfidenceFill.style.width = "0%";
    
    console.log("Webcam stream stopped.");
}

// Start interval loop based on user setting
function startPredictionLoop() {
    const timeInterval = parseInt(intervalSelect.value);
    captureInterval = setInterval(captureAndPredict, timeInterval);
}

// Captures a frame from video element, sends to FastAPI, and renders predictions
async function captureAndPredict() {
    if (!isRunning || webcam.paused || webcam.ended) return;
    
    // 1. Capture snapshot to hidden canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvasOverlay.width;
    tempCanvas.height = canvasOverlay.height;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(webcam, 0, 0, tempCanvas.width, tempCanvas.height);
    
    // Get Base64 image payload (compression level 0.7 to minimize network payload bandwidth)
    const base64Image = tempCanvas.toDataURL('image/jpeg', 0.7);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Image })
        });
        
        if (!response.ok) throw new Error("API call failed");
        
        const data = await response.json();
        
        // 2. Clear canvas prior to drawing new frames
        ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
        
        if (data.count === 0) {
            // No faces detected case
            dominantEmotion.textContent = "Nie wykryto twarzy";
            dominantConfidenceValue.textContent = "0%";
            dominantConfidenceFill.style.style = "0%";
            dominantConfidenceFill.style.backgroundColor = "var(--text-secondary)";
            resetEmotionBars();
            return;
        }
        
        // Process detected faces
        data.faces.forEach((face, idx) => {
            const { x, y, w, h } = face.box;
            const emotion = face.emotion;
            const confidence = face.confidence;
            const color = face.color;
            
            // 3. Draw Bounding Box (use color matching detected emotion)
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.lineJoin = "round";
            ctx.strokeRect(x, y, w, h);
            
            // Draw label box above rectangle (drawing flipped label because canvas scaleX is mirrored)
            ctx.fillStyle = color;
            const labelStr = `${emotion} ${(confidence * 100).toFixed(0)}%`;
            ctx.font = "bold 14px 'Outfit', sans-serif";
            const textWidth = ctx.measureText(labelStr).width;
            const textHeight = 16;
            
            ctx.fillRect(x - 1.5, y - textHeight - 8, textWidth + 15, textHeight + 8);
            
            ctx.fillStyle = '#ffffff';
            ctx.fillText(labelStr, x + 6, y - 8);
            
            // 4. Update dominant analytics side widget (use first detected face as primary)
            if (idx === 0) {
                dominantEmotion.textContent = emotion;
                dominantEmotion.style.color = color;
                dominantConfidenceValue.textContent = `${(confidence * 100).toFixed(1)}%`;
                dominantConfidenceFill.style.width = `${confidence * 100}%`;
                dominantConfidenceFill.style.backgroundColor = color;
                
                // Complete list of probabilities update
                updateEmotionBars(face.probabilities);
            }
        });
        
    } catch (err) {
        console.error("Inference fetch error: ", err);
    }
}

// Animate classification confidence sliders
function updateEmotionBars(probs) {
    if (!probs) return;
    
    for (const [emotion, val] of Object.entries(probs)) {
        const fillElement = document.getElementById(`bar-${emotion}`);
        const textElement = document.getElementById(`value-${emotion}`);
        
        if (fillElement && textElement) {
            const percentage = (val * 100).toFixed(1);
            fillElement.style.width = `${percentage}%`;
            textElement.textContent = `${percentage}%`;
            textElement.style.fontWeight = val > 0.3 ? '700' : '500';
        }
    }
}

// Reset classification confidence sliders to 0.0%
function resetEmotionBars() {
    const fills = document.querySelectorAll('.progress-bar-fill');
    const texts = document.querySelectorAll('.emotion-value-cell');
    
    fills.forEach(fill => fill.style.width = '0%');
    texts.forEach(text => {
        text.textContent = '0.0%';
        text.style.fontWeight = '500';
    });
}

// Keep canvas overlay matching video aspect-ratio on browser window resizing
window.addEventListener('resize', resizeCanvas);
// Call resizing immediately to initialize coordinates
resizeCanvas();
