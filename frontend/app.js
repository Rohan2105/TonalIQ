/* frontend/app.js */

// 1. Dynamic Routing Configuration
const getBackendUrls = () => {
    const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    return {
        ws: isLocal ? "ws://localhost:8000/api/predict/stream" : "wss://your-production-backend-url.com/api/predict/stream",
        http: isLocal ? "http://localhost:8000/api/predict/file" : "https://your-production-backend-url.com/api/predict/file",
        health: isLocal ? "http://localhost:8000/health" : "https://your-production-backend-url.com/health"
    };
};

const URLS = getBackendUrls();

// 2. DOM Elements
const connectionDot = document.getElementById('connectionDot');
const systemStatus = document.getElementById('systemStatus');
const streamBtn = document.getElementById('streamBtn');
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const fileInfo = document.getElementById('fileInfo');
const dominantEmotionValue = document.getElementById('dominantEmotionValue');
const confidenceBadge = document.getElementById('confidenceBadge');
const analysisTime = document.getElementById('analysisTime');
const xaiPlaceholder = document.getElementById('xaiPlaceholder');
const xaiImage = document.getElementById('xaiImage');
const vumeterCanvas = document.getElementById('vumeter');
const vumeterContainer = document.querySelector('.vumeter-container');

// Audio Context elements
let audioContext = null;
let micStream = null;
let processorNode = null;
let socket = null;
let isStreaming = false;

// 3. UI Update Helpers
function updateSystemStatus(status, isConnected = false, isActiveStream = false) {
    systemStatus.textContent = status;
    connectionDot.className = "status-dot";
    if (isConnected) {
        connectionDot.classList.add('connected');
    }
    if (isActiveStream) {
        connectionDot.classList.add('active');
    }
}

function updateGauges(probabilities) {
    if (!probabilities) return;
    
    Object.keys(probabilities).forEach(emotion => {
        // Find row by data-emotion attribute
        const row = document.querySelector(`.gauge-row[data-emotion="${emotion}"]`);
        if (row) {
            const fill = row.querySelector('.gauge-bar-fill');
            const valText = row.querySelector('.gauge-val');
            
            const probPercent = Math.round(probabilities[emotion] * 100);
            
            fill.style.width = `${probPercent}%`;
            valText.textContent = `${probPercent}%`;
        }
    });
}

function resetUI() {
    dominantEmotionValue.textContent = "—";
    confidenceBadge.textContent = "0.0%";
    analysisTime.textContent = "Latency: —ms";
    
    // Clear gauges
    const rows = document.querySelectorAll('.gauge-row');
    rows.forEach(row => {
        row.querySelector('.gauge-bar-fill').style.width = '0%';
        row.querySelector('.gauge-val').textContent = '0%';
    });
    
    // Clear XAI Graph
    xaiImage.style.display = 'none';
    xaiPlaceholder.style.display = 'block';
}

// Check Backend Health on startup
async function checkHealth() {
    try {
        const res = await fetch(URLS.health);
        if (res.ok) {
            const data = await res.json();
            if (data.onnx_loaded) {
                updateSystemStatus("Ready (ONNX)", true);
            } else {
                updateSystemStatus("Ready (PyTorch Fallback)", true);
            }
        } else {
            updateSystemStatus("Offline", false);
        }
    } catch (err) {
        console.error("Health check failed:", err);
        updateSystemStatus("Connection Error", false);
    }
}

// 4. VU Meter Renderer
function drawVUMeter(audioFrame) {
    if (!vumeterCanvas) return;
    const ctx = vumeterCanvas.getContext('2d');
    const width = vumeterCanvas.width;
    const height = vumeterCanvas.height;
    
    // Calculate RMS energy
    let sum = 0;
    for (let i = 0; i < audioFrame.length; i++) {
        sum += audioFrame[i] * audioFrame[i];
    }
    const rms = Math.sqrt(sum / audioFrame.length);
    
    // Scale energy for visualization
    const fillAmount = Math.min(rms * 4, 1.0) * width;
    
    ctx.clearRect(0, 0, width, height);
    
    // Draw dynamic meter bar
    ctx.fillStyle = '#6366f1'; // Soft Indigo
    ctx.fillRect(0, 0, fillAmount, height);
}

// 5. WebSocket Streaming Logic
async function startStreaming() {
    if (isStreaming) return;
    
    try {
        resetUI();
        // Request Microphone Permissions
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        
        // Open WebSocket
        socket = new WebSocket(URLS.ws);
        socket.binaryType = 'arraybuffer';
        
        socket.onopen = () => {
            logger("WebSocket Stream connection opened.");
            updateSystemStatus("Streaming Active", true, true);
            streamBtn.textContent = "Stop Live Stream";
            streamBtn.className = "btn btn-primary recording";
            vumeterContainer.style.display = 'block';
            isStreaming = true;
            
            // Setup Web Audio nodes at 16kHz
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            const source = audioContext.createMediaStreamSource(micStream);
            
            // ScriptProcessor captures mono chunks of 4096 samples
            processorNode = audioContext.createScriptProcessor(4096, 1, 1);
            
            processorNode.onaudioprocess = (e) => {
                const inputFrame = e.inputBuffer.getChannelData(0); // Float32Array (16kHz mono)
                drawVUMeter(inputFrame);
                
                if (socket.readyState === WebSocket.OPEN) {
                    socket.send(inputFrame.buffer);
                }
            };
            
            source.connect(processorNode);
            processorNode.connect(audioContext.destination);
        };
        
        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.dominant_emotion === "Silence") {
                dominantEmotionValue.textContent = "Silence";
                confidenceBadge.textContent = "0.0%";
                return;
            }
            
            dominantEmotionValue.textContent = data.dominant_emotion;
            const domProb = data.probabilities[data.dominant_emotion.toLowerCase()] || 0.0;
            confidenceBadge.textContent = `${(domProb * 100).toFixed(1)}%`;
            analysisTime.textContent = `Streaming Real-Time Logits`;
            
            updateGauges(data.probabilities);
        };
        
        socket.onerror = (err) => {
            console.error("WebSocket encountered an error:", err);
            stopStreaming();
        };
        
        socket.onclose = () => {
            logger("WebSocket Stream connection closed.");
            stopStreaming();
        };
        
    } catch (err) {
        console.error("Microphone initialization failed:", err);
        updateSystemStatus("Mic Access Denied", false);
        alert("Could not access microphone. Ensure permissions are granted.");
    }
}

function stopStreaming() {
    if (!isStreaming) return;
    
    // Close WebSocket
    if (socket) {
        if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
            socket.close();
        }
        socket = null;
    }
    
    // Stop mic stream track inputs
    if (micStream) {
        micStream.getTracks().forEach(track => track.stop());
        micStream = null;
    }
    
    // Close Audio Context nodes
    if (processorNode) {
        processorNode.disconnect();
        processorNode = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    
    isStreaming = false;
    streamBtn.textContent = "Start Live Stream";
    streamBtn.className = "btn btn-primary";
    vumeterContainer.style.display = 'none';
    
    checkHealth();
}

// 6. REST API File Ingestion Logic
async function uploadAudioFile(file) {
    if (!file) return;
    
    resetUI();
    fileInfo.textContent = `Analyzing: ${file.name} (${(file.size/1024).toFixed(1)} KB)`;
    updateSystemStatus("Classifying File...", true);
    
    const formData = new FormData();
    formData.append("file", file);
    
    const startTime = performance.now();
    
    try {
        const res = await fetch(URLS.http, {
            method: "POST",
            body: formData
        });
        
        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || "Inference backend returned an error status.");
        }
        
        const data = await res.json();
        const latency = (performance.now() - startTime).toFixed(0);
        
        // Update DOM
        dominantEmotionValue.textContent = data.dominant_emotion;
        const domProb = data.probabilities[data.dominant_emotion.toLowerCase()] || 0.0;
        confidenceBadge.textContent = `${(domProb * 100).toFixed(1)}%`;
        analysisTime.textContent = `Latency: ${latency}ms`;
        
        updateGauges(data.probabilities);
        
        // Render Saliency Attention Map
        if (data.explainability_graph) {
            xaiPlaceholder.style.display = 'none';
            xaiImage.src = data.explainability_graph;
            xaiImage.style.display = 'block';
        } else {
            xaiImage.style.display = 'none';
            xaiPlaceholder.style.display = 'block';
        }
        
        checkHealth();
        
    } catch (err) {
        console.error("File upload prediction failed:", err);
        fileInfo.textContent = "Classification failed.";
        updateSystemStatus("Inference Error", false);
        alert(`Prediction failed: ${err.message}`);
    }
}

// 7. Event Listeners & Drag-Drop Bindings
streamBtn.addEventListener('click', () => {
    if (isStreaming) {
        stopStreaming();
    } else {
        startStreaming();
    }
});

browseBtn.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        uploadAudioFile(e.target.files[0]);
    }
});

// Drag and drop events
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
        uploadAudioFile(files[0]);
    }
});

// Debug Logger
function logger(msg) {
    console.log(`[SER APP] ${msg}`);
}

// Initialize health status check
checkHealth();
