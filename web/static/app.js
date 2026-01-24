/**
 * MLX Video Processor - Frontend Application
 *
 * Handles video upload, model selection, processing, and results display
 * with real-time WebSocket updates.
 */

// State
const state = {
    videoId: null,
    videoInfo: null,
    selectedModels: new Set(),
    jobId: null,
    websocket: null,
    models: [],
};

// DOM Elements
const elements = {
    // Sections
    uploadSection: document.getElementById('upload-section'),
    modelSection: document.getElementById('model-section'),
    processingSection: document.getElementById('processing-section'),
    resultsSection: document.getElementById('results-section'),
    gallerySection: document.getElementById('gallery-section'),

    // Upload
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('file-input'),
    videoInfo: document.getElementById('video-info'),
    previewVideo: document.getElementById('preview-video'),
    videoFilename: document.getElementById('video-filename'),
    videoResolution: document.getElementById('video-resolution'),
    videoDuration: document.getElementById('video-duration'),
    videoFps: document.getElementById('video-fps'),
    videoFrames: document.getElementById('video-frames'),
    changeVideoBtn: document.getElementById('change-video-btn'),

    // Models
    modelGrid: document.getElementById('model-grid'),
    processBtn: document.getElementById('process-btn'),

    // Options
    frameInterval: document.getElementById('frame-interval'),
    confidence: document.getElementById('confidence'),
    confidenceValue: document.getElementById('confidence-value'),
    maxFrames: document.getElementById('max-frames'),
    targetObjects: document.getElementById('target-objects'),

    // Processing
    progressStatus: document.getElementById('progress-status'),
    progressPercent: document.getElementById('progress-percent'),
    progressFill: document.getElementById('progress-fill'),
    currentFrame: document.getElementById('current-frame'),
    processingFps: document.getElementById('processing-fps'),
    elapsedTime: document.getElementById('elapsed-time'),
    currentModel: document.getElementById('current-model'),
    logs: document.getElementById('logs'),

    // Results
    originalVideo: document.getElementById('original-video'),
    processedVideo: document.getElementById('processed-video'),
    downloadBtn: document.getElementById('download-btn'),
    newVideoBtn: document.getElementById('new-video-btn'),
    summaryContent: document.getElementById('summary-content'),

    // Gallery
    galleryGrid: document.getElementById('gallery-grid'),
    galleryEmpty: document.getElementById('gallery-empty'),

    // Toast
    toastContainer: document.getElementById('toast-container'),
};

// Model icons mapping
const MODEL_ICONS = {
    classification: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82z"/>
        <line x1="7" y1="7" x2="7.01" y2="7"/>
    </svg>`,
    detection: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
        <rect x="7" y="7" width="10" height="10" rx="1"/>
    </svg>`,
    segmentation: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polygon points="12 2 2 7 12 12 22 7 12 2"/>
        <polyline points="2 17 12 22 22 17"/>
        <polyline points="2 12 12 17 22 12"/>
    </svg>`,
    depth: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="2" y="2" width="20" height="20" rx="2"/>
        <path d="M7 7h10v10H7z" opacity="0.5"/>
        <path d="M10 10h4v4h-4z"/>
    </svg>`,
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    loadModels();
    loadGallery();
});

// Event Listeners
function initializeEventListeners() {
    // Drop zone
    elements.dropZone.addEventListener('click', () => elements.fileInput.click());
    elements.dropZone.addEventListener('dragover', handleDragOver);
    elements.dropZone.addEventListener('dragleave', handleDragLeave);
    elements.dropZone.addEventListener('drop', handleDrop);
    elements.fileInput.addEventListener('change', handleFileSelect);

    // Video controls
    elements.changeVideoBtn.addEventListener('click', resetUpload);

    // Options
    elements.confidence.addEventListener('input', (e) => {
        elements.confidenceValue.textContent = e.target.value;
    });

    // Process button
    elements.processBtn.addEventListener('click', startProcessing);

    // Results buttons
    elements.downloadBtn.addEventListener('click', downloadProcessedVideo);
    elements.newVideoBtn.addEventListener('click', resetToUpload);
}

// File Upload Handlers
function handleDragOver(e) {
    e.preventDefault();
    elements.dropZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    elements.dropZone.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

async function uploadFile(file) {
    // Validate file type
    const validTypes = ['.mp4', '.mov', '.avi', '.mkv', '.webm'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!validTypes.includes(ext)) {
        showToast('Invalid file type. Please upload a video file.', 'error');
        return;
    }

    // Validate file size
    if (file.size > 500 * 1024 * 1024) {
        showToast('File too large. Maximum size is 500MB.', 'error');
        return;
    }

    // Show loading state
    elements.dropZone.innerHTML = `
        <div class="drop-zone-content">
            <div class="spinner"></div>
            <p class="drop-text">Uploading video...</p>
            <p class="drop-subtext">Please wait</p>
        </div>
    `;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        const data = await response.json();
        state.videoId = data.video_id;
        state.videoInfo = data;

        showVideoInfo(data);
        showToast('Video uploaded successfully!', 'success');

    } catch (error) {
        showToast(error.message, 'error');
        resetDropZone();
    }
}

function showVideoInfo(info) {
    // Update video preview
    elements.previewVideo.src = `/api/video/${info.video_id}`;

    // Update video details
    elements.videoFilename.textContent = info.filename;
    elements.videoResolution.textContent = `${info.width}x${info.height}`;
    elements.videoDuration.textContent = formatDuration(info.duration);
    elements.videoFps.textContent = `${Math.round(info.fps)} FPS`;
    elements.videoFrames.textContent = `${info.total_frames.toLocaleString()} frames`;

    // Show video info, hide drop zone
    elements.dropZone.classList.add('hidden');
    elements.videoInfo.classList.remove('hidden');

    // Show model section
    elements.modelSection.classList.remove('hidden');
}

function resetDropZone() {
    elements.dropZone.innerHTML = `
        <div class="drop-zone-content">
            <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <p class="drop-text">Drag and drop your video here</p>
            <p class="drop-subtext">or click to browse</p>
            <p class="drop-formats">Supported: MP4, MOV, AVI, MKV, WebM (max 500MB)</p>
        </div>
    `;
}

function resetUpload() {
    state.videoId = null;
    state.videoInfo = null;
    elements.videoInfo.classList.add('hidden');
    elements.dropZone.classList.remove('hidden');
    elements.modelSection.classList.add('hidden');
    resetDropZone();
    elements.fileInput.value = '';
}

// Model Loading
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        state.models = data.models;
        renderModels();
    } catch (error) {
        showToast('Failed to load models', 'error');
    }
}

function renderModels() {
    elements.modelGrid.innerHTML = state.models.map(model => `
        <div class="model-card" data-model="${model.name}" onclick="toggleModel('${model.name}')">
            <div class="model-checkbox">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
                    <polyline points="20 6 9 17 4 12"/>
                </svg>
            </div>
            <div class="model-card-header">
                <div class="model-icon ${model.type.toLowerCase()}">
                    ${MODEL_ICONS[model.type.toLowerCase()] || MODEL_ICONS.detection}
                </div>
                <div>
                    <div class="model-name">${model.display_name}</div>
                    <div class="model-type">${model.type}</div>
                </div>
            </div>
            <p class="model-description">${model.description}</p>
        </div>
    `).join('');
}

function toggleModel(modelName) {
    const card = document.querySelector(`[data-model="${modelName}"]`);

    if (state.selectedModels.has(modelName)) {
        state.selectedModels.delete(modelName);
        card.classList.remove('selected');
    } else {
        state.selectedModels.add(modelName);
        card.classList.add('selected');
    }

    // Update process button
    elements.processBtn.disabled = state.selectedModels.size === 0;
}

// Make toggleModel available globally
window.toggleModel = toggleModel;

// Processing
async function startProcessing() {
    if (!state.videoId || state.selectedModels.size === 0) {
        showToast('Please upload a video and select at least one model', 'warning');
        return;
    }

    // Get options
    const frameInterval = parseInt(elements.frameInterval.value);
    const confidence = parseFloat(elements.confidence.value);
    const maxFrames = elements.maxFrames.value ? parseInt(elements.maxFrames.value) : null;
    const targetObjects = elements.targetObjects.value.trim();

    // Show processing section
    elements.modelSection.classList.add('hidden');
    elements.processingSection.classList.remove('hidden');

    // Clear logs
    elements.logs.innerHTML = '';
    addLog('Starting processing...', 'info');

    try {
        const response = await fetch('/api/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                video_id: state.videoId,
                models: Array.from(state.selectedModels),
                frame_interval: frameInterval,
                confidence_threshold: confidence,
                max_frames: maxFrames,
                target_objects: targetObjects || null,
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to start processing');
        }

        const data = await response.json();
        state.jobId = data.job_id;

        addLog(`Job started: ${data.job_id}`, 'success');
        addLog(`Models: ${Array.from(state.selectedModels).join(', ')}`, 'info');

        // Connect WebSocket for progress updates
        connectWebSocket(data.job_id);

    } catch (error) {
        showToast(error.message, 'error');
        addLog(error.message, 'error');
    }
}

function connectWebSocket(jobId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${jobId}`;

    state.websocket = new WebSocket(wsUrl);

    state.websocket.onopen = () => {
        addLog('Connected to server', 'success');
    };

    state.websocket.onmessage = (event) => {
        if (event.data === 'pong' || event.data === 'keepalive') {
            return;
        }

        try {
            const data = JSON.parse(event.data);
            updateProgress(data);
        } catch (e) {
            // Ignore non-JSON messages
        }
    };

    state.websocket.onerror = (error) => {
        addLog('WebSocket error', 'error');
    };

    state.websocket.onclose = () => {
        // Try to get final status
        setTimeout(() => checkJobStatus(jobId), 1000);
    };

    // Send periodic pings
    const pingInterval = setInterval(() => {
        if (state.websocket && state.websocket.readyState === WebSocket.OPEN) {
            state.websocket.send('ping');
        } else {
            clearInterval(pingInterval);
        }
    }, 25000);
}

async function checkJobStatus(jobId) {
    try {
        const response = await fetch(`/api/job/${jobId}`);
        const data = await response.json();
        updateProgress(data);
    } catch (error) {
        // Ignore errors
    }
}

function updateProgress(data) {
    // Update progress display
    elements.progressStatus.textContent = getStatusText(data.status);
    elements.progressPercent.textContent = `${Math.round(data.progress)}%`;
    elements.progressFill.style.width = `${data.progress}%`;

    elements.currentFrame.textContent = `${data.current_frame.toLocaleString()} / ${data.total_frames.toLocaleString()}`;
    elements.processingFps.textContent = `${data.fps.toFixed(1)} FPS`;
    elements.elapsedTime.textContent = formatDuration(data.elapsed_time);
    elements.currentModel.textContent = data.current_model || '-';

    // Handle completion
    if (data.status === 'completed') {
        addLog('Processing completed!', 'success');
        showResults(data);
    } else if (data.status === 'error') {
        addLog(`Error: ${data.error}`, 'error');
        showToast(data.error, 'error');
    }
}

function getStatusText(status) {
    const statusMap = {
        pending: 'Initializing...',
        processing: 'Processing frames...',
        completed: 'Complete!',
        error: 'Error',
    };
    return statusMap[status] || status;
}

function showResults(data) {
    // Hide processing, show results
    elements.processingSection.classList.add('hidden');
    elements.resultsSection.classList.remove('hidden');

    // Set video sources
    elements.originalVideo.src = `/api/video/${state.videoId}`;
    elements.processedVideo.src = `/api/output/${data.output_video}`;

    // Store output filename for download
    state.outputVideo = data.output_video;

    // Show summary
    elements.summaryContent.innerHTML = `
        <div class="summary-item">
            <span class="summary-label">Frames</span>
            <span class="summary-value">${data.total_frames.toLocaleString()}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Processing Time</span>
            <span class="summary-value">${formatDuration(data.elapsed_time)}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Avg FPS</span>
            <span class="summary-value">${data.fps.toFixed(1)}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Models</span>
            <span class="summary-value">${state.selectedModels.size}</span>
        </div>
    `;

    // Refresh gallery
    loadGallery();

    showToast('Video processed successfully!', 'success');
}

function downloadProcessedVideo() {
    if (state.outputVideo) {
        const link = document.createElement('a');
        link.href = `/api/output/${state.outputVideo}`;
        link.download = state.outputVideo;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}

function resetToUpload() {
    // Reset state
    state.videoId = null;
    state.videoInfo = null;
    state.selectedModels.clear();
    state.jobId = null;
    state.outputVideo = null;

    if (state.websocket) {
        state.websocket.close();
        state.websocket = null;
    }

    // Reset UI
    elements.resultsSection.classList.add('hidden');
    elements.processingSection.classList.add('hidden');
    elements.modelSection.classList.add('hidden');
    elements.uploadSection.classList.remove('hidden');
    elements.videoInfo.classList.add('hidden');
    elements.dropZone.classList.remove('hidden');

    // Reset model selection
    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.remove('selected');
    });
    elements.processBtn.disabled = true;

    // Reset form
    elements.fileInput.value = '';
    elements.frameInterval.value = '1';
    elements.confidence.value = '0.5';
    elements.confidenceValue.textContent = '0.5';
    elements.maxFrames.value = '';

    resetDropZone();
}

// Gallery
async function loadGallery() {
    try {
        const response = await fetch('/api/gallery');
        const data = await response.json();

        if (data.videos.length === 0) {
            elements.galleryGrid.classList.add('hidden');
            elements.galleryEmpty.classList.remove('hidden');
            return;
        }

        elements.galleryEmpty.classList.add('hidden');
        elements.galleryGrid.classList.remove('hidden');

        elements.galleryGrid.innerHTML = data.videos.map(video => `
            <div class="gallery-item">
                <div class="gallery-thumbnail">
                    <video src="/api/output/${video.filename}" muted></video>
                </div>
                <div class="gallery-info">
                    <div class="gallery-filename">${video.filename}</div>
                    <div class="gallery-meta">${video.size_mb.toFixed(1)} MB - ${formatDate(video.created)}</div>
                </div>
            </div>
        `).join('');

        // Add hover play functionality
        document.querySelectorAll('.gallery-item video').forEach(video => {
            video.addEventListener('mouseenter', () => video.play());
            video.addEventListener('mouseleave', () => {
                video.pause();
                video.currentTime = 0;
            });
        });

    } catch (error) {
        // Silently fail for gallery
    }
}

// Logging
function addLog(message, type = 'info') {
    const time = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-message ${type}">${message}</span>
    `;
    elements.logs.appendChild(entry);
    elements.logs.scrollTop = elements.logs.scrollHeight;
}

// Toast Notifications
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span>${message}</span>
    `;
    elements.toastContainer.appendChild(toast);

    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

// Utility Functions
function formatDuration(seconds) {
    if (!seconds || isNaN(seconds)) return '0:00';

    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function formatDate(isoString) {
    const date = new Date(isoString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
