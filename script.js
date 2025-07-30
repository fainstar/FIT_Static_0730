import { models } from './js/config.js';
import { preprocess, postprocess } from './js/onnx-handler.js';
import { drawBoundingBoxes } from './js/ui.js';
import { translations } from './js/i18n.js';

const imageUpload = document.getElementById('image-upload');
const imageCanvas = document.getElementById('image-canvas');
const modelSelect = document.getElementById('model-select');
const progressBarContainer = document.getElementById('progress-bar-container');
const progressBar = document.getElementById('progress-bar');
const langSelect = document.getElementById('lang-select');
const langSelectBtn = document.getElementById('lang-select-btn');
const loader = document.getElementById('loader');
const blurOverlay = document.querySelector('.blur-overlay');
const blurGrid = document.querySelector('.blur-grid');
const statsContainer = document.getElementById('stats-container');
const detectedObjectsEl = document.getElementById('detected-objects');
const inferenceTimeEl = document.getElementById('inference-time');
const avgConfidenceEl = document.getElementById('avg-confidence');
const inferenceDeviceEl = document.getElementById('inference-device');
const iouThresholdSlider = document.getElementById('iou-threshold');
const iouValueSpan = document.getElementById('iou-value');
const currentModelNameEl = document.getElementById('current-model-name');
const ctx = imageCanvas.getContext('2d');

let session;
let currentModel = 'lin_0725'; // Default model
let preProcessInfo = {}; // To store scaling info
let currentExecutionProvider = 'wasm'; // Default execution provider
let isModelLoaded = false; // Flag to track model loading status

// --- Main Application Logic ---

async function loadModel() {
    updateProgressBar(0, 'Loading model...');
    progressBarContainer.style.display = 'block';
    try {   
        const model = models[currentModel];
        updateProgressBar(30, 'Creating session...');
        // Set the execution provider
        const executionProvider = currentExecutionProvider;
        session = await ort.InferenceSession.create(model.path, {
            executionProviders: [executionProvider],
            graphOptimizationLevel: 'all'
        });
        updateProgressBar(100, 'Model loaded!');
        // Display the backend used by the session
        console.log('Execution Provider:', executionProvider);
        inferenceDeviceEl.textContent = executionProvider;
        console.log(`Model ${currentModel} loaded successfully.`);
        isModelLoaded = true; // Set the flag to true
        updateCurrentModelName();
        const lang = localStorage.getItem('language') || 'en';
        const promptText = translations[lang]['modelLoadedPrompt'].replace('{{modelName}}', currentModel);
        // The status paragraph is now part of the intro card, so we don't update it here anymore.
        // A more robust solution would be to have a dedicated status element.
        setTimeout(() => { progressBarContainer.style.display = 'none'; }, 500);
    } catch (e) {
        console.error(`Failed to load the model: ${e}`);
        document.querySelector('p').textContent = `Error: Failed to load model. ${e.message}`;
        progressBarContainer.style.display = 'none';
    }
}

function updateStats(boxes, inferenceTime) {
    const numObjects = boxes.length;
    const avgConfidence = numObjects > 0
        ? (boxes.reduce((acc, box) => acc + box.score, 0) / numObjects * 100).toFixed(2)
        : 0;

    detectedObjectsEl.textContent = numObjects;
    inferenceTimeEl.textContent = `${inferenceTime.toFixed(2)} ms`;
    avgConfidenceEl.textContent = `${avgConfidence}%`;
    // The inference device is set when the model is loaded

    statsContainer.style.display = 'flex';
}

async function detectObjects(img, scale) {
    if (!session || !isModelLoaded) {
        console.error('Session not initialized or model not loaded.');
        // Optionally, show a user-friendly message
        updateProgressBar(0, 'modelNotReady'); // A new i18n key
        setTimeout(() => {
            progressBarContainer.style.display = 'none';
            blurOverlay.style.display = 'none';
            blurGrid.style.display = 'none';
            blurGrid.innerHTML = '';
            loader.style.display = 'none';
        }, 2000);
        return;
    }

    // The blur animation is already running, just update the progress bar
    progressBarContainer.style.display = 'block';
    updateProgressBar(0, 'Preprocessing...');

    const model = models[currentModel];
    const inputTensor = preprocess(img, model.inputShape, preProcessInfo);
    updateProgressBar(30, 'Detecting objects...');

    try {
        const startTime = performance.now();
        const feeds = { 'images': inputTensor };
        const results = await session.run(feeds);
        const endTime = performance.now();

        const outputTensor = results.output0;
        updateProgressBar(70, 'Postprocessing...');

        const iouThreshold = parseFloat(iouThresholdSlider.value);
        const boxes = postprocess(outputTensor, preProcessInfo, model.outputShape, iouThreshold);
        drawBoundingBoxes(ctx, boxes, imageCanvas, scale);
        updateStats(boxes, endTime - startTime);

        updateProgressBar(100, 'Done!');
        setTimeout(() => {
            progressBarContainer.style.display = 'none';
            blurOverlay.style.display = 'none';
            blurGrid.style.display = 'none';
            blurGrid.innerHTML = ''; // Clear the grid
            loader.style.display = 'none'; // Also hide the initial loader
        }, 500);
    } catch (e) {
        console.error(`Inference failed: ${e}`);
        progressBarContainer.style.display = 'none';
        blurOverlay.style.display = 'none';
        blurGrid.style.display = 'none';
        blurGrid.innerHTML = ''; // Clear the grid
        loader.style.display = 'none'; // Also hide the initial loader
    }
}

function populateBlurGrid() {
    blurGrid.innerHTML = ''; // Clear previous grid
    for (let i = 0; i < 100; i++) {
        const cell = document.createElement('div');
        cell.classList.add('grid-cell');
        // Random delay for a more dynamic effect
        cell.style.animationDelay = `${Math.random() * 1.5}s`;
        blurGrid.appendChild(cell);
    }
}

function updateProgressBar(value, textKey) {
    const lang = localStorage.getItem('language') || 'en';
    const text = translations[lang][textKey] || textKey;
    progressBar.style.width = `${value}%`;
    progressBar.textContent = text || `${value}%`;
}

// --- I18n --- 

function setLanguage(lang) {
    localStorage.setItem('language', lang);
    langSelect.value = lang;
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (translations[lang] && translations[lang][key]) {
            let text = translations[lang][key];
            if (key === 'modelLoadedPrompt') {
                text = text.replace('{{modelName}}', currentModel);
            }
            
            const textNode = Array.from(el.childNodes).find(node => node.nodeType === Node.TEXT_NODE && node.parentElement.id !== 'theme-toggle');
            if (textNode) {
                textNode.textContent = text;
            } else if (el.querySelector('span')){
                el.querySelector('span').textContent = text;
            } else if (el.id !== 'theme-toggle') {
                el.textContent = text;
            }
        }
    });
    // The logic to update the prompt message has been simplified as the description paragraph was removed.
}

langSelect.addEventListener('change', (e) => {
    setLanguage(e.target.value);
    langSelect.style.display = 'none'; // Hide dropdown after selection
});

langSelectBtn.addEventListener('click', () => {
    langSelect.style.display = 'block';
    langSelect.focus();
});

langSelect.addEventListener('blur', () => {
    langSelect.style.display = 'none';
});

// --- Event Listeners ---

document.getElementById('model-select-btn').addEventListener('click', () => {
    modelSelect.style.display = 'block';
    modelSelect.focus();
});

modelSelect.addEventListener('blur', () => {
    modelSelect.style.display = 'none';
});

modelSelect.addEventListener('change', (e) => {
    currentModel = e.target.value;
    loadModel();
    modelSelect.blur(); // Hide dropdown after selection
});

iouThresholdSlider.addEventListener('input', (e) => {
    iouValueSpan.textContent = e.target.value;
    if (imageCanvas.dataset.hasImage === 'true') {
        const img = new Image();
        img.src = imageCanvas.toDataURL();
        img.onload = () => {
            const scale = Math.min(imageCanvas.width / img.width, imageCanvas.height / img.height);
            detectObjects(img, scale);
        };
    }
});

function updateCurrentModelName() {
    const selectedOption = modelSelect.options[modelSelect.selectedIndex];
    currentModelNameEl.textContent = selectedOption.text;
}



document.getElementById('upload-btn').addEventListener('click', () => {
    imageUpload.click();
});

async function handleImage(imageSource) {
    // --- Stage 1: Immediate Feedback ---
    document.querySelector('.canvas-container').style.display = 'block';
    loader.style.display = 'grid'; // Show initial block loader
    // Clear canvas and hide other elements
    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    blurOverlay.style.display = 'none';
    blurGrid.style.display = 'none';

    const img = new Image();
    img.onload = async () => {
        // --- Stage 2: Image Loaded, Switch to Blur Animation ---
        loader.style.display = 'none'; // Hide initial loader

        const maxWidth = imageCanvas.parentElement.clientWidth;
        const scale = Math.min(1, maxWidth / img.width);
        imageCanvas.width = img.width * scale;
        imageCanvas.height = img.height * scale;
        ctx.drawImage(img, 0, 0, imageCanvas.width, imageCanvas.height); // Draw the scaled image



        // Start the blur animation
        blurOverlay.style.display = 'block';
        blurGrid.style.display = 'grid';
        populateBlurGrid();

        // Wait for the model to be loaded before detecting objects
        if (!isModelLoaded) {
            console.log('Waiting for model to load...');
            // You can show a waiting message to the user here
            updateProgressBar(0, 'waitingForModel');
            await new Promise(resolve => {
                const interval = setInterval(() => {
                    if (isModelLoaded) {
                        clearInterval(interval);
                        resolve();
                    }
                }, 100); // Check every 100ms
            });
        }

        await detectObjects(img, scale); // Start detection
    };
    img.src = imageSource;
}

imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        handleImage(event.target.result);
    };
    reader.readAsDataURL(file);
});

document.querySelectorAll('.image-grid img').forEach(img => {
    img.addEventListener('click', () => {
        handleImage(img.src);
    });
});

// --- Theme Toggler --- 

const themeToggle = document.getElementById('theme-toggle');

function setTheme(theme) {
    document.body.dataset.theme = theme;
    localStorage.setItem('theme', theme);
    const sunIcon = document.querySelector('#theme-toggle .sun');
    const moonIcon = document.querySelector('#theme-toggle .moon');
    const langIcon = document.querySelector('.lang-icon');

    if (theme === 'dark') {
        sunIcon.style.display = 'none';
        moonIcon.style.display = 'block';
    } else {
        sunIcon.style.display = 'block';
        moonIcon.style.display = 'none';
    }
}

themeToggle.addEventListener('click', () => {
    const currentTheme = document.body.dataset.theme || 'light';
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
});

// --- Initialize Application ---

function initialize() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    const savedLang = localStorage.getItem('language') || 'en';
    setTheme(savedTheme);
    setLanguage(savedLang);
    loadModel();
    updateCurrentModelName();
}

// --- Initialize Application ---
initialize();