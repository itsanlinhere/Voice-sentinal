// Voice Sentinel Dashboard Interactions

document.addEventListener('DOMContentLoaded', () => {
  setupDropzone();
  setupThresholdSlider();
});

function setupDropzone() {
  const dropzone = document.getElementById('audioDropzone');
  const fileInput = document.getElementById('audioFileInput');
  if (!dropzone || !fileInput) return;

  dropzone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      fileInput.files = files;
      handleFileSelected(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });
}

function handleFileSelected(file) {
  const selectedName = document.getElementById('selectedFileName');
  const previewAudio = document.getElementById('audioPlayerPreview');
  const playerContainer = document.getElementById('audioPlayerContainer');

  if (selectedName) selectedName.innerText = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  
  if (previewAudio && playerContainer) {
    const url = URL.createObjectURL(file);
    previewAudio.src = url;
    playerContainer.style.display = 'block';
  }

  const btnAnalyze = document.getElementById('btnSubmitAnalyze');
  if (btnAnalyze) btnAnalyze.disabled = false;
}

function setupThresholdSlider() {
  const slider = document.getElementById('thresholdSlider');
  const valDisplay = document.getElementById('thresholdValueDisplay');
  if (!slider || !valDisplay) return;

  slider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    valDisplay.innerText = `${(val * 100).toFixed(0)}%`;
    recomputeThresholdEvaluation(val);
  });
}

function recomputeThresholdEvaluation(newThreshold) {
  const aiProbElem = document.getElementById('analysisAiProbRaw');
  const predElem = document.getElementById('analysisPredictionValue');
  const banner = document.getElementById('analysisResultBanner');

  if (!aiProbElem || !predElem) return;

  const aiProb = parseFloat(aiProbElem.value) / 100.0;
  const isAi = aiProb >= newThreshold;

  predElem.innerText = isAi ? 'AI-GENERATED' : 'HUMAN';
  predElem.className = 'prediction-value ' + (isAi ? 'val-ai' : 'val-human');

  if (banner) {
    banner.className = 'result-banner ' + (isAi ? 'result-ai' : 'result-human');
  }
}
