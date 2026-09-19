// Voice Sentinel - Real-Time Microphone & Live Chunk Recorder
let audioContext = null;
let mediaStream = null;
let analyserNode = null;
let mediaRecorder = null;
let recordedChunks = [];
let isListening = false;
let chunkInterval = null;
let historyTimeline = [];
let rollingHistory = [];
let visualizerAnimFrame = null;

// Oscilloscope Visualizer
function startOscilloscope(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !analyserNode) return;
  const ctx = canvas.getContext('2d');

  const bufferLength = analyserNode.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function draw() {
    if (!isListening) return;
    visualizerAnimFrame = requestAnimationFrame(draw);

    analyserNode.getByteTimeDomainData(dataArray);

    ctx.fillStyle = '#0B1120';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 2;
    ctx.strokeStyle = '#06B6D4';
    ctx.beginPath();

    const sliceWidth = canvas.width * 1.0 / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = v * canvas.height / 2;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
      x += sliceWidth;
    }

    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
  }

  draw();
}

async function startLiveSentinel(canvasId) {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: true
      }
    });

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(mediaStream);
    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 2048;
    source.connect(analyserNode);

    isListening = true;
    startOscilloscope(canvasId);

    // Continuous 3.0s chunk sliding window
    beginChunkRecording();

    updateMicUI(true);
  } catch (err) {
    console.error('Microphone access error:', err);
    alert('Microphone permission denied or unavailable: ' + err.message);
    updateMicUI(false);
  }
}

function beginChunkRecording() {
  if (!isListening || !mediaStream) return;

  const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' :
                   (MediaRecorder.isTypeSupported('audio/ogg') ? 'audio/ogg' : 'audio/wav');

  let currentRecorder = new MediaRecorder(mediaStream, { mimeType: mimeType });
  let chunks = [];

  currentRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) {
      chunks.push(e.data);
    }
  };

  currentRecorder.onstop = async () => {
    if (chunks.length > 0 && isListening) {
      const blob = new Blob(chunks, { type: mimeType });
      await sendLiveChunk(blob);
    }
    // Schedule next chunk if still listening
    if (isListening) {
      setTimeout(beginChunkRecording, 500);
    }
  };

  currentRecorder.start();

  // Record for 3.0 seconds per window
  setTimeout(() => {
    if (currentRecorder.state === 'recording') {
      currentRecorder.stop();
    }
  }, 3000);
}

async function sendLiveChunk(audioBlob) {
  const formData = new FormData();
  const ext = audioBlob.type.includes('webm') ? 'webm' : (audioBlob.type.includes('ogg') ? 'ogg' : 'wav');
  formData.append('audio', audioBlob, `live_mic_chunk.${ext}`);
  formData.append('history', JSON.stringify(rollingHistory.slice(-5)));

  const threshold = parseFloat(document.getElementById('liveThreshold')?.value || 0.5);
  formData.append('threshold', threshold);

  try {
    const statusText = document.getElementById('liveAnalyzingBadge');
    if (statusText) statusText.innerText = 'Extracting Acoustic Features...';

    const resp = await fetch('/live-analyze', {
      method: 'POST',
      body: formData
    });

    const data = await resp.json();
    if (statusText) statusText.innerText = 'Awaiting Next Audio Window...';

    if (data.status === 'success') {
      processLiveResult(data);
    } else {
      console.warn('Live chunk rejected:', data.message);
    }
  } catch (err) {
    console.error('Live analysis upload failed:', err);
  }
}

function processLiveResult(data) {
  // Push to rolling history
  rollingHistory.push({
    ai_prob: data.ai_probability,
    prediction: data.prediction,
    timestamp: new Date().toLocaleTimeString()
  });

  // Calculate rolling average
  const recent = rollingHistory.slice(-5);
  const avgAi = recent.reduce((sum, item) => sum + item.ai_prob, 0) / recent.length;
  const avgHuman = 100 - avgAi;

  // Update UI indicators
  const resultCard = document.getElementById('liveResultCard');
  const predText = document.getElementById('livePredictionText');
  const aiBar = document.getElementById('liveAiBar');
  const humanBar = document.getElementById('liveHumanBar');
  const aiPct = document.getElementById('liveAiPct');
  const humanPct = document.getElementById('liveHumanPct');
  const rollingAiText = document.getElementById('liveRollingAi');
  const windowCount = document.getElementById('liveWindowCount');

  if (windowCount) windowCount.innerText = rollingHistory.length;
  if (rollingAiText) rollingAiText.innerText = avgAi.toFixed(1) + '%';

  if (aiBar) aiBar.style.width = avgAi + '%';
  if (humanBar) humanBar.style.width = avgHuman + '%';
  if (aiPct) aiPct.innerText = avgAi.toFixed(1) + '%';
  if (humanPct) humanPct.innerText = avgHuman.toFixed(1) + '%';

  const threshold = parseFloat(document.getElementById('liveThreshold')?.value || 0.5) * 100;
  const isAi = avgAi >= threshold;

  if (predText) {
    predText.innerText = isAi ? 'AI-GENERATED' : 'HUMAN';
    predText.className = 'prediction-value ' + (isAi ? 'val-ai' : 'val-human');
  }

  if (resultCard) {
    resultCard.className = 'result-banner ' + (isAi ? 'result-ai' : 'result-human');
  }

  // Update timeline table
  addTimelineEntry(data, avgAi, isAi);
}

function addTimelineEntry(data, avgAi, isAi) {
  const tbody = document.getElementById('liveTimelineBody');
  if (!tbody) return;

  const tr = document.createElement('tr');
  const time = new Date().toLocaleTimeString();
  tr.innerHTML = `
    <td><code>${time}</code></td>
    <td>${data.duration_seconds}s</td>
    <td><span style="color: ${data.prediction === 'AI-GENERATED' ? '#F43F5E' : '#10B981'}; font-weight: 700;">${data.prediction}</span></td>
    <td>${data.ai_probability.toFixed(1)}%</td>
    <td><strong>${avgAi.toFixed(1)}%</strong></td>
    <td>${data.confidence.toFixed(1)}%</td>
  `;

  if (tbody.children.length >= 12) {
    tbody.removeChild(tbody.lastChild);
  }
  tbody.insertBefore(tr, tbody.firstChild);
}

function stopLiveSentinel() {
  isListening = false;
  if (visualizerAnimFrame) {
    cancelAnimationFrame(visualizerAnimFrame);
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
  }
  if (audioContext) {
    audioContext.close();
  }
  updateMicUI(false);
}

function updateMicUI(listening) {
  const btnStart = document.getElementById('btnStartMic');
  const btnStop = document.getElementById('btnStopMic');
  const micStatus = document.getElementById('micStatusText');
  const pulseElem = document.getElementById('micPulseIndicator');

  if (btnStart) btnStart.disabled = listening;
  if (btnStop) btnStop.disabled = !listening;

  if (micStatus) {
    micStatus.innerText = listening ? 'LISTENING & STREAMING AUDIO' : 'IDLE - MICROPHONE OFF';
    micStatus.style.color = listening ? '#10B981' : '#64748B';
  }

  if (pulseElem) {
    if (listening) {
      pulseElem.classList.add('pulse-recording');
    } else {
      pulseElem.classList.remove('pulse-recording');
    }
  }
}
