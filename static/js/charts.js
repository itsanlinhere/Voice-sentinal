// Voice Sentinel Chart.js Visualizers

// Waveform Time-Domain Renderer
function renderWaveform(canvasId, waveData) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !waveData) return;
  const ctx = canvas.getContext('2d');
  
  if (window[canvasId + '_chart']) {
    window[canvasId + '_chart'].destroy();
  }

  const labels = waveData.map((_, idx) => (idx * 0.02).toFixed(2) + 's');

  window[canvasId + '_chart'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Normalized Amplitude',
        data: waveData,
        borderColor: '#06B6D4',
        backgroundColor: 'rgba(6, 182, 212, 0.1)',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        tension: 0.1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            label: (ctx) => `Amp: ${ctx.parsed.y.toFixed(3)}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(51, 65, 85, 0.3)' },
          ticks: { color: '#64748B', maxTicksLimit: 8 }
        },
        y: {
          min: -1.0,
          max: 1.0,
          grid: { color: 'rgba(51, 65, 85, 0.3)' },
          ticks: { color: '#64748B' }
        }
      }
    }
  });
}

// Pitch (F0) Trajectory Renderer
function renderPitchContour(canvasId, pitchData) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !pitchData) return;
  const ctx = canvas.getContext('2d');

  if (window[canvasId + '_chart']) {
    window[canvasId + '_chart'].destroy();
  }

  const labels = pitchData.map(p => p.time + 's');
  const values = pitchData.map(p => p.f0);

  window[canvasId + '_chart'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Fundamental Frequency F0 (Hz)',
        data: values,
        borderColor: '#10B981',
        backgroundColor: 'rgba(16, 185, 129, 0.08)',
        borderWidth: 2,
        pointRadius: 2,
        pointBackgroundColor: '#10B981',
        fill: true,
        tension: 0.2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#94A3B8' } },
        tooltip: {
          callbacks: {
            label: (ctx) => `F0: ${ctx.parsed.y.toFixed(1)} Hz`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(51, 65, 85, 0.3)' },
          ticks: { color: '#64748B', maxTicksLimit: 10 }
        },
        y: {
          title: { display: true, text: 'Pitch (Hz)', color: '#64748B' },
          grid: { color: 'rgba(51, 65, 85, 0.3)' },
          ticks: { color: '#64748B' }
        }
      }
    }
  });
}

// Mel Spectrogram Heatmap Canvas Renderer
function renderMelSpectrogram(canvasId, melGrid) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !melGrid || melGrid.length === 0) return;
  const ctx = canvas.getContext('2d');

  const rows = melGrid.length;
  const cols = melGrid[0].length;
  const width = canvas.width = canvas.parentElement.clientWidth || 500;
  const height = canvas.height = 180;

  const cellWidth = width / cols;
  const cellHeight = height / rows;

  // Find min/max for colormap normalization
  let minDb = -80.0, maxDb = 0.0;

  for (let r = 0; r < rows; r++) {
    // In audio spectrograms, low frequencies are at bottom (invert y axis)
    const y = height - (r + 1) * cellHeight;
    for (let c = 0; c < cols; c++) {
      const val = melGrid[r][c];
      const norm = Math.max(0, Math.min(1, (val - minDb) / (maxDb - minDb)));
      
      // Cyber-Forensics thermal colormap (dark navy -> cyan -> amber -> bright white)
      let red = 0, green = 0, blue = 0;
      if (norm < 0.33) {
        const t = norm / 0.33;
        red = Math.floor(10 * (1 - t) + 6 * t);
        green = Math.floor(15 * (1 - t) + 182 * t);
        blue = Math.floor(35 * (1 - t) + 212 * t);
      } else if (norm < 0.66) {
        const t = (norm - 0.33) / 0.33;
        red = Math.floor(6 * (1 - t) + 245 * t);
        green = Math.floor(182 * (1 - t) + 158 * t);
        blue = Math.floor(212 * (1 - t) + 11 * t);
      } else {
        const t = (norm - 0.66) / 0.34;
        red = Math.floor(245 * (1 - t) + 255 * t);
        green = Math.floor(158 * (1 - t) + 255 * t);
        blue = Math.floor(11 * (1 - t) + 255 * t);
      }

      ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
      ctx.fillRect(c * cellWidth, y, Math.ceil(cellWidth), Math.ceil(cellHeight));
    }
  }
}

// ROC Curve Renderer
function renderRocCurve(canvasId, rocPoints, aucVal) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !rocPoints) return;
  const ctx = canvas.getContext('2d');

  if (window[canvasId + '_chart']) {
    window[canvasId + '_chart'].destroy();
  }

  const data = rocPoints.map(p => ({ x: p.fpr, y: p.tpr }));

  window[canvasId + '_chart'] = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: `ROC Curve (AUC = ${aucVal !== null ? aucVal.toFixed(2) : '1.00'})`,
          data: data,
          showLine: true,
          borderColor: '#06B6D4',
          backgroundColor: 'rgba(6, 182, 212, 0.15)',
          borderWidth: 2,
          fill: true
        },
        {
          label: 'Random Chance Baseline',
          data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
          showLine: true,
          borderColor: '#475569',
          borderDash: [5, 5],
          borderWidth: 1,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#94A3B8' } }
      },
      scales: {
        x: {
          min: 0,
          max: 1,
          title: { display: true, text: 'False Positive Rate (FPR)', color: '#64748B' },
          grid: { color: 'rgba(51, 65, 85, 0.3)' },
          ticks: { color: '#64748B' }
        },
        y: {
          min: 0,
          max: 1,
          title: { display: true, text: 'True Positive Rate (TPR / Recall)', color: '#64748B' },
          grid: { color: 'rgba(51, 65, 85, 0.3)' },
          ticks: { color: '#64748B' }
        }
      }
    }
  });
}

// Feature Importance Horizontal Bar Chart
function renderFeatureImportance(canvasId, topFeatures) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !topFeatures) return;
  const ctx = canvas.getContext('2d');

  if (window[canvasId + '_chart']) {
    window[canvasId + '_chart'].destroy();
  }

  const sorted = [...topFeatures].slice(0, 10).reverse();
  const labels = sorted.map(f => f.name.replace(/_/g, ' '));
  const values = sorted.map(f => f.percentage);

  window[canvasId + '_chart'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Gini Importance %',
        data: values,
        backgroundColor: 'rgba(6, 182, 212, 0.7)',
        borderColor: '#06B6D4',
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Importance: ${ctx.parsed.x.toFixed(2)}%`
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Feature Importance (%)', color: '#64748B' },
          grid: { color: 'rgba(51, 65, 85, 0.3)' },
          ticks: { color: '#64748B' }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#E2E8F0', font: { size: 11 } }
        }
      }
    }
  });
}
