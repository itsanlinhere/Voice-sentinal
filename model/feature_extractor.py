import numpy as np
import librosa
import scipy.signal
from utils.audio_utils import preprocess_audio

# Human-readable acoustic feature descriptions
FEATURE_DESCRIPTIONS = {
    "mfcc": "Mel-Frequency Cepstral Coefficients capture the phonetic vocal tract filter envelope and timbre.",
    "mel_spectrogram": "Energy distribution across perceptual frequency sub-bands, revealing synthetic harmonic gaps or smoothness.",
    "stft": "Short-Time Fourier Transform statistics and octave-band spectral contrast between peaks and valleys.",
    "pitch": "Fundamental frequency (F0) trajectory, vocal fold vibration rate, and intonation limits.",
    "jitter": "Micro-instability in pitch period length; human vocal cords exhibit natural micro-variations.",
    "shimmer": "Cycle-to-cycle amplitude perturbation reflecting vocal fold biomechanical dynamics.",
    "hnr": "Harmonics-to-Noise Ratio measuring periodic voiced harmonics relative to aperiodic aspiration noise.",
    "prosody": "Rhythmic dynamics including energy modulation, voiced/unvoiced ratios, and natural speech pauses.",
    "spectral": "Spectral distribution markers including brightness (centroid), bandwidth, rolloff, and zero crossings."
}

def extract_pitch_f0(y, sr):
    """
    Extracts fundamental frequency (F0) using librosa.pyin (Probabilistic YIN).
    Returns voiced F0 values, voiced flags, and time stamps.
    """
    fmin = librosa.note_to_hz('C2')  # ~65 Hz
    fmax = librosa.note_to_hz('C7')  # ~2093 Hz
    frame_length = 2048
    hop_length = 512

    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length
        )
    except Exception:
        f0 = np.array([])
        voiced_flag = np.zeros(1, dtype=bool)

    # Filter out NaNs
    valid_f0 = f0[~np.isnan(f0)] if f0 is not None and len(f0) > 0 else np.array([])
    return valid_f0, f0, voiced_flag

def calculate_jitter(valid_f0):
    """
    Calculates relative pitch-period variation (Jitter local and RAP).
    """
    if len(valid_f0) < 4:
        return 0.0, 0.0

    # Periods = 1 / F0 in seconds
    periods = 1.0 / valid_f0
    diffs = np.abs(np.diff(periods))
    mean_period = np.mean(periods)

    if mean_period <= 0:
        return 0.0, 0.0

    # Jitter (local): relative average difference between consecutive periods
    jitter_local = float(np.mean(diffs) / mean_period)

    # Jitter (RAP - Relative Average Perturbation over 3 periods)
    if len(periods) >= 3:
        rap_diffs = []
        for i in range(1, len(periods) - 1):
            smooth = (periods[i-1] + periods[i] + periods[i+1]) / 3.0
            rap_diffs.append(abs(periods[i] - smooth))
        jitter_rap = float(np.mean(rap_diffs) / mean_period) if rap_diffs else jitter_local
    else:
        jitter_rap = jitter_local

    # Bound reasonable values and prevent NaN
    jitter_local = min(max(jitter_local, 0.0), 0.5)
    jitter_rap = min(max(jitter_rap, 0.0), 0.5)
    return jitter_local, jitter_rap

def calculate_shimmer(y, valid_f0, sr):
    """
    Calculates relative amplitude perturbation (Shimmer local and APQ3).
    """
    if len(valid_f0) < 4 or len(y) < 1024:
        return 0.0, 0.0

    # Extract cycle peak amplitudes
    hop = 512
    rms_frames = librosa.feature.rms(y=y, hop_length=hop)[0]
    valid_amps = rms_frames[rms_frames > 1e-4]

    if len(valid_amps) < 4:
        return 0.0, 0.0

    mean_amp = np.mean(valid_amps)
    if mean_amp <= 0:
        return 0.0, 0.0

    diffs = np.abs(np.diff(valid_amps))
    shimmer_local = float(np.mean(diffs) / mean_amp)

    # APQ3 (3-point Amplitude Perturbation Quotient)
    apq3_diffs = []
    for i in range(1, len(valid_amps) - 1):
        smooth = (valid_amps[i-1] + valid_amps[i] + valid_amps[i+1]) / 3.0
        apq3_diffs.append(abs(valid_amps[i] - smooth))
    shimmer_apq3 = float(np.mean(apq3_diffs) / mean_amp) if apq3_diffs else shimmer_local

    shimmer_local = min(max(shimmer_local, 0.0), 1.0)
    shimmer_apq3 = min(max(shimmer_apq3, 0.0), 1.0)
    return shimmer_local, shimmer_apq3

def calculate_hnr(y, sr):
    """
    Harmonics-to-Noise Ratio (HNR) via Harmonic-Percussive Source Separation.
    Returns mean and standard deviation in dB.
    """
    try:
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        h_power = np.maximum(y_harmonic**2, 1e-10)
        n_power = np.maximum(y_percussive**2, 1e-10)
        hnr_db = 10.0 * np.log10(h_power / n_power)
        # Filter extreme values
        hnr_clean = np.clip(hnr_db, -20.0, 40.0)
        return float(np.mean(hnr_clean)), float(np.std(hnr_clean))
    except Exception:
        return 12.0, 3.0

def extract_features(audio_path_or_array, sr=16000, return_visuals=False):
    """
    Extracts 88 fixed, rigorously engineered acoustic features from an audio file or signal.
    Guarantees fixed feature ordering, no NaNs/Infs, and reproducible scaling.

    Returns:
        feature_vector (np.ndarray): 1D array of length 88
        feature_names (list[str]): Names of each feature in exact order
        feature_details (dict): Structured breakdown by acoustic category with human explanations
        visual_data (dict, optional): Downsampled waveform, mel spectrogram grid, pitch contour
    """
    if isinstance(audio_path_or_array, str):
        y, sample_rate, duration, meta = preprocess_audio(audio_path_or_array, target_sr=sr)
    else:
        y = np.nan_to_num(audio_path_or_array, nan=0.0).astype(np.float32)
        sample_rate = sr
        duration = float(len(y)) / float(sr)

    features = {}

    # ----------------------------------------------------
    # A. MFCC (13 coefficients -> 13 means + 13 stds = 26 features)
    # ----------------------------------------------------
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
    for i in range(13):
        coeff_name = f"mfcc_{i+1:02d}"
        features[f"{coeff_name}_mean"] = float(np.mean(mfcc[i]))
        features[f"{coeff_name}_std"] = float(np.std(mfcc[i]))

    # ----------------------------------------------------
    # B. Mel Spectrogram (8 frequency band summaries + global = 18 features)
    # ----------------------------------------------------
    melspec = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=128, n_fft=2048, hop_length=512)
    log_melspec = librosa.power_to_db(melspec, ref=np.max)

    # Summarize 128 mel filters into 8 frequency macro-bands (16 mel filters each)
    for b in range(8):
        band_slice = log_melspec[b*16:(b+1)*16, :]
        features[f"mel_band_{b+1:02d}_mean"] = float(np.mean(band_slice))
        features[f"mel_band_{b+1:02d}_std"] = float(np.std(band_slice))

    features["mel_global_mean"] = float(np.mean(log_melspec))
    features["mel_global_std"] = float(np.std(log_melspec))

    # ----------------------------------------------------
    # C. STFT (Magnitude stats + 7 spectral contrast bands = 16 features)
    # ----------------------------------------------------
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    features["stft_mag_mean"] = float(np.mean(stft))
    features["stft_mag_std"] = float(np.std(stft))

    spec_contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate, n_bands=6)
    # spectral_contrast returns 7 bands (6 bands + 1 base)
    for b in range(7):
        features[f"spectral_contrast_band_{b+1}_mean"] = float(np.mean(spec_contrast[b]))
        features[f"spectral_contrast_band_{b+1}_std"] = float(np.std(spec_contrast[b]))

    # ----------------------------------------------------
    # D. Pitch / Fundamental Frequency F0 (5 features)
    # ----------------------------------------------------
    valid_f0, full_f0, voiced_flags = extract_pitch_f0(y, sample_rate)
    if len(valid_f0) > 0:
        f0_mean = float(np.mean(valid_f0))
        f0_std = float(np.std(valid_f0))
        f0_min = float(np.min(valid_f0))
        f0_max = float(np.max(valid_f0))
        f0_range = float(f0_max - f0_min)
    else:
        # Fallback when pitch detection finds unvoiced/whisper audio
        f0_mean = 0.0
        f0_std = 0.0
        f0_min = 0.0
        f0_max = 0.0
        f0_range = 0.0

    features["f0_mean"] = f0_mean
    features["f0_std"] = f0_std
    features["f0_min"] = f0_min
    features["f0_max"] = f0_max
    features["f0_range"] = f0_range

    # ----------------------------------------------------
    # E. Jitter (2 features)
    # ----------------------------------------------------
    jitter_local, jitter_rap = calculate_jitter(valid_f0)
    features["jitter_local"] = jitter_local
    features["jitter_rap"] = jitter_rap

    # ----------------------------------------------------
    # F. Shimmer (2 features)
    # ----------------------------------------------------
    shimmer_local, shimmer_apq3 = calculate_shimmer(y, valid_f0, sample_rate)
    features["shimmer_local"] = shimmer_local
    features["shimmer_apq3"] = shimmer_apq3

    # ----------------------------------------------------
    # G. HNR (2 features)
    # ----------------------------------------------------
    hnr_mean, hnr_std = calculate_hnr(y, sample_rate)
    features["hnr_mean_db"] = hnr_mean
    features["hnr_std_db"] = hnr_std

    # ----------------------------------------------------
    # H. Prosody (7 features)
    # ----------------------------------------------------
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    features["rms_energy_mean"] = float(np.mean(rms))
    features["rms_energy_std"] = float(np.std(rms))
    features["rms_energy_range"] = float(np.max(rms) - np.min(rms)) if len(rms) > 0 else 0.0

    # Pitch coefficient of variation (prosodic inflection)
    features["f0_cv"] = float(f0_std / f0_mean) if f0_mean > 0 else 0.0

    total_frames = len(voiced_flags) if voiced_flags is not None and len(voiced_flags) > 0 else 1
    voiced_count = int(np.sum(voiced_flags)) if voiced_flags is not None else 0
    features["voiced_fraction"] = float(voiced_count) / float(total_frames)
    features["unvoiced_fraction"] = 1.0 - features["voiced_fraction"]

    # Pause ratio (frames with RMS < 10% of mean RMS)
    pause_threshold = 0.15 * np.mean(rms) if len(rms) > 0 else 0.01
    pause_frames = int(np.sum(rms < pause_threshold))
    features["pause_ratio"] = float(pause_frames) / float(len(rms)) if len(rms) > 0 else 0.0

    # ----------------------------------------------------
    # I. Spectral Features (10 features)
    # ----------------------------------------------------
    cent = librosa.feature.spectral_centroid(y=y, sr=sample_rate, hop_length=512)[0]
    features["spectral_centroid_mean"] = float(np.mean(cent))
    features["spectral_centroid_std"] = float(np.std(cent))

    bw = librosa.feature.spectral_bandwidth(y=y, sr=sample_rate, hop_length=512)[0]
    features["spectral_bandwidth_mean"] = float(np.mean(bw))
    features["spectral_bandwidth_std"] = float(np.std(bw))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sample_rate, roll_percent=0.85, hop_length=512)[0]
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))

    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=512)[0]
    features["zero_crossing_rate_mean"] = float(np.mean(zcr))
    features["zero_crossing_rate_std"] = float(np.std(zcr))

    flatness = librosa.feature.spectral_flatness(y=y, hop_length=512)[0]
    features["spectral_flatness_mean"] = float(np.mean(flatness))
    features["spectral_flatness_std"] = float(np.std(flatness))

    # Clean any accidental NaNs / Infs into 0.0
    for k, v in features.items():
        if np.isnan(v) or np.isinf(v):
            features[k] = 0.0

    # Fixed feature order
    feature_names = sorted(list(features.keys()))
    feature_vector = np.array([features[name] for name in feature_names], dtype=np.float32)

    # Acoustic feature groups for dashboard visualization
    feature_details = {
        "MFCC": {
            "title": "Mel-Frequency Cepstral Coefficients",
            "description": FEATURE_DESCRIPTIONS["mfcc"],
            "summary": f"Mean C1: {features['mfcc_01_mean']:.2f}, C2: {features['mfcc_02_mean']:.2f}, Std C1: {features['mfcc_01_std']:.2f}",
            "values": {f"mfcc_{i+1:02d}": {"mean": round(features[f"mfcc_{i+1:02d}_mean"], 2), "std": round(features[f"mfcc_{i+1:02d}_std"], 2)} for i in range(13)}
        },
        "Mel Spectrogram": {
            "title": "Mel-Scale Spectral Bands",
            "description": FEATURE_DESCRIPTIONS["mel_spectrogram"],
            "summary": f"Global Energy: {features['mel_global_mean']:.1f} dB (std: {features['mel_global_std']:.1f})",
            "values": {f"Band {b+1}": {"mean_db": round(features[f"mel_band_{b+1:02d}_mean"], 1), "std": round(features[f"mel_band_{b+1:02d}_std"], 1)} for b in range(8)}
        },
        "STFT": {
            "title": "Short-Time Fourier Transform & Contrast",
            "description": FEATURE_DESCRIPTIONS["stft"],
            "summary": f"STFT Magnitude: {features['stft_mag_mean']:.3f} | Octave Contrast: {features['spectral_contrast_band_1_mean']:.1f} dB",
            "values": {"magnitude_mean": round(features["stft_mag_mean"], 3), "contrast_bands": [round(features[f"spectral_contrast_band_{b+1}_mean"], 1) for b in range(7)]}
        },
        "Pitch / F0": {
            "title": "Fundamental Frequency (F0)",
            "description": FEATURE_DESCRIPTIONS["pitch"],
            "summary": f"Mean F0: {features['f0_mean']:.1f} Hz (Range: {features['f0_min']:.1f} - {features['f0_max']:.1f} Hz)",
            "values": {"mean_hz": round(features["f0_mean"], 1), "std_hz": round(features["f0_std"], 1), "min_hz": round(features["f0_min"], 1), "max_hz": round(features["f0_max"], 1), "range_hz": round(features["f0_range"], 1)}
        },
        "Jitter": {
            "title": "Pitch-Period Perturbation",
            "description": FEATURE_DESCRIPTIONS["jitter"],
            "summary": f"Local Jitter: {features['jitter_local']*100:.2f}% | RAP Jitter: {features['jitter_rap']*100:.2f}%",
            "values": {"local_pct": round(features["jitter_local"] * 100, 3), "rap_pct": round(features["jitter_rap"] * 100, 3)}
        },
        "Shimmer": {
            "title": "Cycle Amplitude Perturbation",
            "description": FEATURE_DESCRIPTIONS["shimmer"],
            "summary": f"Local Shimmer: {features['shimmer_local']*100:.2f}% | APQ3: {features['shimmer_apq3']*100:.2f}%",
            "values": {"local_pct": round(features["shimmer_local"] * 100, 3), "apq3_pct": round(features["shimmer_apq3"] * 100, 3)}
        },
        "HNR": {
            "title": "Harmonics-to-Noise Ratio",
            "description": FEATURE_DESCRIPTIONS["hnr"],
            "summary": f"Mean HNR: {features['hnr_mean_db']:.1f} dB (std: {features['hnr_std_db']:.1f} dB)",
            "values": {"mean_db": round(features["hnr_mean_db"], 1), "std_db": round(features["hnr_std_db"], 1)}
        },
        "Prosody": {
            "title": "Vocal Prosody & Dynamics",
            "description": FEATURE_DESCRIPTIONS["prosody"],
            "summary": f"Voiced Ratio: {features['voiced_fraction']*100:.1f}% | Pause Ratio: {features['pause_ratio']*100:.1f}% | Pitch CV: {features['f0_cv']:.2f}",
            "values": {"voiced_pct": round(features["voiced_fraction"] * 100, 1), "pause_pct": round(features["pause_ratio"] * 100, 1), "energy_mean": round(features["rms_energy_mean"], 3), "pitch_cv": round(features["f0_cv"], 3)}
        },
        "Spectral Features": {
            "title": "Spectral Distribution Markers",
            "description": FEATURE_DESCRIPTIONS["spectral"],
            "summary": f"Centroid: {features['spectral_centroid_mean']:.0f} Hz | Bandwidth: {features['spectral_bandwidth_mean']:.0f} Hz | Rolloff: {features['spectral_rolloff_mean']:.0f} Hz",
            "values": {"centroid_hz": round(features["spectral_centroid_mean"], 1), "bandwidth_hz": round(features["spectral_bandwidth_mean"], 1), "rolloff_hz": round(features["spectral_rolloff_mean"], 1), "zcr": round(features["zero_crossing_rate_mean"], 4), "flatness": round(features["spectral_flatness_mean"], 4)}
        }
    }

    if not return_visuals:
        return feature_vector, feature_names, feature_details

    # Visualizations: downsampled waveform, pitch contour, mel spectrogram grid
    # Waveform (150 points for fast, crisp Chart.js rendering)
    num_wave_points = 150
    step = max(1, len(y) // num_wave_points)
    wave_sample = [round(float(y[idx]), 3) for idx in range(0, len(y), step)[:num_wave_points]]

    # Pitch contour (voiced points with time)
    hop_length = 512
    times = librosa.frames_to_time(np.arange(len(full_f0)), sr=sample_rate, hop_length=hop_length)
    pitch_contour = []
    for t, p in zip(times, full_f0):
        if not np.isnan(p) and p > 0:
            pitch_contour.append({"time": round(float(t), 2), "f0": round(float(p), 1)})
    # Downsample pitch contour if large
    if len(pitch_contour) > 80:
        p_step = max(1, len(pitch_contour) // 80)
        pitch_contour = pitch_contour[::p_step]

    # Mel Spectrogram downsampled preview grid (24 frequency bands x 40 time steps)
    mel_resized = scipy.signal.resample(log_melspec, 24, axis=0)
    mel_resized = scipy.signal.resample(mel_resized, 40, axis=1)
    mel_grid = np.round(mel_resized, 1).tolist()

    visual_data = {
        "waveform": wave_sample,
        "pitch_contour": pitch_contour,
        "mel_grid": mel_grid,
        "duration": round(duration, 2),
        "sample_rate": sample_rate
    }

    return feature_vector, feature_names, feature_details, visual_data
