import os
import glob
import traceback
import subprocess
import numpy as np
import soundfile as sf
import librosa

TARGET_SAMPLE_RATE = 16000
MIN_DURATION_SECONDS = 0.4

def convert_to_wav(input_path, output_path, target_sr=TARGET_SAMPLE_RATE):
    """
    Converts any supported audio format (WAV, MP3, M4A, WebM, OGG) to standard mono WAV using ffmpeg.
    """
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1",
        "-ar", str(target_sr),
        output_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise ValueError(f"ffmpeg conversion failed: {result.stderr.decode('utf-8', errors='ignore')}")
    return output_path

def preprocess_audio(audio_path, target_sr=TARGET_SAMPLE_RATE):
    """
    Loads, cleans, converts to mono, resamples to target_sr, and normalizes audio.
    Returns:
        y (np.ndarray): 1D normalized float32 array
        sr (int): Sample rate (always target_sr)
        duration (float): Audio duration in seconds
        metadata (dict): Preprocessing details
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    file_size = os.path.getsize(audio_path)
    if file_size < 100:
        raise ValueError("Audio file is empty or too small to contain valid audio data.")

    # Try standard librosa load first
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    except Exception as primary_err:
        # Fallback to ffmpeg conversion for WebM, Opus, or header quirks
        temp_wav = f"/tmp/conv_{os.getpid()}_{os.path.basename(audio_path)}.wav"
        try:
            convert_to_wav(audio_path, temp_wav, target_sr=target_sr)
            y, sr = librosa.load(temp_wav, sr=target_sr, mono=True)
        except Exception as fallback_err:
            raise ValueError(f"Unable to decode audio: {str(primary_err)}")
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass

    duration = float(len(y)) / float(sr)

    if duration < MIN_DURATION_SECONDS:
        raise ValueError(f"Audio is too short for reliable analysis ({duration:.2f}s). Minimum {MIN_DURATION_SECONDS}s required.")

    # Check for complete silence
    peak = np.max(np.abs(y)) if len(y) > 0 else 0.0
    rms = np.sqrt(np.mean(y**2)) if len(y) > 0 else 0.0

    if peak < 1e-4:
        raise ValueError("Audio recording is virtually silent. Please provide an audible speech sample.")

    # Peak normalize to -1.0 to 1.0 safely without distortion
    if peak > 0:
        y = y / peak * 0.95

    # Ensure float32 and no NaN/Inf
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    metadata = {
        "original_file": os.path.basename(audio_path),
        "sample_rate": sr,
        "samples": len(y),
        "duration": round(duration, 3),
        "peak_amplitude": float(peak),
        "rms_energy": float(rms)
    }

    return y, sr, duration, metadata

def inspect_audio_file(file_path):
    """
    Safely inspects an audio file without throwing fatal exceptions.
    Returns inspection dictionary with validity flag.
    """
    info = {
        "path": file_path,
        "filename": os.path.basename(file_path),
        "size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
        "format": os.path.splitext(file_path)[1].lower().replace(".", ""),
        "valid": False,
        "sample_rate": None,
        "duration": 0.0,
        "channels": 1,
        "error": None
    }

    try:
        # First try soundfile info
        sf_info = sf.info(file_path)
        info["sample_rate"] = sf_info.samplerate
        info["duration"] = round(sf_info.duration, 3)
        info["channels"] = sf_info.channels
        info["format"] = sf_info.format.lower()
        if sf_info.duration >= MIN_DURATION_SECONDS:
            info["valid"] = True
        else:
            info["error"] = f"Duration too short ({sf_info.duration:.2f}s < {MIN_DURATION_SECONDS}s)"
    except Exception as e1:
        # Try librosa inspect
        try:
            dur = librosa.get_duration(path=file_path)
            info["duration"] = round(dur, 3)
            info["valid"] = (dur >= MIN_DURATION_SECONDS)
            if not info["valid"]:
                info["error"] = f"Duration too short ({dur:.2f}s < {MIN_DURATION_SECONDS}s)"
        except Exception as e2:
            info["valid"] = False
            info["error"] = f"Corrupted or unsupported format: {str(e1)}"

    return info

def scan_dataset(dataset_root="dataset"):
    """
    Scans dataset/human/ and dataset/ai/ recursively and dynamically.
    Returns actual counts, speaker lists, engine lists, duration stats,
    formats, sample rates, and invalid/corrupted files list.
    """
    human_dir = os.path.join(dataset_root, "human")
    ai_dir = os.path.join(dataset_root, "ai")

    human_files = []
    ai_files = []
    corrupted_files = []

    speakers = {}
    engines = {}

    formats_count = {}
    sample_rates_count = {}
    durations = []

    # Scan Human directory
    if os.path.exists(human_dir):
        for speaker_name in sorted(os.listdir(human_dir)):
            speaker_path = os.path.join(human_dir, speaker_name)
            if os.path.isdir(speaker_path):
                speakers[speaker_name] = []
                for fname in sorted(os.listdir(speaker_path)):
                    fpath = os.path.join(speaker_path, fname)
                    if os.path.isfile(fpath):
                        insp = inspect_audio_file(fpath)
                        insp["speaker"] = speaker_name
                        insp["class"] = "human"
                        if insp["valid"]:
                            human_files.append(insp)
                            speakers[speaker_name].append(insp)
                            durations.append(insp["duration"])
                            fmt = insp["format"]
                            formats_count[fmt] = formats_count.get(fmt, 0) + 1
                            sr = insp["sample_rate"]
                            if sr:
                                sample_rates_count[str(sr)] = sample_rates_count.get(str(sr), 0) + 1
                        else:
                            corrupted_files.append(insp)

    # Scan AI directory
    if os.path.exists(ai_dir):
        for engine_name in sorted(os.listdir(ai_dir)):
            engine_path = os.path.join(ai_dir, engine_name)
            if os.path.isdir(engine_path):
                engines[engine_name] = []
                for fname in sorted(os.listdir(engine_path)):
                    fpath = os.path.join(engine_path, fname)
                    if os.path.isfile(fpath):
                        insp = inspect_audio_file(fpath)
                        insp["engine"] = engine_name
                        insp["class"] = "ai"
                        if insp["valid"]:
                            ai_files.append(insp)
                            engines[engine_name].append(insp)
                            durations.append(insp["duration"])
                            fmt = insp["format"]
                            formats_count[fmt] = formats_count.get(fmt, 0) + 1
                            sr = insp["sample_rate"]
                            if sr:
                                sample_rates_count[str(sr)] = sample_rates_count.get(str(sr), 0) + 1
                        else:
                            corrupted_files.append(insp)

    dur_stats = {
        "count": len(durations),
        "mean": round(float(np.mean(durations)), 2) if durations else 0.0,
        "std": round(float(np.std(durations)), 2) if durations else 0.0,
        "min": round(float(np.min(durations)), 2) if durations else 0.0,
        "max": round(float(np.max(durations)), 2) if durations else 0.0,
    }

    total_valid = len(human_files) + len(ai_files)
    balance_ratio = round(len(human_files) / max(1, len(ai_files)), 2) if total_valid > 0 else 1.0

    summary = {
        "dataset_root": dataset_root,
        "total_valid_recordings": total_valid,
        "human_recordings": len(human_files),
        "ai_recordings": len(ai_files),
        "human_speakers_count": len(speakers),
        "ai_engines_count": len(engines),
        "human_speakers": list(speakers.keys()),
        "ai_engines": list(engines.keys()),
        "speakers_breakdown": {spk: len(files) for spk, files in speakers.items()},
        "engines_breakdown": {eng: len(files) for eng, files in engines.items()},
        "audio_formats": formats_count,
        "sample_rates": sample_rates_count,
        "duration_statistics": dur_stats,
        "corrupted_files_count": len(corrupted_files),
        "corrupted_files": corrupted_files,
        "class_balance_ratio": balance_ratio,
        "is_balanced": (0.8 <= balance_ratio <= 1.25) if total_valid > 0 else False,
        "sufficient_data": total_valid >= 10
    }

    return summary, human_files, ai_files
