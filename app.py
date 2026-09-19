import os
import sys
import json
import uuid
import traceback
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils.audio_utils import scan_dataset, preprocess_audio, inspect_audio_file
from model.predictor import predict_audio, is_model_trained, get_model_metadata
from model.train_model import train_model
from model.evaluator import get_evaluation_report

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "voice_sentinel_cyber_secret_2026")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB limit
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "webm"}

def is_allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.context_processor
def inject_global_status():
    """Provides model status and metadata to all Jinja templates."""
    trained = is_model_trained()
    return {
        "model_status": "TRAINED" if trained else "NOT_TRAINED",
        "is_model_trained": trained
    }

# ----------------------------------------------------------------------
# Web Views
# ----------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Main dashboard & analyzer view."""
    summary, _, _ = scan_dataset()
    return render_template(
        "index.html",
        active_page="index",
        dataset=summary,
        result=None
    )

@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyzes an uploaded audio recording or selected dataset preset."""
    if not is_model_trained():
        flash("Model is not trained yet. Please train the model before running audio analysis.", "error")
        return redirect(url_for("train_page"))

    preset_path = request.form.get("sample_preset")
    threshold = float(request.form.get("threshold", 0.5))
    target_file = None
    cleanup_needed = False
    audio_url = None

    try:
        if preset_path and os.path.exists(preset_path):
            target_file = preset_path
            audio_url = f"/audio-serve?path={preset_path}"
        elif "audio_file" in request.files:
            file = request.files["audio_file"]
            if file.filename == "":
                flash("No audio file was selected for upload.", "error")
                return redirect(url_for("index"))

            if not is_allowed_file(file.filename):
                flash("Unsupported audio format. Supported: WAV, MP3, M4A, OGG, WebM.", "error")
                return redirect(url_for("index"))

            ext = file.filename.rsplit(".", 1)[1].lower()
            safe_name = f"upload_{uuid.uuid4().hex[:10]}.{ext}"
            target_file = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
            file.save(target_file)
            cleanup_needed = True
            audio_url = f"/audio-serve?path=uploads/{safe_name}"
        else:
            flash("No valid audio file or preset provided.", "error")
            return redirect(url_for("index"))

        # Run acoustic inference
        result = predict_audio(target_file, threshold=threshold)
        result["audio_url"] = audio_url

        summary, _, _ = scan_dataset()
        return render_template(
            "index.html",
            active_page="index",
            dataset=summary,
            result=result
        )

    except ValueError as val_err:
        flash(f"Audio Validation Error: {str(val_err)}", "error")
        return redirect(url_for("index"))
    except Exception as e:
        flash(f"Forensic Analysis Error: {str(e)}", "error")
        return redirect(url_for("index"))

@app.route("/live", methods=["GET"])
def live_page():
    """Live microphone detection interface."""
    return render_template("live.html", active_page="live")

@app.route("/live-analyze", methods=["POST"])
def live_analyze():
    """Receives streaming audio chunks from browser MediaRecorder."""
    if not is_model_trained():
        return jsonify({
            "status": "error",
            "message": "MODEL NOT TRAINED. Train the model before live stream detection."
        }), 400

    if "audio" not in request.files:
        return jsonify({"status": "error", "message": "No audio chunk provided."}), 400

    audio_file = request.files["audio"]
    threshold = float(request.form.get("threshold", 0.5))

    temp_path = os.path.join("/tmp", f"live_chunk_{uuid.uuid4().hex[:8]}.wav")
    audio_file.save(temp_path)

    try:
        result = predict_audio(temp_path, threshold=threshold)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Chunk analysis error: {str(e)}"
        }), 400
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.route("/train", methods=["GET"])
def train_page():
    """Model training interface."""
    summary, _, _ = scan_dataset()
    metadata = get_model_metadata()
    return render_template(
        "train.html",
        active_page="train",
        dataset=summary,
        metadata=metadata
    )

@app.route("/train-model", methods=["POST"])
def run_train():
    """Executes model training pipeline on the dataset."""
    try:
        meta = train_model()
        flash(f"Model successfully trained! Holdout Test Accuracy: {meta['metrics']['accuracy_pct']}%, F1: {meta['metrics']['f1_score_pct']}%.", "success")
    except Exception as e:
        flash(f"Training Failed: {str(e)}", "error")

    return redirect(url_for("train_page"))

@app.route("/dataset", methods=["GET"])
def dataset_page():
    """Dataset explorer and validation summary."""
    summary, _, _ = scan_dataset()
    return render_template("dataset.html", active_page="dataset", summary=summary)

@app.route("/api/dataset/upload", methods=["POST"])
def upload_to_dataset():
    """Allows uploading new speech recordings directly into dataset folders."""
    if "dataset_file" not in request.files:
        flash("No file was uploaded.", "error")
        return redirect(url_for("dataset_page"))

    file = request.files["dataset_file"]
    target_folder = request.form.get("target_folder", "human/speaker_01")

    # Prevent path traversal
    if ".." in target_folder or target_folder.startswith("/"):
        flash("Invalid target folder specified.", "error")
        return redirect(url_for("dataset_page"))

    if not is_allowed_file(file.filename):
        flash("Invalid audio format. Allowed: WAV, MP3, M4A, OGG, WebM.", "error")
        return redirect(url_for("dataset_page"))

    folder_path = os.path.join(BASE_DIR, "dataset", target_folder)
    os.makedirs(folder_path, exist_ok=True)

    safe_name = secure_filename(file.filename)
    dest_path = os.path.join(folder_path, safe_name)
    file.save(dest_path)

    # Validate uploaded audio file
    insp = inspect_audio_file(dest_path)
    if insp["valid"]:
        flash(f"Audio file '{safe_name}' successfully added to dataset ({insp['duration']}s, {insp['sample_rate']} Hz).", "success")
    else:
        flash(f"Audio file saved, but flagged with warning: {insp.get('error', 'Unrecognized audio format')}", "error")

    return redirect(url_for("dataset_page"))

@app.route("/evaluation", methods=["GET"])
def evaluation_page():
    """Model evaluation report & benchmarks."""
    report = get_evaluation_report()
    return render_template("evaluation.html", active_page="evaluation", report=report)

@app.route("/model-info", methods=["GET"])
def model_info_page():
    """Acoustic feature specifications and technical architecture."""
    return render_template("model_info.html", active_page="model_info")

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for monitoring."""
    summary, _, _ = scan_dataset()
    return jsonify({
        "status": "ok",
        "service": "Voice Sentinel",
        "model_loaded": is_model_trained(),
        "valid_dataset_recordings": summary["total_valid_recordings"]
    })

@app.route("/audio-serve")
def audio_serve():
    """Safely serves audio files for playback previews."""
    path_param = request.args.get("path", "")
    if ".." in path_param or path_param.startswith("/"):
        return "Invalid file path", 400

    full_path = os.path.join(BASE_DIR, path_param)
    if not os.path.exists(full_path):
        return "File not found", 404

    dir_name = os.path.dirname(full_path)
    file_name = os.path.basename(full_path)
    return send_from_directory(dir_name, file_name)

# ----------------------------------------------------------------------
# Application Launch
# ----------------------------------------------------------------------
if __name__ == "__main__":
    PORT = int(os.environ.get("FLASK_PORT", 5000))
    print(f"Starting Voice Sentinel Forensic Server on 0.0.0.0:{PORT}...")
    app.run(host="0.0.0.0", port=PORT, debug=False)
