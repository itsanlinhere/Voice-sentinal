import os
import json
import numpy as np
import joblib
from utils.audio_utils import preprocess_audio, MIN_DURATION_SECONDS
from model.feature_extractor import extract_features, FEATURE_DESCRIPTIONS

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "voice_detector.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

def is_model_trained():
    """Checks if all required model artifacts exist and are readable."""
    return (
        os.path.exists(MODEL_PATH) and
        os.path.exists(SCALER_PATH) and
        os.path.exists(FEATURE_NAMES_PATH) and
        os.path.exists(METADATA_PATH)
    )

def get_model_metadata():
    """Returns stored model metadata or NOT TRAINED indicator."""
    if not is_model_trained():
        return {
            "status": "NOT TRAINED",
            "message": "Model has not been trained yet. Please visit /train to train on the dataset."
        }
    try:
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        return {
            "status": "ERROR",
            "message": f"Error loading model metadata: {str(e)}"
        }

def predict_audio(audio_path, threshold=0.5):
    """
    Analyzes an audio file using the trained StandardScaler and RandomForestClassifier.
    Returns real probabilities, prediction, confidence, acoustic feature details, and visualizations.
    """
    if not is_model_trained():
        raise RuntimeError("MODEL NOT TRAINED: Please train the model before analyzing audio.")

    # Load artifacts
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURE_NAMES_PATH, "r") as f:
        feature_names = json.load(f)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    # Preprocess and extract features with visualizations
    fv, names, feature_details, visual_data = extract_features(audio_path, return_visuals=True)

    # Scale feature vector
    X_scaled = scaler.transform([fv])

    # True Model Probabilities
    probas = clf.predict_proba(X_scaled)[0]
    p_human = float(probas[0])
    p_ai = float(probas[1])

    # Configurable decision threshold (default 0.5)
    prediction = "AI-GENERATED" if p_ai >= threshold else "HUMAN"
    confidence = float(max(p_human, p_ai))

    # Top feature contributions from trained model
    top_features = metadata.get("top_features", [])
    feature_dict = dict(zip(names, fv))

    enriched_top_features = []
    for tf in top_features[:10]:
        name = tf["name"]
        val = feature_dict.get(name, 0.0)
        enriched_top_features.append({
            "rank": tf["rank"],
            "name": name,
            "importance_pct": tf["percentage"],
            "measured_value": round(float(val), 4)
        })

    result = {
        "status": "success",
        "file_name": os.path.basename(audio_path),
        "duration_seconds": visual_data["duration"],
        "sample_rate": visual_data["sample_rate"],
        "prediction": prediction,
        "ai_probability": round(p_ai * 100, 2),
        "human_probability": round(p_human * 100, 2),
        "confidence": round(confidence * 100, 2),
        "threshold_used": threshold,
        "decision_rationale": (
            f"Trained Random Forest classified the acoustic profile with P(AI) = {p_ai*100:.1f}%, "
            f"P(Human) = {p_human*100:.1f}%. Decision threshold: {threshold*100:.0f}%."
        ),
        "acoustic_features": feature_details,
        "top_contributing_features": enriched_top_features,
        "visual_data": visual_data,
        "scientific_disclaimer": (
            "Voice Sentinel provides a machine-learning assessment, not absolute proof of authenticity. "
            "Performance depends on recording quality, compression, and the acoustic diversity of the training dataset."
        )
    }

    return result
