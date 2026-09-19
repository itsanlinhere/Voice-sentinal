import os
import json
from model.predictor import is_model_trained, get_model_metadata

def get_evaluation_report():
    """
    Returns comprehensive evaluation data based on the trained model and test partition.
    """
    if not is_model_trained():
        return {
            "status": "NOT TRAINED",
            "message": "Model has not been trained yet. Train the model to view evaluation metrics."
        }

    metadata = get_model_metadata()

    # Formatted classification report dictionary
    metrics = metadata.get("metrics", {})
    cm = metadata.get("confusion_matrix", {})

    report = {
        "status": "TRAINED",
        "classifier_name": metadata.get("classifier_name"),
        "training_timestamp": metadata.get("training_timestamp"),
        "test_methodology": metadata.get("test_methodology"),
        "test_samples_count": metadata.get("test_samples_count"),
        "train_samples_count": metadata.get("train_samples_count"),
        "metrics": metrics,
        "confusion_matrix": cm,
        "roc_curve": metadata.get("roc_curve", []),
        "top_features": metadata.get("top_features", []),
        "generalization_experiment": metadata.get("generalization_experiment"),
        "disclaimer": "Evaluation is based on the current dataset and does not guarantee performance on unseen real-world voice-cloning systems."
    }

    return report
