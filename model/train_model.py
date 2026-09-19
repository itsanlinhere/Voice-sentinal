import os
import sys
import json
import datetime
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from utils.audio_utils import scan_dataset
from model.feature_extractor import extract_features

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train_model(dataset_root="dataset", test_speaker="speaker_03", test_engine="engine_03", run_generalization=True):
    """
    Scans dataset, extracts features, performs speaker-independent train/test split,
    trains StandardScaler + RandomForestClassifier, computes real evaluation metrics,
    and saves trained artifacts to models/.

    Returns:
        dict: Training summary and real evaluation results.
    """
    summary, human_files, ai_files = scan_dataset(dataset_root)

    total_valid = len(human_files) + len(ai_files)
    if total_valid < 4:
        raise ValueError(f"Dataset has only {total_valid} valid audio files. At least 4 valid files (human & AI) are required to train.")

    print(f"Starting feature extraction for {len(human_files)} human and {len(ai_files)} AI recordings...")

    X_list = []
    y_list = []
    groups_list = [] # Speaker ID or Engine ID
    sources_list = []
    file_paths = []
    feature_names = None

    # Process Human Files (Label = 0)
    for h in human_files:
        try:
            fv, fn, _ = extract_features(h["path"])
            if feature_names is None:
                feature_names = fn
            X_list.append(fv)
            y_list.append(0)
            groups_list.append(h["speaker"])
            sources_list.append(h["speaker"])
            file_paths.append(h["path"])
        except Exception as e:
            print(f"Skipping corrupted human file {h['path']}: {e}")

    # Process AI Files (Label = 1)
    for a in ai_files:
        try:
            fv, fn, _ = extract_features(a["path"])
            if feature_names is None:
                feature_names = fn
            X_list.append(fv)
            y_list.append(1)
            groups_list.append(a["engine"])
            sources_list.append(a["engine"])
            file_paths.append(a["path"])
        except Exception as e:
            print(f"Skipping corrupted AI file {a['path']}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(groups_list)

    if len(X) == 0:
        raise ValueError("No valid audio files could be processed for feature extraction.")

    # ----------------------------------------------------
    # Speaker-Independent Data Splitting
    # ----------------------------------------------------
    speakers = summary["human_speakers"]
    engines = summary["ai_engines"]

    small_dataset_warning = None
    if len(speakers) < 3 or len(engines) < 2 or total_valid < 20:
        small_dataset_warning = (
            f"Dataset warning: Currently {len(speakers)} human speaker(s) and {len(engines)} AI engine(s) "
            f"with {total_valid} total samples. Statistical significance requires expanded multi-speaker data."
        )

    # Standard Speaker-Independent Split:
    # Reserve one speaker (e.g. speaker_03) and one engine (e.g. engine_03 or balanced portion) exclusively for testing
    if len(speakers) >= 2 and len(engines) >= 2:
        test_spk = test_speaker if test_speaker in speakers else speakers[-1]
        test_eng = test_engine if test_engine in engines else engines[-1]

        # Standard test set: unseen speaker + unseen engine
        is_test = (groups == test_spk) | (groups == test_eng)
        # If test set is too large or too small, ensure at least 1 of each class in test
        human_test = (groups == test_spk)
        ai_test = (groups == test_eng)
        is_test = human_test | ai_test
        test_methodology = f"Speaker-Independent Split: Holdout Human Speaker '{test_spk}', Holdout AI Engine '{test_eng}'"
    else:
        # Fallback to stratified 70/30 split with warning
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, test_idx = next(sss.split(X, y))
        is_test = np.zeros(len(y), dtype=bool)
        is_test[test_idx] = True
        test_methodology = "Stratified Holdout Split (insufficient distinct speakers for independent group split)"
        if not small_dataset_warning:
            small_dataset_warning = "Insufficient distinct speaker groups; used stratified holdout split."

    X_train, X_test = X[~is_test], X[is_test]
    y_train, y_test = y[~is_test], y[is_test]

    # Ensure train and test both have at least one sample of each class
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
        train_idx, test_idx = next(sss.split(X, y))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_methodology = "Stratified Split (adjusted to maintain both classes in test partition)"

    # ----------------------------------------------------
    # Model Training: StandardScaler + RandomForestClassifier
    # ----------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)

    # ----------------------------------------------------
    # Real Evaluation on Unseen Test Set
    # ----------------------------------------------------
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        roc_auc = None

    cm = confusion_matrix(y_test, y_pred).tolist()
    # cm is [[TN, FP], [FN, TP]] where 0=Human, 1=AI
    # TN: Human predicted Human
    # FP: Human predicted AI
    # FN: AI predicted Human
    # TP: AI predicted AI

    # Compute ROC Curve coordinates
    roc_points = []
    try:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        for fp, tp in zip(fpr, tpr):
            roc_points.append({"fpr": round(float(fp), 3), "tpr": round(float(tp), 3)})
    except Exception:
        roc_points = [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}]

    # Top Feature Importances
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_features = []
    for idx in sorted_idx[:15]:
        top_features.append({
            "rank": len(top_features) + 1,
            "name": feature_names[idx],
            "importance": round(float(importances[idx]), 4),
            "percentage": round(float(importances[idx] * 100), 2)
        })

    # ----------------------------------------------------
    # Optional AI Engine Generalization Experiment
    # ----------------------------------------------------
    generalization_results = None
    if run_generalization and len(engines) >= 3:
        # Train on engine_01, engine_02; test on unseen engine_03
        gen_test_engine = engines[-1]
        gen_train_engines = engines[:-1]

        gen_train_mask = (y == 0) | np.isin(groups, gen_train_engines)
        gen_test_mask = (groups == gen_test_engine)

        if np.sum(gen_train_mask) > 5 and np.sum(gen_test_mask) > 1:
            X_g_train = X[gen_train_mask]
            y_g_train = y[gen_train_mask]
            X_g_test = X[gen_test_mask]
            y_g_test = y[gen_test_mask] # All label 1

            g_scaler = StandardScaler()
            X_g_train_s = g_scaler.fit_transform(X_g_train)
            X_g_test_s = g_scaler.transform(X_g_test)

            g_clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
            g_clf.fit(X_g_train_s, y_g_train)

            g_preds = g_clf.predict(X_g_test_s)
            g_probas = g_clf.predict_proba(X_g_test_s)[:, 1]
            g_detected_ai_count = int(np.sum(g_preds == 1))
            g_recall = float(g_detected_ai_count / len(y_g_test))

            generalization_results = {
                "unseen_tested_engine": gen_test_engine,
                "training_engines": gen_train_engines,
                "test_samples_count": len(y_g_test),
                "detected_as_ai_count": g_detected_ai_count,
                "unseen_engine_detection_rate": round(g_recall * 100, 2),
                "mean_ai_probability": round(float(np.mean(g_probas)) * 100, 2),
                "interpretation": (
                    f"Trained on {len(gen_train_engines)} AI engine(s) and evaluated on completely unseen '{gen_test_engine}'. "
                    f"Successfully flagged {g_detected_ai_count}/{len(y_g_test)} ({g_recall*100:.1f}%) unseen synthetic recordings as AI."
                )
            }

    # ----------------------------------------------------
    # Save Model Artifacts
    # ----------------------------------------------------
    model_path = os.path.join(MODELS_DIR, "voice_detector.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    feature_names_path = os.path.join(MODELS_DIR, "feature_names.json")
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")

    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)

    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    metadata = {
        "status": "TRAINED",
        "training_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "classifier_name": "RandomForestClassifier(n_estimators=100, class_weight='balanced')",
        "feature_count": len(feature_names),
        "total_dataset_samples": total_valid,
        "total_human_samples": len(human_files),
        "total_ai_samples": len(ai_files),
        "speakers_count": len(speakers),
        "engines_count": len(engines),
        "speakers_list": speakers,
        "engines_list": engines,
        "train_samples_count": int(len(y_train)),
        "test_samples_count": int(len(y_test)),
        "train_human_count": int(np.sum(y_train == 0)),
        "train_ai_count": int(np.sum(y_train == 1)),
        "test_human_count": int(np.sum(y_test == 0)),
        "test_ai_count": int(np.sum(y_test == 1)),
        "test_methodology": test_methodology,
        "warning": small_dataset_warning,
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "accuracy_pct": round(acc * 100, 2),
            "precision_pct": round(prec * 100, 2),
            "recall_pct": round(rec * 100, 2),
            "f1_score_pct": round(f1 * 100, 2),
            "roc_auc_pct": round(roc_auc * 100, 2) if roc_auc is not None else None
        },
        "confusion_matrix": {
            "matrix": cm,
            "true_negatives_human_as_human": cm[0][0] if len(cm) > 0 else 0,
            "false_positives_human_as_ai": cm[0][1] if len(cm) > 0 else 0,
            "false_negatives_ai_as_human": cm[1][0] if len(cm) > 1 else 0,
            "true_positives_ai_as_ai": cm[1][1] if len(cm) > 1 else 0
        },
        "roc_curve": roc_points,
        "top_features": top_features,
        "generalization_experiment": generalization_results
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Model training successfully completed!")
    print(f"Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%")
    return metadata

if __name__ == "__main__":
    train_model()
