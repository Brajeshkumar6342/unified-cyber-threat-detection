
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# ---------------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "raw" / "nids" / "UNSW_NB15_training-set.csv"
REPORT_DIR = ROOT / "reports" / "tables" / "unsw_15_models"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
THREADS = 4

REPORT_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# 2. LOAD ORIGINAL TRAINING DATA
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("UNSW-NB15: 15 MACHINE LEARNING MODEL EXPERIMENT")
print("=" * 65)

print("\nLoading original UNSW-NB15 training CSV...")

df = pd.read_csv(DATA_PATH)

if "label" not in df.columns:
    raise ValueError("Target column 'label' was not found.")

y = df["label"].astype(int)

# Exclude identifier and target-related columns.
drop_columns = ["label", "id", "attack_cat"]
X = df.drop(columns=drop_columns, errors="ignore")

print("Dataset shape:", df.shape)
print("Feature columns:", X.shape[1])
print("\nClass distribution:")
print(y.value_counts())


# ---------------------------------------------------------
# 3. SPLIT BEFORE PREPROCESSING
# ---------------------------------------------------------

print("\nSplitting official training data into train and validation...")

X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X,
    y,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("Training records:", len(X_train_raw))
print("Validation records:", len(X_val_raw))


# ---------------------------------------------------------
# 4. PREPROCESSING
# ---------------------------------------------------------

numeric_columns = X_train_raw.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X_train_raw.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

transformers = []

if numeric_columns:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_columns)
    )

if categorical_columns:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_columns)
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
    sparse_threshold=1.0,
)

print("\nFitting preprocessing on training data only...")

X_train = preprocessor.fit_transform(X_train_raw)
X_val = preprocessor.transform(X_val_raw)

# Ensure sparse CSR representation.
if not hasattr(X_train, "tocsr"):
    X_train = csr_matrix(X_train)

if not hasattr(X_val, "tocsr"):
    X_val = csr_matrix(X_val)

X_train = X_train.tocsr()
X_val = X_val.tocsr()

y_train = np.asarray(y_train)
y_val = np.asarray(y_val)

print("Processed training shape:", X_train.shape)
print("Processed validation shape:", X_val.shape)
print("Preprocessing complete.")


# ---------------------------------------------------------
# 5. DEFINE THE 15 MODELS
# ---------------------------------------------------------

models = {
    "01_Logistic_Regression": LogisticRegression(
        max_iter=1000,
        solver="saga",
        random_state=RANDOM_STATE,
        n_jobs=THREADS,
    ),

    "02_Ridge_Classifier": RidgeClassifier(),

    "03_Decision_Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),

    "04_Random_Forest": RandomForestClassifier(
        n_estimators=150,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=THREADS,
    ),

    "05_Extra_Trees": ExtraTreesClassifier(
        n_estimators=150,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=THREADS,
    ),

    "06_AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
    ),

    "07_Gradient_Boosting": GradientBoostingClassifier(
        n_estimators=80,
        learning_rate=0.1,
        max_depth=3,
        random_state=RANDOM_STATE,
    ),

    "08_Hist_Gradient_Boosting": HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
    ),

    "09_XGBoost": XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=THREADS,
        random_state=RANDOM_STATE,
        verbosity=0,
    ),

    "10_LightGBM": LGBMClassifier(
        n_estimators=150,
        learning_rate=0.1,
        num_leaves=31,
        class_weight="balanced",
        n_jobs=THREADS,
        random_state=RANDOM_STATE,
        verbosity=-1,
    ),

    "11_CatBoost": CatBoostClassifier(
        iterations=150,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        verbose=False,
        thread_count=THREADS,
        random_seed=RANDOM_STATE,
    ),

    "12_Linear_SVM": LinearSVC(
        C=1.0,
        max_iter=3000,
        random_state=RANDOM_STATE,
    ),

    "13_KNN": KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        n_jobs=THREADS,
    ),

    "14_Gaussian_Naive_Bayes": GaussianNB(),

    "15_MLP": MLPClassifier(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        max_iter=100,
        early_stopping=True,
        random_state=RANDOM_STATE,
    ),
}


# ---------------------------------------------------------
# 6. HELPER: OBTAIN MODEL SCORES
# ---------------------------------------------------------

def get_model_scores(model, X_data):
    """Return probability or decision scores for ROC-AUC and PR-AUC."""

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_data)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X_data)

    return model.predict(X_data)


# ---------------------------------------------------------
# 7. MODEL EXPERIMENT LOOP
# ---------------------------------------------------------

results = []
confusion_results = []
error_results = []

# Dense conversion is performed only when needed.
X_train_dense = None
X_val_dense = None

print("\nStarting model experiments...")

for model_name, model in models.items():

    print("\n" + "-" * 65)
    print("MODEL:", model_name)
    print("-" * 65)

    evaluation_scope = "Full training and validation sets"

    try:
        X_fit = X_train
        y_fit = y_train
        X_evaluate = X_val
        y_evaluate = y_val

        # HistGradientBoosting and GaussianNB need dense input.
        if model_name in [
            "08_Hist_Gradient_Boosting",
            "14_Gaussian_Naive_Bayes",
        ]:
            if X_train_dense is None:
                print("Converting matrices to float32 dense format...")
                X_train_dense = X_train.toarray().astype(np.float32)
                X_val_dense = X_val.toarray().astype(np.float32)

            X_fit = X_train_dense
            X_evaluate = X_val_dense

        # CatBoost uses dense float32 arrays here.
        if model_name == "11_CatBoost":
            if X_train_dense is None:
                print("Converting matrices to float32 dense format...")
                X_train_dense = X_train.toarray().astype(np.float32)
                X_val_dense = X_val.toarray().astype(np.float32)

            X_fit = X_train_dense
            X_evaluate = X_val_dense

        # KNN is computationally expensive.
        # Use a reproducible, stratified subset and disclose it.
        if model_name == "13_KNN":

            train_indices, _ = train_test_split(
                np.arange(len(y_train)),
                train_size=min(3000, len(y_train)),
                random_state=RANDOM_STATE,
                stratify=y_train,
            )

            val_indices, _ = train_test_split(
                np.arange(len(y_val)),
                train_size=min(3000, len(y_val)),
                random_state=RANDOM_STATE,
                stratify=y_val,
            )

            X_fit = X_train[train_indices].toarray().astype(np.float32)
            y_fit = y_train[train_indices]

            X_evaluate = X_val[val_indices].toarray().astype(np.float32)
            y_evaluate = y_val[val_indices]

            evaluation_scope = (
                "Sampled: 3000 training and 3000 validation records"
            )

        # ------------------ TRAIN ------------------

        start_train = time.perf_counter()

        model.fit(X_fit, y_fit)

        training_seconds = time.perf_counter() - start_train

        # ------------------ PREDICT ------------------

        start_prediction = time.perf_counter()

        predictions = model.predict(X_evaluate)
        scores = get_model_scores(model, X_evaluate)

        inference_seconds = time.perf_counter() - start_prediction

        # ------------------ METRICS ------------------

        accuracy = accuracy_score(y_evaluate, predictions)

        precision = precision_score(
            y_evaluate, predictions, zero_division=0
        )

        recall = recall_score(
            y_evaluate, predictions, zero_division=0
        )

        f1 = f1_score(
            y_evaluate, predictions, zero_division=0
        )

        roc_auc = roc_auc_score(y_evaluate, scores)

        pr_auc = average_precision_score(y_evaluate, scores)

        tn, fp, fn, tp = confusion_matrix(
            y_evaluate, predictions, labels=[0, 1]
        ).ravel()

        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0

        result = {
            "model": model_name,
            "evaluation_scope": evaluation_scope,
            "training_samples": len(y_fit),
            "validation_samples": len(y_evaluate),
            "features": X_train.shape[1],
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "fpr": fpr,
            "fnr": fnr,
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "milliseconds_per_sample": (
                inference_seconds / len(y_evaluate) * 1000
            ),
            "status": "SUCCESS",
        }

        results.append(result)

        confusion_results.append({
            "model": model_name,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        })

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"PR-AUC:    {pr_auc:.4f}")
        print(f"FPR:       {fpr:.4f}")
        print(f"FNR:       {fnr:.4f}")
        print(f"Training time:  {training_seconds:.2f} seconds")
        print(f"Inference time: {inference_seconds:.2f} seconds")

        # Save an individual classification report.
        report = classification_report(
            y_evaluate,
            predictions,
            labels=[0, 1],
            target_names=["BENIGN", "ATTACK"],
            zero_division=0,
        )

        report_path = REPORT_DIR / f"{model_name}_report.txt"
        report_path.write_text(report, encoding="utf-8")

    except Exception as error:

        print("MODEL FAILED:", str(error))

        error_results.append({
            "model": model_name,
            "error": str(error),
        })


# ---------------------------------------------------------
# 8. SAVE EXPERIMENT RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

if not results_df.empty:

    results_df = results_df.sort_values(
        by="f1",
        ascending=False,
    )

    results_df.to_csv(
        REPORT_DIR / "unsw_15_models_metrics.csv",
        index=False,
    )

    pd.DataFrame(confusion_results).to_csv(
        REPORT_DIR / "unsw_15_models_confusion_matrices.csv",
        index=False,
    )

    results_df.to_json(
        REPORT_DIR / "unsw_15_models_metrics.json",
        orient="records",
        indent=4,
    )

if error_results:
    pd.DataFrame(error_results).to_csv(
        REPORT_DIR / "unsw_15_models_errors.csv",
        index=False,
    )

# Save the exact validation preprocessing configuration.
joblib.dump(
    preprocessor,
    REPORT_DIR / "unsw_validation_preprocessor.joblib",
)

# Save experiment settings.
experiment_info = {
    "dataset": "UNSW-NB15",
    "target": "label",
    "positive_class": 1,
    "random_state": RANDOM_STATE,
    "validation_size": VALIDATION_SIZE,
    "training_samples": int(len(y_train)),
    "validation_samples": int(len(y_val)),
    "features_after_encoding": int(X_train.shape[1]),
    "models_defined": len(models),
    "models_succeeded": len(results),
    "models_failed": len(error_results),
    "note": (
        "Validation experiment. KNN uses a sampled subset. "
        "Official test data was not used."
    ),
}

with open(
    REPORT_DIR / "experiment_info.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(experiment_info, file, indent=4)


# ---------------------------------------------------------
# 9. FINAL SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("UNSW-NB15 EXPERIMENT SUMMARY")
print("=" * 65)

print("Models defined:", len(models))
print("Models successful:", len(results))
print("Models failed:", len(error_results))

if not results_df.empty:
    print("\nMODEL RANKING BY VALIDATION F1:")
    print(
        results_df[
            ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "fpr"]
        ].to_string(index=False)
    )

print("\nResults saved in:")
print(REPORT_DIR)

print("\nOfficial test data has NOT been used.")
print("Experiment completed.")
