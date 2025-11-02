
# app/main.py
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from .utils.config import DEFAULT_MODEL_PATH, MODELS_DIR
from .utils.data_loading import load_training_dataframe, load_inference_dataframe
from .algorithms import linear_regression, random_forest, gradient_boosting, decision_tree

FEATURES = ["weight_courier_raw", "travel_diff_times", "time_diff_label_till_shippingdate"]
TARGET = "time_in_transit"

def train(algo: str, save_path: Path):
    df = load_training_dataframe()
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if algo == "linear_regression":
        model, metrics = linear_regression.train(X_train, y_train)
        extra = {}
    elif algo == "random_forest":
        model, metrics = random_forest.train(X_train, y_train)
        extra = {}
    elif algo == "gradient_boosting":
            model, metrics, params = gradient_boosting.train(X_train, y_train, do_grid=False)
            extra = {"params": params}
    elif algo == "decision_tree":
            model, metrics = decision_tree.train(X_train, y_train)
            extra = {}

    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    print(f"[i] Trained {algo}. Saved model to: {save_path}")
    print(f"[i] Train metrics: {metrics}")
    if extra:
        print(f"[i] Extra: {extra}")

def predict(save_path: Path):
    model = joblib.load(save_path)
    df = load_inference_dataframe()
    X = df[FEATURES]
    preds = model.predict(X)
    out = pd.DataFrame({"prediction_hours": preds})
    print(out.head())
    return out

def main():
    parser = argparse.ArgumentParser(description="Predict ETA - modular project")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--algo", choices=["linear_regression", "random_forest", "gradient_boosting", "decision_tree"], default="gradient_boosting")
    p_train.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)

    p_pred = sub.add_parser("predict", help="Run inference using saved model")
    p_pred.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)

    args = parser.parse_args()
    if args.cmd == "train":
        train(args.algo, args.out)
    elif args.cmd == "predict":
        predict(args.model)

if __name__ == "__main__":
    main()
