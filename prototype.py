import os
import glob
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

BASE_DIR = r"G:\Kaggle competition\Alpha Radar Solana Sprint"
DATA_DIR = os.path.join(BASE_DIR, "Dataset", "alpha-radar-solana-sprint")
TARGET_PATH = os.path.join(BASE_DIR, "Dataset", "target_tokens.csv")

def parse_timestamp(ts: str) -> float:
    if isinstance(ts, float) and np.isnan(ts):
        return np.nan
    if not isinstance(ts, str):
        return np.nan
    try:
        minutes, seconds = ts.split(":")
        return int(minutes) * 60 + float(seconds)
    except Exception:
        return np.nan

def load_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "index" in df.columns:
        df = df.drop(columns=["index"])
    df["timestamp_seconds"] = df["timestamp"].apply(parse_timestamp).astype("float32")
    return df

def build_features(events: pd.DataFrame) -> pd.DataFrame:
    ordered = events.sort_values(["mint_token_id", "timestamp_seconds"]).reset_index(drop=True)
    numeric_cols = ordered.select_dtypes(include=[np.number]).columns.tolist()
    group = ordered.groupby("mint_token_id", sort=False)
    agg = group[numeric_cols].agg(["mean", "std", "min", "max", "last"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg["event_count"] = group.size().astype("int32")
    holder_unique = group["holder"].nunique().rename("unique_holders")
    agg = agg.join(holder_unique, how="left")
    trade_counts = ordered.pivot_table(index="mint_token_id", columns="trade_mode", values="timestamp_seconds", aggfunc="count", fill_value=0)
    trade_counts.columns = [f"trade_mode_{c}_count" for c in trade_counts.columns]
    agg = agg.join(trade_counts, how="left")
    creator_counts = group["creator"].nunique().rename("unique_creators")
    agg = agg.join(creator_counts, how="left")
    agg = agg.fillna(0.0)
    return agg

def main():
    sample_path = os.path.join(DATA_DIR, "Sample_Dataset.csv")
    sample_events = load_events(sample_path)
    features = build_features(sample_events)
    targets = pd.read_csv(TARGET_PATH)
    targets.columns = ["mint_token_id"]
    target_set = set(targets["mint_token_id"])
    features["is_target"] = features.index.isin(target_set).astype("int8")
    X = features.drop(columns=["is_target"])
    y = features["is_target"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pos_weight = 5.0
    model = CatBoostClassifier(
        depth=8,
        learning_rate=0.08,
        iterations=2000,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        scale_pos_weight=pos_weight,
        od_type="Iter",
        od_wait=100,
        verbose=100,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    val_prob = model.predict_proba(X_val)[:, 1]
    val_pred = (val_prob >= 0.5).astype(int)
    acc = accuracy_score(y_val, val_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, val_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_val, val_prob)
    tn, fp, fn, tp = confusion_matrix(y_val, val_pred).ravel()
    print("Accuracy:", acc)
    print("Precision:", precision, "Recall:", recall, "F1:", f1, "AUC:", auc)
    print("Confusion matrix:", tn, fp, fn, tp)
    for thr in np.linspace(0.1, 0.9, 9):
        preds = (val_prob >= thr).astype(int)
        acc_thr = accuracy_score(y_val, preds)
        precision_thr, recall_thr, f1_thr, _ = precision_recall_fscore_support(y_val, preds, average="binary", zero_division=0)
        print(f"Thr={thr:.2f} -> Acc={acc_thr:.3f}, Prec={precision_thr:.3f}, Recall={recall_thr:.3f}, F1={f1_thr:.3f}")

if __name__ == "__main__":
    main()
