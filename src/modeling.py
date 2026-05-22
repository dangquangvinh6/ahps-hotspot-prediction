import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    precision_score, recall_score, precision_recall_curve,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from xgboost import XGBClassifier

from src.clustering import FEATURE_COLS


CATEGORICAL_CANDIDATES = ['Weather_enc', 'WindDir_enc', 'State_enc', 'County_enc', 'rain_intensity']


def split_data(df_final: pd.DataFrame):
    X = df_final[FEATURE_COLS]
    y = df_final['label']

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

    print(f"Features : {X.shape[1]}")
    print(f"Train    : {len(X_train):,} ({len(X_train) / len(df_final) * 100:.1f}%)")
    print(f"Val      : {len(X_val):,} ({len(X_val) / len(df_final) * 100:.1f}%)")
    print(f"Test     : {len(X_test):,} ({len(X_test) / len(df_final) * 100:.1f}%)")
    return X_train, X_val, X_test, y_train, y_val, y_test


def _build_logistic_pipeline(X_train: pd.DataFrame) -> Pipeline:
    categorical_lr = [c for c in CATEGORICAL_CANDIDATES if c in X_train.columns]
    numeric_lr = [c for c in X_train.columns if c not in categorical_lr]

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_lr),
        ('cat', categorical_transformer, categorical_lr),
    ], remainder='drop')

    return Pipeline([
        ('preprocess', preprocessor),
        ('model', LogisticRegression(
            solver='saga', penalty='l2', C=1.0,
            max_iter=3000, class_weight='balanced', random_state=42,
        )),
    ])


def _find_best_threshold(y_true, y_proba) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    return float(thresholds[f1s.argmax()])


def _eval_model(name: str, model, X_test, y_test, threshold: float = 0.5) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'threshold': threshold,
        'AUC': roc_auc_score(y_test, y_proba),
        'F1': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Accuracy': (y_pred == y_test).mean(),
    }
    print(classification_report(y_test, y_pred, target_names=['Non-hotspot', 'Hotspot']))
    print(f"  AUC-ROC: {metrics['AUC']:.4f} | F1: {metrics['F1']:.4f} | threshold: {threshold:.3f}")
    return metrics


def _soft_vote_ensemble(results: dict, X_val, y_val, X_test, y_test) -> dict:
    probas_test = np.stack([r['y_proba'] for r in results.values()])
    avg_proba_test = probas_test.mean(axis=0)

    probas_val = np.stack([
        r['model'].predict_proba(X_val)[:, 1] for r in results.values()
    ])
    avg_proba_val = probas_val.mean(axis=0)
    best_th = _find_best_threshold(y_val, avg_proba_val)

    y_pred = (avg_proba_test >= best_th).astype(int)
    return {
        'model': None,
        'y_pred': y_pred,
        'y_proba': avg_proba_test,
        'threshold': best_th,
        'AUC': roc_auc_score(y_test, avg_proba_test),
        'F1': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Accuracy': (y_pred == y_test).mean(),
    }


def train_all_models(X_train, X_val, X_test, y_train, y_val, y_test) -> dict:
    # Tính ratio imbalance cho XGBoost scale_pos_weight
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Imbalance ratio (neg/pos): {ratio:.2f} → dùng làm scale_pos_weight cho XGBoost")

    results = {}

    print("\n--- Logistic Regression ---")
    logit_pipeline = _build_logistic_pipeline(X_train)
    logit_pipeline.fit(X_train, y_train)
    lr_val_proba = logit_pipeline.predict_proba(X_val)[:, 1]
    lr_th = _find_best_threshold(y_val, lr_val_proba)
    results['Logistic Regression'] = _eval_model(
        'Logistic Regression', logit_pipeline, X_test, y_test, threshold=lr_th)

    tree_models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_features='sqrt',
            min_samples_split=10, min_samples_leaf=5,
            class_weight='balanced',
            random_state=42, n_jobs=-1,
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=8, num_leaves=63,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1,
        ),
        'XGBoost': XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            scale_pos_weight=ratio,
            random_state=42, n_jobs=-1, eval_metric='auc',
            early_stopping_rounds=50, verbosity=0,
        ),
    }

    for name, model in tree_models.items():
        print(f"\n--- {name} ---")
        if name == 'LightGBM':
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(period=-1)])
        elif name == 'XGBoost':
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        val_proba = model.predict_proba(X_val)[:, 1]
        best_th = _find_best_threshold(y_val, val_proba)
        results[name] = _eval_model(name, model, X_test, y_test, threshold=best_th)

    print("\n--- Ensemble (Soft Vote) ---")
    ensemble_result = _soft_vote_ensemble(results, X_val, y_val, X_test, y_test)
    y_pred_ens = ensemble_result['y_pred']
    print(classification_report(y_test, y_pred_ens, target_names=['Non-hotspot', 'Hotspot']))
    print(f"  AUC-ROC: {ensemble_result['AUC']:.4f} | F1: {ensemble_result['F1']:.4f} "
          f"| threshold: {ensemble_result['threshold']:.3f}")
    results['Ensemble (Soft Vote)'] = ensemble_result

    print("\n=== Summary (sorted by F1) ===")
    results_sorted = dict(sorted(results.items(), key=lambda x: x[1]['F1'], reverse=True))
    print(f"{'Model':<26} {'AUC-ROC':>8} {'F1':>8} {'Precision':>11} {'Recall':>8} {'Threshold':>10}")
    print("-" * 75)
    best_f1 = max(r['F1'] for r in results.values())
    for name, r in results_sorted.items():
        star = " *" if r['F1'] == best_f1 else ""
        print(f"  {name:<24} {r['AUC']:>8.4f} {r['F1']:>8.4f} "
              f"{r['Precision']:>11.4f} {r['Recall']:>8.4f} {r['threshold']:>10.3f}{star}")

    return results


def save_models(results: dict, output_dir: str = ".") -> None:
    import os
    os.makedirs(output_dir, exist_ok=True)
    name_map = {
        'Logistic Regression': 'logistic_regression',
        'Random Forest': 'random_forest',
        'LightGBM': 'lightgbm',
        'XGBoost': 'xgboost',
    }
    for name, r in results.items():
        if r['model'] is None:
            continue
        fname = f"model_{name_map.get(name, name.lower().replace(' ', '_'))}.pkl"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, 'wb') as f:
            pickle.dump(r['model'], f)
        print(f"Saved: {fpath}")

    thresholds = {name: r['threshold'] for name, r in results.items() if r['model'] is not None}
    thresh_path = os.path.join(output_dir, 'best_thresholds.pkl')
    with open(thresh_path, 'wb') as f:
        pickle.dump(thresholds, f)
    print(f"Saved thresholds: {thresh_path}")