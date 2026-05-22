import pickle
import os
import numpy as np
import optuna
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, precision_score, recall_score
import lightgbm as lgb
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)


def find_best_threshold(y_true, y_proba) -> tuple[float, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = f1s.argmax()
    return float(thresholds[best_idx]), float(f1s[best_idx])


def evaluate_with_threshold(model, X_test, y_test, threshold: float = 0.5) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return {
        'AUC': roc_auc_score(y_test, y_proba),
        'F1': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'threshold': threshold,
        'y_pred': y_pred,
        'y_proba': y_proba,
    }


def tune_lightgbm(X_train, X_val, X_test, y_train, y_val, y_test, n_trials: int = 100):
    """Tune LightGBM với class_weight='balanced' (không dùng SMOTE)."""
    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int('n_estimators', 200, 800),
            learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            max_depth         = trial.suggest_int('max_depth', 4, 10),
            num_leaves        = trial.suggest_int('num_leaves', 20, 150),
            min_child_samples = trial.suggest_int('min_child_samples', 10, 100),
            min_split_gain    = trial.suggest_float('min_split_gain', 1e-4, 1.0, log=True),
            subsample         = trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_alpha         = trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
            reg_lambda        = trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
            class_weight      = 'balanced',
            random_state=42, n_jobs=-1, verbose=-1,
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                             lgb.log_evaluation(-1)])
        val_proba = model.predict_proba(X_val)[:, 1]
        _, best_f1 = find_best_threshold(y_val, val_proba)
        return best_f1

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"LightGBM best val F1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    best_params = study.best_params.copy()
    best_params.update({'class_weight': 'balanced', 'random_state': 42,
                        'n_jobs': -1, 'verbose': -1})
    model_tuned = lgb.LGBMClassifier(**best_params)
    model_tuned.fit(X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(-1)])

    val_proba = model_tuned.predict_proba(X_val)[:, 1]
    best_th, _ = find_best_threshold(y_val, val_proba)
    result = evaluate_with_threshold(model_tuned, X_test, y_test, threshold=best_th)
    print(f"LightGBM tuned — AUC: {result['AUC']:.4f} | F1: {result['F1']:.4f} | threshold: {best_th:.3f}")
    return model_tuned, best_th, result


def tune_xgboost(X_train, X_val, X_test, y_train, y_val, y_test, n_trials: int = 100):
    """Tune XGBoost với scale_pos_weight là hyperparameter (không dùng SMOTE)."""
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Imbalance ratio: {ratio:.2f} → scale_pos_weight search range: [{ratio*0.3:.2f}, {ratio*3.0:.2f}]")

    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int('n_estimators', 200, 800),
            learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            max_depth        = trial.suggest_int('max_depth', 4, 10),
            subsample        = trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_alpha        = trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
            reg_lambda       = trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
            scale_pos_weight = trial.suggest_float('scale_pos_weight',
                                                   ratio * 0.3, ratio * 3.0),
            random_state=42, n_jobs=-1, eval_metric='logloss',
            early_stopping_rounds=30, verbosity=0,
        )
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_proba = model.predict_proba(X_val)[:, 1]
        _, best_f1 = find_best_threshold(y_val, val_proba)
        return best_f1

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"XGBoost best val F1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    best_params = study.best_params.copy()
    best_params.update({'random_state': 42, 'n_jobs': -1, 'eval_metric': 'logloss',
                        'early_stopping_rounds': 50, 'verbosity': 0})
    model_tuned = XGBClassifier(**best_params)
    model_tuned.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_proba = model_tuned.predict_proba(X_val)[:, 1]
    best_th, _ = find_best_threshold(y_val, val_proba)
    result = evaluate_with_threshold(model_tuned, X_test, y_test, threshold=best_th)
    print(f"XGBoost tuned — AUC: {result['AUC']:.4f} | F1: {result['F1']:.4f} | threshold: {best_th:.3f}")
    return model_tuned, best_th, result


def save_tuned_models(lgb_model, xgb_model, lgb_th: float, xgb_th: float,
                      output_dir: str = ".") -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'model_lgb_tuned.pkl'), 'wb') as f:
        pickle.dump(lgb_model, f)
    with open(os.path.join(output_dir, 'model_xgb_tuned.pkl'), 'wb') as f:
        pickle.dump(xgb_model, f)
    thresh_path = os.path.join(output_dir, 'best_thresholds.pkl')
    if os.path.exists(thresh_path):
        with open(thresh_path, 'rb') as f:
            thresholds = pickle.load(f)
    else:
        thresholds = {}
    thresholds['LightGBM'] = lgb_th
    thresholds['XGBoost'] = xgb_th
    with open(thresh_path, 'wb') as f:
        pickle.dump(thresholds, f)
    print(f"Tuned models and thresholds saved to {output_dir}")
    print(f"  LightGBM threshold: {lgb_th:.3f}")
    print(f"  XGBoost  threshold: {xgb_th:.3f}")