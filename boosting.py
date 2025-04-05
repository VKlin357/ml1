from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

from typing import Optional, Tuple

def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])

class Boosting:

    def __init__(
        self,
        dart: bool = False,
        dropout_rate: float = 0.05,
        goss: bool = False,
        goss_k: float = 0.2,
        base_model_class=DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        early_stopping_rounds: Optional[int] = None,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        bootstrap_type: Optional[str] = 'Bernoulli',
        subsample: float | int = 1.0,
        bagging_temperature: float = 1.0,
        rsm: float | int = 1.0,
        quantization_type: Optional[str] = None,
        nbins: int = 255,
    ):
        self.dart = dart
        self.dropout_rate = dropout_rate
        self.goss = goss
        self.goss_k = goss_k
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params
        self.n_estimators: int = n_estimators
        self.learning_rate: float = learning_rate
        self.models: list = []
        self.gammas: list = []
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_set = eval_set
        self.best_iteration_: Optional[int] = None
        self.best_score_: float = float("inf")
        self.history = defaultdict(list)
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        self.bootstrap_type = bootstrap_type
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.rsm = rsm
        self.quantization_type = quantization_type
        self.nbins = nbins
        self.random_state_ = np.random.RandomState(1337)
        self.selected_features_per_iter = []
        self.bin_edges_ = None


    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False):
        if self.eval_set is not None and (X_val is None or y_val is None):
            X_val, y_val = self.eval_set

        if self.quantization_type is not None:
            self.bin_edges_ = self._build_bins(X_train)

        train_predictions = np.zeros_like(y_train, dtype=float)
        val_predictions = None
        if (X_val is not None) and (y_val is not None):
            val_predictions = np.zeros_like(y_val, dtype=float)

        no_improvement_count = 0
        best_val_loss_local = float('inf')

        for i in range(self.n_estimators):
            if self.quantization_type is not None:
                X_train_transformed = self._apply_bins(X_train)
            else:
                X_train_transformed = X_train

            if (X_val is not None) and (y_val is not None):
                if self.quantization_type is not None:
                    X_val_transformed = self._apply_bins(X_val)
                else:
                    X_val_transformed = X_val
            else:
                X_val_transformed = None

            dropped_trees = []
            if self.dart and (len(self.models) > 0):
                dropped_indices = []
                for idx_model in range(len(self.models)):
                    if self.random_state_.rand() < self.dropout_rate:
                        dropped_indices.append(idx_model)
                if len(dropped_indices) == 0:
                    dropped_indices = [self.random_state_.randint(len(self.models))]

                for d_idx in sorted(dropped_indices, reverse=True):
                    drop_tree, drop_feats = self.models[d_idx]
                    drop_gamma = self.gammas[d_idx]

                    pred_drop_train = drop_tree.predict(X_train_transformed[:, drop_feats])
                    train_predictions -= self.learning_rate * drop_gamma * pred_drop_train

                    if (val_predictions is not None) and (X_val_transformed is not None):
                        pred_drop_val = drop_tree.predict(X_val_transformed[:, drop_feats])
                        val_predictions -= self.learning_rate * drop_gamma * pred_drop_val

                    dropped_trees.append((drop_tree, drop_feats, drop_gamma))
                    self.models.pop(d_idx)
                    self.gammas.pop(d_idx)

            gradients = -self.loss_derivative(y_train, train_predictions)

            X_train_final, selected_feats = self._feature_sampling(X_train_transformed)
            grad_final = gradients
            sample_weight = None

            if self.goss:
                idx_sorted = np.argsort(np.abs(grad_final))[::-1]
                n = len(grad_final)
                n_big = int(np.ceil(self.goss_k * n))
                idx_big = idx_sorted[:n_big]
                idx_small_all = idx_sorted[n_big:]
                n_small_take = int(np.ceil(self.subsample * len(idx_small_all)))
                idx_small = self.random_state_.choice(idx_small_all, size=n_small_take, replace=False)
                idx_goss = np.concatenate([idx_big, idx_small])
                idx_goss.sort()

                X_train_final = X_train_final[idx_goss]
                y_boot = y_train[idx_goss]
                grad_final = grad_final[idx_goss]
                sample_weight = np.ones_like(grad_final, dtype=float)

                mask_small = np.isin(idx_goss, idx_small)
                factor = (n - n_big) / float(n_small_take)
                sample_weight[mask_small] = factor
            else:
                if self.bootstrap_type == 'Bernoulli':
                    X_train_final, y_boot, grad_final = self._bootstrap_bernoulli(X_train_final, y_train, grad_final)
                elif self.bootstrap_type == 'Bayesian':
                    X_train_final, y_boot, grad_final, sample_weight = self._bootstrap_bayesian(X_train_final, y_train, grad_final)
                else:
                    y_boot = y_train

            model = self.base_model_class(**self.base_model_params)
            model.fit(X_train_final, grad_final, sample_weight=sample_weight)

            base_pred_train = model.predict(X_train_transformed[:, selected_feats])
            gamma = self.find_optimal_gamma(y_train, train_predictions, base_pred_train)

            if self.dart and len(dropped_trees) > 0:
                k = len(dropped_trees)
                gamma /= max(1, k)

            train_predictions += self.learning_rate * gamma * base_pred_train

            if self.dart and len(dropped_trees) > 0:
                for (old_tree, old_feats, old_gamma) in dropped_trees:
                    old_gamma_new = old_gamma * (k / (k + 1))
                    self.models.append((old_tree, old_feats))
                    self.gammas.append(old_gamma_new)
                gamma /= (k + 1)

            self.models.append((model, selected_feats))
            self.gammas.append(gamma)

            train_auc = roc_auc_score(y_train, self.sigmoid(train_predictions))
            train_loss = self.loss_fn(y_train, train_predictions)
            self.history["train_roc_auc"].append(train_auc)
            self.history["train_loss"].append(train_loss)

            if (val_predictions is not None) and (X_val_transformed is not None):
                base_pred_val = model.predict(X_val_transformed[:, selected_feats])
                val_predictions += self.learning_rate * gamma * base_pred_val

                val_auc = roc_auc_score(y_val, self.sigmoid(val_predictions))
                val_loss = self.loss_fn(y_val, val_predictions)
                self.history["val_roc_auc"].append(val_auc)
                self.history["val_loss"].append(val_loss)

                print(f"Estimator {i+1}/{self.n_estimators} - "
                    f"Train AUC: {train_auc:.4f} - Train Loss: {train_loss:.4f} - "
                    f"Val AUC: {val_auc:.4f} - Val Loss: {val_loss:.4f}")

                if self.early_stopping_rounds and (self.early_stopping_rounds > 0):
                    if val_loss < best_val_loss_local:
                        best_val_loss_local = val_loss
                        no_improvement_count = 0
                        self.best_iteration_ = i
                        self.best_score_ = val_loss
                    else:
                        no_improvement_count += 1
                    if no_improvement_count >= self.early_stopping_rounds:
                        print(f"Early stopping on iteration {i+1}")
                        break
            else:
                print(f"Estimator {i+1}/{self.n_estimators} - "
                    f"Train AUC: {train_auc:.4f} - Train Loss: {train_loss:.4f}")

        self._compute_feature_importances(X_train.shape[1])

        if plot and (X_val is not None) and (y_val is not None):
            self.plot_history()

    def _compute_feature_importances(self, n_features: int):
        importances = np.zeros(n_features, dtype=float)
        for (model, feats) in self.models:
            if hasattr(model, "feature_importances_"):
                tree_importance = model.feature_importances_
                for i, f_idx in enumerate(feats):
                    importances[f_idx] += tree_importance[i]
        total = importances.sum()
        if total > 1e-12:
            importances /= total
        self.feature_importances_ = importances

    def partial_fit(self, X_train_rsm, y_train, current_predictions, selected_features):
        gradients = -self.loss_derivative(y_train, current_predictions)
        if self.goss:
            idx_sorted = np.argsort(np.abs(gradients))[::-1]
            n = len(gradients)
            n_big = int(np.ceil(self.goss_k * n))
            idx_big = idx_sorted[:n_big]
            idx_small_all = idx_sorted[n_big:]
            n_small_take = int(np.ceil(self.subsample * len(idx_small_all)))
            idx_small = np.random.choice(idx_small_all, size=n_small_take, replace=False)
            idx_goss = np.concatenate([idx_big, idx_small])
            idx_goss.sort()
            X_boot = X_train_rsm[idx_goss]
            y_boot = y_train[idx_goss]
            grad_boot = gradients[idx_goss]
            sample_weight = np.ones_like(grad_boot, dtype=float)
            mask_small = np.isin(idx_goss, idx_small)
            factor = (n - n_big) / float(n_small_take)
            sample_weight[mask_small] = factor
        else:
            gradients = -self.loss_derivative(y_train, current_predictions)
            if self.bootstrap_type == 'Bernoulli':
                X_boot, y_boot, grad_boot = self._bootstrap_bernoulli(X_train_rsm, y_train, gradients)
                sample_weight = None
            elif self.bootstrap_type == 'Bayesian':
                X_boot, y_boot, grad_boot, sample_weight = self._bootstrap_bayesian(X_train_rsm, y_train, gradients)
            else:
                X_boot, y_boot, grad_boot = X_train_rsm, y_train, gradients
                sample_weight = None
        model = self.base_model_class(**self.base_model_params)
        model.fit(X_boot, grad_boot, sample_weight=sample_weight)
        self.models.append((model, selected_features))
        base_pred = model.predict(X_train_rsm)
        gamma = self.find_optimal_gamma(y_train, current_predictions, base_pred)
        self.gammas.append(gamma)
        current_predictions += self.learning_rate * gamma * base_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.quantization_type is not None and self.bin_edges_ is not None:
            X_trans = self._apply_bins(X)
        else:
            X_trans = X
        cumulative_prediction = np.zeros(X.shape[0], dtype=float)
        for (model, feats), gamma in zip(self.models, self.gammas):
            X_slice = X_trans[:, feats]
            cumulative_prediction += self.learning_rate * gamma * model.predict(X_slice)
        proba = self.sigmoid(cumulative_prediction)
        return np.vstack([1 - proba, proba]).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [
            self.loss_fn(y, old_predictions + gamma * new_predictions) 
            for gamma in gammas
        ]
        best_gamma = gammas[np.argmin(losses)]
        return best_gamma

    def score(self, X, y) -> float:
        return score(self, X, y)

    def plot_history(self):
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_roc_auc'], label='Train ROC-AUC')
        if 'val_roc_auc' in self.history:
            plt.plot(self.history['val_roc_auc'], label='Validation ROC-AUC')
        plt.xlabel('Number of Estimators')
        plt.ylabel('ROC-AUC')
        plt.title('ROC-AUC Over Iterations')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_loss'], label='Train Loss')
        if 'val_loss' in self.history:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Log Loss')
        plt.title('Loss Over Iterations')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _feature_sampling(self, X: np.ndarray, selected_features: Optional[list[int]] = None) -> tuple[np.ndarray, list[int]]:
        n_features = X.shape[1]
        if selected_features is not None:
            return X[:, selected_features], selected_features
        if isinstance(self.rsm, float):
            if self.rsm < 1.0:
                n_select = int(np.ceil(n_features * self.rsm))
            else:
                n_select = n_features
        else:
            n_select = min(int(self.rsm), n_features)
        feats = self.random_state_.choice(n_features, size=n_select, replace=False)
        feats = np.sort(feats)
        self.selected_features_per_iter.append(feats)
        return X[:, feats], feats

    def _build_bins(self, X: np.ndarray) -> list[np.ndarray]:
        n_features = X.shape[1]
        bin_edges = []
        for j in range(n_features):
            col = X[:, j]
            if hasattr(col, "toarray"):
                col = col.toarray().ravel()
            elif hasattr(col, "todense"):
                col = col.todense().ravel()
            else:
                col = np.array(col).ravel()
            if self.quantization_type is None:
                bin_edges.append(None)
                continue
            col = np.nan_to_num(col, nan=np.nanmin(col))
            c_min, c_max = col.min(), col.max()
            if self.quantization_type == 'Uniform':
                edges = np.linspace(c_min, c_max, self.nbins + 1)
                bin_edges.append(edges)
            elif self.quantization_type == 'Quantile':
                sorted_col = np.sort(col)
                quantiles = np.linspace(0, 1, self.nbins + 1)
                edges = np.quantile(sorted_col, quantiles)
                bin_edges.append(edges)
            elif self.quantization_type == 'MinEntropy':
                sorted_col = np.sort(col)
                edges = self._build_min_entropy_bins(sorted_col, self.nbins)
                bin_edges.append(edges)
            elif self.quantization_type == 'PiecewiseEncoding':
                sorted_col = np.sort(col)
                quantiles = np.linspace(0, 1, self.nbins + 1)
                edges = np.quantile(sorted_col, quantiles)
                bin_edges.append(edges)
            else:
                bin_edges.append(None)
        return bin_edges

    def _build_min_entropy_bins(self, sorted_col: np.ndarray, nbins: int) -> np.ndarray:
        n = len(sorted_col)
        if n == 0 or nbins < 1:
            return np.array([])
        if nbins == 1:
            return np.array([sorted_col[0], sorted_col[-1]])
        min_val = sorted_col[0]
        shift = 0.0
        if min_val < 0:
            shift = -min_val
        col_shifted = sorted_col + shift
        total_sum = np.sum(col_shifted)
        if total_sum == 0:
            return np.array([sorted_col[0], sorted_col[-1]])
        col_normalized = col_shifted / total_sum
        def chunk_entropy(arr):
            return np.sum(arr * np.log(np.maximum(arr, 1e-15)))
        edges_list = []
        current_start = 0
        step = n // nbins
        for b in range(1, nbins):
            split_idx = min(n-1, current_start + step)
            edges_list.append(sorted_col[split_idx])
            current_start = split_idx
        edges = [sorted_col[0]] + edges_list + [sorted_col[-1]]
        return np.array(edges)

    def _build_piecewise_bins(self, col: np.ndarray, y: np.ndarray, nbins: int) -> np.ndarray:
        n = len(col)
        if n == 0 or nbins < 1:
            return np.array([])
        pairs = list(zip(col, y))
        pairs.sort(key=lambda p: p[0])
        sorted_col = np.array([p[0] for p in pairs])
        sorted_y   = np.array([p[1] for p in pairs])
        step = n // nbins
        edges_list = []
        current_start = 0
        for b in range(1, nbins):
            idx = min(n-1, current_start + step)
            edges_list.append(sorted_col[idx])
            current_start = idx
        edges = [sorted_col[0]] + edges_list + [sorted_col[-1]]
        return np.array(edges)

    def _apply_bins(self, X: np.ndarray) -> np.ndarray:
        if X is None:
            return None
        X_binned = np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
        for j in range(X.shape[1]):
            col = X[:, j]
            edges = self.bin_edges_[j]
            if edges is not None:
                if hasattr(col, "toarray"):
                    col = col.toarray().ravel()
                elif hasattr(col, "todense"):
                    col = col.todense().ravel()
                else:
                    col = np.array(col).ravel()
                binned_col = np.digitize(col, edges) - 1
                binned_col = np.clip(binned_col, 0, self.nbins - 1)
                X_binned[:, j] = binned_col
            else:
                if hasattr(col, "toarray"):
                    col = col.toarray().ravel()
                elif hasattr(col, "todense"):
                    col = col.todense().ravel()
                else:
                    col = np.array(col).ravel()
                X_binned[:, j] = col
        return X_binned

    def _bootstrap_bernoulli(self, X: np.ndarray, y: np.ndarray, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(self.subsample, float):
            if self.subsample < 1.0:
                mask = (np.random.rand(X.shape[0]) < self.subsample)
                return X[mask], y[mask], grad[mask]
            else:
                return X, y, grad
        else:
            subsample_size = min(int(self.subsample), X.shape[0])
            idx = np.random.choice(X.shape[0], size=subsample_size, replace=False)
            return X[idx], y[idx], grad[idx]

    def _bootstrap_bayesian(self, X: np.ndarray, y: np.ndarray, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        U = np.random.rand(X.shape[0])
        sample_weight = (-np.log(U)) ** self.bagging_temperature
        return X, y, grad, sample_weight
