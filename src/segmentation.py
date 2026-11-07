"""Customer segmentation using a 46-to-16 deep autoencoder and K-Means."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor

from src.preprocessing import build_preprocessor

AUTOENCODER_INPUT_DIM = 46
LATENT_DIM = 16


class AutoencoderKMeansSegmenter:
    """Fit a symmetric deep autoencoder and cluster its latent vectors."""

    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
        max_iter: int = 300,
    ) -> None:
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.preprocessor: Any = None
        self.autoencoder: MLPRegressor | None = None
        self.kmeans: KMeans | None = None
        self.feature_names: list[str] = []

    def fit(self, features: pd.DataFrame) -> AutoencoderKMeansSegmenter:
        self.preprocessor = build_preprocessor(features)
        transformed = np.asarray(
            self.preprocessor.fit_transform(features), dtype="float64"
        )
        if transformed.shape[1] != AUTOENCODER_INPUT_DIM:
            raise ValueError(
                "The Telco encoder expects 46 preprocessed features; "
                f"received {transformed.shape[1]}."
            )

        self.feature_names = self.preprocessor.get_feature_names_out().tolist()
        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=(32, LATENT_DIM, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=self.max_iter,
            early_stopping=True,
            random_state=self.random_state,
        )
        self.autoencoder.fit(transformed, transformed)
        latent = self._encode_transformed(transformed)
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.random_state,
        ).fit(latent)
        return self

    def _encode_transformed(self, transformed: np.ndarray) -> np.ndarray:
        if self.autoencoder is None:
            raise RuntimeError("Segmenter must be fitted before encoding.")
        values = transformed
        for layer_index in range(2):
            values = values @ self.autoencoder.coefs_[layer_index]
            values = values + self.autoencoder.intercepts_[layer_index]
            values = np.maximum(values, 0)
        return values

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise RuntimeError("Segmenter must be fitted before transforming.")
        transformed = np.asarray(
            self.preprocessor.transform(features), dtype="float64"
        )
        return self._encode_transformed(transformed)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.kmeans is None:
            raise RuntimeError("Segmenter must be fitted before predicting.")
        return self.kmeans.predict(self.transform(features))

    def fit_predict(
        self,
        features: pd.DataFrame,
        identifiers: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, AutoencoderKMeansSegmenter]:
        self.fit(features)
        latent = self.transform(features)
        result = pd.DataFrame(
            latent,
            columns=[f"latent_{index:02d}" for index in range(LATENT_DIM)],
            index=features.index,
        )
        if identifiers is not None:
            result.insert(0, "customerID", identifiers.to_numpy())
        result["segment"] = self.kmeans.labels_
        return result, self
