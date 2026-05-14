"""
Autoencoder anomaly detection module.
"""

import logging
from typing import Dict, Any, Optional

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    tf = None
    keras = None

from config import AUTOENCODER_CONFIG

logger = logging.getLogger(__name__)


class AutoencoderAnomalyDetector:
    """Autoencoder-based anomaly detector."""

    def __init__(self):
        self.model = None
        self.config = AUTOENCODER_CONFIG
        self._loaded = False
        self.load_model()

    def load_model(self):
        if self._loaded:
            return

        if keras is not None:
            try:
                self.model = self._build_model()
                logger.info("Loaded placeholder Autoencoder model")
            except Exception as exc:
                logger.warning(f"Failed to build Autoencoder: {exc}")
                self.model = None
        else:
            logger.warning("TensorFlow not installed; using rule-based anomaly placeholder")
            self.model = None

        self._loaded = True

    def _build_model(self):
        input_dim = self.config['input_dim']
        latent_dim = self.config['latent_dim']

        encoder = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(self.config['hidden_dims'][0], activation='relu'),
            layers.Dense(self.config['hidden_dims'][1], activation='relu'),
            layers.Dense(latent_dim, activation='relu')
        ])

        decoder = keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(self.config['hidden_dims'][1], activation='relu'),
            layers.Dense(self.config['hidden_dims'][0], activation='relu'),
            layers.Dense(input_dim, activation='linear')
        ])

        inputs = keras.Input(shape=(input_dim,))
        encoded = encoder(inputs)
        decoded = decoder(encoded)
        autoencoder = keras.Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def detect(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            return self._rule_based_detection(sensor_data)

        features = self._build_feature_vector(sensor_data)
        reconstructed = self.model.predict(features, verbose=0)
        error = self._reconstruction_error(features, reconstructed)
        anomaly_score = float(error)

        return {'anomaly_score': anomaly_score, 'reason': 'reconstruction_error'}

    def _build_feature_vector(self, sensor_data: Dict[str, Any]):
        import numpy as np
        imu = sensor_data.get('imu', {})
        vector = [
            imu.get('acc_x', 0.0),
            imu.get('acc_y', 0.0),
            imu.get('acc_z', 0.0),
            imu.get('gyro_x', 0.0),
            imu.get('gyro_y', 0.0),
            imu.get('gyro_z', 0.0)
        ]
        padding = [0.0] * (self.config['input_dim'] - len(vector))
        features = np.array([vector + padding], dtype=np.float32)
        return features

    def _reconstruction_error(self, original, reconstructed):
        import numpy as np
        return np.mean(np.square(original - reconstructed))

    def _rule_based_detection(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        imu = sensor_data.get('imu', {})
        acc_magnitude = (imu.get('acc_x', 0.0) ** 2 +
                         imu.get('acc_y', 0.0) ** 2 +
                         imu.get('acc_z', 0.0) ** 2) ** 0.5

        anomaly_score = 0.0
        reason = 'normal'
        if acc_magnitude > 15.0:
            anomaly_score = min(1.0, (acc_magnitude - 9.81) / 20.0)
            reason = 'high_acceleration'

        return {'anomaly_score': anomaly_score, 'reason': reason}
