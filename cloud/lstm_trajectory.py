"""
LSTM trajectory prediction module.
"""

import logging
from typing import Dict, Any, List, Optional

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    tf = None
    keras = None

from config import LSTM_CONFIG

logger = logging.getLogger(__name__)


class LSTMTrajectoryPredictor:
    """LSTM-based future trajectory predictor."""

    def __init__(self):
        self.model = None
        self.config = LSTM_CONFIG
        self._loaded = False
        self.load_model()

    def load_model(self):
        if self._loaded:
            return

        if keras is not None:
            try:
                self.model = self._build_model()
                logger.info("Loaded placeholder LSTM trajectory model")
            except Exception as exc:
                logger.warning(f"Failed to build LSTM model: {exc}")
                self.model = None
        else:
            logger.warning("TensorFlow not installed; using rule-based LSTM placeholder")
            self.model = None

        self._loaded = True

    def _build_model(self):
        input_shape = (self.config['sequence_length'], self.config['input_dim'])
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(self.config['hidden_dim'], return_sequences=False),
            layers.Dense(self.config['output_dim'] * self.config['prediction_horizon'])
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict(self,
                location: Dict[str, float],
                velocity: Dict[str, float]) -> Dict[str, Any]:
        if self.model is None:
            return self._rule_based_prediction(location, velocity)

        # Build dummy sequence from current state
        sequence = self._build_sequence(location, velocity)
        predicted = self.model.predict(sequence, verbose=0)
        trajectory = self._decode_prediction(predicted[0])
        collision_risk = self._estimate_risk(velocity)

        return {'trajectory': trajectory, 'collision_risk': collision_risk}

    def _build_sequence(self,
                        location: Dict[str, float],
                        velocity: Dict[str, float]) -> Any:
        import numpy as np
        seq = np.tile([
            location['x'], location['y'], location['z'],
            velocity['vx'], velocity['vy'], velocity['vz']
        ], (self.config['sequence_length'], 1))
        return seq.reshape(1, self.config['sequence_length'], self.config['input_dim'])

    def _decode_prediction(self, predicted_vector: List[float]) -> List[Dict[str, float]]:
        trajectory = []
        for i in range(self.config['prediction_horizon']):
            idx = i * self.config['output_dim']
            trajectory.append({
                'x': float(predicted_vector[idx]),
                'y': float(predicted_vector[idx + 1]),
                'z': float(predicted_vector[idx + 2])
            })
        return trajectory

    def _estimate_risk(self, velocity: Dict[str, float]) -> float:
        risk = 0.0
        if abs(velocity['vx']) > 1.0 or abs(velocity['vy']) > 1.0:
            risk = min(1.0, (abs(velocity['vx']) + abs(velocity['vy'])) / 10.0)
        return risk

    def _rule_based_prediction(self,
                               location: Dict[str, float],
                               velocity: Dict[str, float]) -> Dict[str, Any]:
        trajectory = []
        for i in range(self.config['prediction_horizon']):
            t = i * 0.1
            trajectory.append({
                'x': location['x'] + velocity['vx'] * t,
                'y': location['y'] + velocity['vy'] * t,
                'z': location['z'] + velocity['vz'] * t
            })

        return {'trajectory': trajectory, 'collision_risk': self._estimate_risk(velocity)}
