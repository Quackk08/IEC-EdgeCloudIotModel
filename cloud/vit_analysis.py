"""
Vision Transformer analysis module.
"""

import logging
from typing import List, Dict, Any, Optional

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    tf = None
    keras = None

from config import VIT_CONFIG, ACTIVITY_CLASSES

logger = logging.getLogger(__name__)


class ViTAnalyzer:
    """Vision Transformer based activity analyzer."""

    def __init__(self):
        self.model = None
        self.model_name = VIT_CONFIG['model_variant']
        self.classes = ACTIVITY_CLASSES
        self._loaded = False
        self.load_model()

    def load_model(self):
        """Load or initialize the ViT model."""
        if self._loaded:
            return

        if keras is not None:
            try:
                # Placeholder model: actual ViT load/finetune logic goes here
                self.model = self._build_placeholder_model()
                logger.info(f"Loaded placeholder ViT model: {self.model_name}")
            except Exception as exc:
                logger.warning(f"Failed to build ViT model: {exc}")
                self.model = None
        else:
            logger.warning("TensorFlow not installed; using rule-based ViT placeholder")
            self.model = None

        self._loaded = True

    def _build_placeholder_model(self):
        """Build a placeholder Keras model for API compatibility."""
        if keras is None:
            raise RuntimeError("Keras is unavailable")

        model = keras.Sequential([
            keras.layers.Input(shape=(*VIT_CONFIG['input_size'], 3)),
            keras.layers.Rescaling(1.0 / 255.0),
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(len(self.classes), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model

    def analyze(self,
                yolo_detections: List[Dict[str, Any]],
                image_frame_hash: Optional[str] = None) -> Dict[str, Any]:
        """Analyze activity using detection data and optional frame hash."""
        if self.model is None:
            return self._rule_based_analysis(yolo_detections)

        # Placeholder: actual ViT inference would use image embeddings
        return self._rule_based_analysis(yolo_detections)

    def _rule_based_analysis(self, yolo_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not yolo_detections:
            return {'activity_type': 'idle', 'confidence': 0.80}

        has_helmet = any(d.get('class_name') == 'helmet' for d in yolo_detections)
        has_obstacle = any(d.get('class_name') == 'obstacle' for d in yolo_detections)

        if not has_helmet:
            return {'activity_type': 'danger_no_helmet', 'confidence': 0.95}

        if has_obstacle:
            return {'activity_type': 'danger_collision_risk', 'confidence': 0.85}

        return {'activity_type': 'normal_work', 'confidence': 0.90}
