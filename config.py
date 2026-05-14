"""
Project Configuration and Constants
"""

import os
from pathlib import Path

# ============================================================================
# Project Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.absolute()
EDGE_DIR = PROJECT_ROOT / "edge"
CLOUD_DIR = PROJECT_ROOT / "cloud"
ARCHIVE_DIR = PROJECT_ROOT / "archive"
TESTS_DIR = PROJECT_ROOT / "tests"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = EDGE_DIR / "models"

# Create directories if not exist
for directory in [EDGE_DIR, CLOUD_DIR, ARCHIVE_DIR, TESTS_DIR, DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Model Configuration
# ============================================================================

YOLO_CONFIG = {
    'model_name': 'yolov8n',  # nano variant for edge
    'input_size': 640,
    'confidence_threshold': 0.5,
    'nms_threshold': 0.45,
    'inference_timeout_ms': 30  # Target 30ms
}

SENSOR_FUSION_CONFIG = {
    'imu_frequency': 30.0,  # Hz (helmet camera)
    'uwb_frequency': 3.0,   # Hz (UWB RTLS)
    'gps_frequency': 1.0,   # Hz (GPS fallback)
    'dt': 1.0 / 30,
    'process_noise': 0.01,
    'measurement_noise_uwb': 0.1,
    'measurement_noise_imu': 0.5
}

VIT_CONFIG = {
    'model_variant': 'ViT-Base',
    'pretrain': 'COCO',
    'input_size': (384, 384),
    'num_classes': 10,  # Custom construction safety classes
    'batch_size': 8
}

LSTM_CONFIG = {
    'input_dim': 6,  # [x, y, z, vx, vy, vz]
    'hidden_dim': 128,
    'num_layers': 2,
    'output_dim': 3,  # [x, y, z] prediction
    'prediction_horizon': 30,  # 3 seconds (30 frames @ 10Hz)
    'sequence_length': 10
}

AUTOENCODER_CONFIG = {
    'input_dim': 256,
    'latent_dim': 32,
    'hidden_dims': [256, 128, 64],
    'reconstruction_threshold': 0.5
}

# ============================================================================
# Communication Configuration
# ============================================================================

MQTT_CONFIG = {
    'protocol': 'MQTT',
    'qos': 1,  # At least once
    'keepalive': 60,
    'reconnect_delay': 5,
    'max_retries': 3
}

HTTP_CONFIG = {
    'protocol': 'HTTP',
    'timeout': 5,
    'retries': 3,
    'backoff_factor': 1.5
}

# Topic structure for MQTT
MQTT_TOPICS = {
    'telemetry': 'construction/{node_id}/telemetry',
    'command': 'construction/{node_id}/command',
    'status': 'construction/{node_id}/status',
    'alert': 'construction/{node_id}/alert'
}

# ============================================================================
# Cloud Server Configuration
# ============================================================================

CLOUD_SERVER_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'workers': 4,
    'log_level': 'info',
    'reload': False  # Set to True for development
}

API_ENDPOINTS = {
    'telemetry': '/api/construction/{node_id}/telemetry',
    'analysis': '/api/analysis/{node_id}',
    'nodes': '/api/nodes',
    'status': '/api/status/{node_id}',
    'metrics': '/api/metrics',
    'health': '/health'
}

# ============================================================================
# Archive Configuration
# ============================================================================

ARCHIVE_CONFIG = {
    'storage_type': 'sqlite',  # or 'postgresql'
    'storage_path': str(ARCHIVE_DIR / 'hash_chain.db'),
    'backup_interval': 3600,  # seconds
    'retention_days': 365,  # 1 year
    'enable_compression': False
}

# ============================================================================
# Alert Levels
# ============================================================================

ALERT_LEVELS = {
    'NORMAL': 0,
    'WARNING': 1,
    'CRITICAL': 2
}

ALERT_THRESHOLDS = {
    'collision_risk': 0.3,
    'anomaly_score': 0.5,
    'no_helmet': 0.95,
    'high_acceleration': 20.0  # m/s^2
}

# ============================================================================
# Construction Safety Classes (for ViT fine-tuning)
# ============================================================================

ACTIVITY_CLASSES = {
    0: 'normal_work',
    1: 'danger_no_helmet',
    2: 'danger_collision_risk',
    3: 'danger_high_fall_risk',
    4: 'danger_equipment_hazard',
    5: 'danger_chemical_exposure',
    6: 'abnormal_behavior',
    7: 'equipment_malfunction',
    8: 'emergency_evacuation',
    9: 'idle'
}

# ============================================================================
# Logging Configuration
# ============================================================================

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': str(PROJECT_ROOT / 'logs' / 'app.log')
        }
    },
    'loggers': {
        '': {
            'level': 'DEBUG',
            'handlers': ['console', 'file']
        }
    }
}

# Create logs directory
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Development / Testing
# ============================================================================

TEST_CONFIG = {
    'sensor_simulation_duration': 10.0,  # seconds
    'test_batch_size': 5,
    'benchmark_iterations': 100
}

# ============================================================================
# Hardware Configuration
# ============================================================================

EDGE_HARDWARE = {
    'device_type': 'smart_helmet',
    'processor': 'NPU',  # TensorRT, CoreML, NNAPI
    'memory_mb': 512,
    'storage_gb': 16,
    'battery_mah': 2500,
    'wifi_standard': 'WiFi6',
    'expected_latency_ms': 30
}

# ============================================================================
# Feature Flags
# ============================================================================

FEATURES = {
    'enable_mqtt': True,
    'enable_http_fallback': True,
    'enable_gps_fallback': True,
    'enable_local_onnx_inference': True,
    'enable_bim_matching': False,  # MVP: not implemented
    'enable_rl_swarm': False,  # MVP: not implemented
    'enable_real_blockchain': False,  # MVP: using hash chain
    'enable_model_caching': True,
    'enable_telemetry_compression': False
}

# ============================================================================
# Performance Targets
# ============================================================================

PERFORMANCE_TARGETS = {
    'yolo_inference_ms': 30,
    'sensor_fusion_ms': 5,
    'lstm_prediction_ms': 50,
    'vit_analysis_ms': 500,
    'autoencoder_ms': 200,
    'bim_matching_ms': 1000,
    'total_e2e_latency_ms': 2000,  # Edge to Cloud to Response
    'cloud_api_p95_latency_ms': 100
}
