"""
Smart Construction Safety IoT System - Package Initialization
"""

__version__ = "1.0.0-MVP"
__author__ = "Engineering Competition Team"
__description__ = "Edge AI + UWB RTLS for Real-time Construction Safety Monitoring"

# Import main modules
try:
    from edge.sensor_fusion import SensorFusionPipeline, KalmanFilter3D
    from edge.yolov8_to_onnx import YOLOv8Converter
    from edge.communication import EdgeNode, MQTTClient, HTTPClient
except ImportError as e:
    print(f"Warning: Could not import edge modules: {e}")

try:
    from cloud.api_server import app, CloudModelManager
except ImportError as e:
    print(f"Warning: Could not import cloud modules: {e}")

try:
    from archive.hash_chain import HashChain, ArchiveManager
except ImportError as e:
    print(f"Warning: Could not import archive modules: {e}")

__all__ = [
    'SensorFusionPipeline',
    'KalmanFilter3D',
    'YOLOv8Converter',
    'EdgeNode',
    'MQTTClient',
    'HTTPClient',
    'CloudModelManager',
    'HashChain',
    'ArchiveManager'
]
