"""
Cloud Analytics Server: FastAPI
Purpose: Receive edge telemetry, run AI models, orchestrate analysis
Components: ViT, LSTM, Autoencoder, BIM matching
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import asyncio
import time
from datetime import datetime
import numpy as np
import json

from config import ARCHIVE_CONFIG, ALERT_THRESHOLDS
from archive.hash_chain import ArchiveManager
from cloud.vit_analysis import ViTAnalyzer
from cloud.lstm_trajectory import LSTMTrajectoryPredictor
from cloud.autoencoder import AutoencoderAnomalyDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================

class LocationData(BaseModel):
    """Location data from edge."""
    x: float
    y: float
    z: float


class VelocityData(BaseModel):
    """Velocity data from edge."""
    vx: float
    vy: float
    vz: float


class IMUData(BaseModel):
    """IMU sensor data."""
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: Optional[float] = None
    gyro_y: Optional[float] = None
    gyro_z: Optional[float] = None


class YOLODetection(BaseModel):
    """Object detection result."""
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]


class EdgeTelemetry(BaseModel):
    """Telemetry message from edge node."""
    node_id: str
    timestamp: float
    location: LocationData
    velocity: VelocityData
    imu: Optional[IMUData] = None
    yolo_detections: Optional[List[YOLODetection]] = None
    image_frame_hash: Optional[str] = None
    confidence: float = 1.0


class AnalysisResult(BaseModel):
    """AI analysis result from cloud."""
    node_id: str
    timestamp: float
    
    # Detection results
    yolo_detections: Optional[List[YOLODetection]] = None
    
    # High-level analysis (Vision Transformer)
    activity_type: Optional[str] = None  # 'normal_work', 'danger', 'anomaly'
    activity_confidence: Optional[float] = None
    
    # Trajectory prediction (LSTM)
    predicted_trajectory: Optional[List[LocationData]] = None
    collision_risk: Optional[float] = None  # 0-1
    recommended_action: Optional[str] = None
    
    # Anomaly detection (Autoencoder)
    anomaly_score: Optional[float] = None  # 0-1, >0.5 = anomaly
    bim_deviation_mm: Optional[float] = None
    
    # Overall alert level
    alert_level: str = "normal"  # 'normal', 'warning', 'critical'
    

# ============================================================================
# Cloud Analytics Models (Lazy Loaded)
# ============================================================================

class CloudModelManager:
    """Manage AI model instances."""

    def __init__(self):
        self.vit_analyzer = None
        self.lstm_predictor = None
        self.autoencoder_detector = None
        self.models_loaded = False

        logger.info("✓ Cloud model manager initialized")

    def load_models(self):
        """Load AI models (lazy loading)."""
        if self.models_loaded:
            return

        logger.info("Loading cloud AI models...")

        try:
            self.vit_analyzer = ViTAnalyzer()
            self.lstm_predictor = LSTMTrajectoryPredictor()
            self.autoencoder_detector = AutoencoderAnomalyDetector()
            self.models_loaded = True
            logger.info("✓ All AI modules initialized")
        except Exception as e:
            self.models_loaded = False
            logger.error(f"Model loading failed: {e}")

    def analyze_activity(self,
                         yolo_detections: List[Dict[str, Any]],
                         image_frame_hash: Optional[str] = None) -> Dict[str, Any]:
        if self.vit_analyzer is None:
            return {'activity_type': 'idle', 'confidence': 0.8}
        return self.vit_analyzer.analyze(yolo_detections, image_frame_hash)

    def predict_trajectory(self,
                           location: Dict[str, float],
                           velocity: Dict[str, float]) -> Dict[str, Any]:
        if self.lstm_predictor is None:
            trajectory = []
            for i in range(30):
                t = i * 0.1
                trajectory.append({
                    'x': location['x'] + velocity['vx'] * t,
                    'y': location['y'] + velocity['vy'] * t,
                    'z': location['z'] + velocity['vz'] * t
                })
            collision_risk = 0.0
            if abs(velocity['vx']) > 1.0 or abs(velocity['vy']) > 1.0:
                collision_risk = min(0.5, (abs(velocity['vx']) + abs(velocity['vy'])) / 10.0)
            return {'trajectory': trajectory, 'collision_risk': collision_risk}

        return self.lstm_predictor.predict(location, velocity)

    def detect_anomaly(self,
                       sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.autoencoder_detector is None:
            imu = sensor_data.get('imu', {})
            acc_magnitude = np.sqrt(
                imu.get('acc_x', 0)**2 +
                imu.get('acc_y', 0)**2 +
                imu.get('acc_z', -9.81)**2
            )
            anomaly_score = 0.0
            if acc_magnitude > 15:
                anomaly_score = min(1.0, (acc_magnitude - 9.81) / 20.0)
            return {'anomaly_score': anomaly_score, 'reason': 'fallback'}

        return self.autoencoder_detector.detect(sensor_data)


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Construction Safety AI Platform",
    description="Cloud analytics server for smart helmet IoT system",
    version="1.0.0"
)

# Initialize model manager
model_manager = CloudModelManager()
archive_manager = ArchiveManager(storage_path=ARCHIVE_CONFIG['storage_path'])

# In-memory storage for demo (replace with database in production)
telemetry_store = {}
analysis_results = {}
server_start_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    model_manager.load_models()
    logger.info("✓ Cloud server started")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'models_loaded': model_manager.models_loaded
    }


@app.post("/api/construction/{node_id}/telemetry")
async def receive_telemetry(
    node_id: str,
    telemetry: EdgeTelemetry,
    background_tasks: BackgroundTasks
) -> JSONResponse:
    """
    Receive telemetry from edge node.
    
    Args:
        node_id: Edge node ID
        telemetry: Telemetry data
        background_tasks: Background task queue
        
    Returns:
        Acknowledgment and immediate response
    """
    # Store telemetry
    telemetry_store[node_id] = telemetry.dict()
    logger.info(f"Received telemetry from {node_id}")
    
    # Queue analysis task (async)
    background_tasks.add_task(
        analyze_telemetry,
        node_id,
        telemetry
    )
    
    return {
        'status': 'received',
        'node_id': node_id,
        'timestamp': telemetry.timestamp
    }


async def analyze_telemetry(node_id: str, telemetry: EdgeTelemetry):
    """
    Analyze telemetry and produce result.
    
    Args:
        node_id: Edge node ID
        telemetry: Telemetry data
    """
    try:
        # Run analysis models
        
        # 1. Activity analysis (Vision Transformer)
        activity = model_manager.analyze_activity(
            yolo_detections=telemetry.yolo_detections or [],
            image_frame_hash=telemetry.image_frame_hash
        )
        
        # 2. Trajectory prediction (LSTM)
        trajectory = model_manager.predict_trajectory(
            location=telemetry.location.dict(),
            velocity=telemetry.velocity.dict()
        )
        
        # 3. Anomaly detection (Autoencoder)
        anomaly = model_manager.detect_anomaly({
            'imu': telemetry.imu.dict() if telemetry.imu else {}
        })
        
        # Determine alert level
        alert_level = 'normal'
        if activity['activity_type'].startswith('danger'):
            alert_level = 'critical'
        elif trajectory['collision_risk'] > 0.3:
            alert_level = 'warning'
        elif anomaly['anomaly_score'] > 0.5:
            alert_level = 'warning'
        
        # Create result
        result = AnalysisResult(
            node_id=node_id,
            timestamp=telemetry.timestamp,
            yolo_detections=telemetry.yolo_detections,
            activity_type=activity['activity_type'],
            activity_confidence=activity['confidence'],
            predicted_trajectory=[
                LocationData(**pos) for pos in trajectory['trajectory'][:10]
            ],
            collision_risk=trajectory['collision_risk'],
            anomaly_score=anomaly['anomaly_score'],
            alert_level=alert_level
        )
        
        # Store result
        analysis_results[node_id] = result.dict()

        # Archive the analysis result for digital forensics
        try:
            archive_hash = archive_manager.archive_analysis_result(node_id, result.dict())
            logger.info(f"Archived analysis result for {node_id}: {archive_hash[:16]}...")
        except Exception as archive_exc:
            logger.error(f"Archive failed for {node_id}: {archive_exc}")

        # Log alert
        if alert_level != 'normal':
            logger.warning(f"ALERT: {alert_level.upper()} for {node_id} - {activity['activity_type']}")

        logger.info(f"Analysis complete for {node_id}: {alert_level}")
        
    except Exception as e:
        logger.error(f"Analysis failed for {node_id}: {e}")


@app.get("/api/analysis/{node_id}")
async def get_analysis_result(node_id: str) -> AnalysisResult:
    """
    Get latest analysis result for a node.
    
    Args:
        node_id: Edge node ID
        
    Returns:
        Latest analysis result
    """
    if node_id not in analysis_results:
        raise HTTPException(status_code=404, detail="No analysis results yet")
    
    result = analysis_results[node_id]
    return AnalysisResult(**result)


@app.get("/api/nodes")
async def list_nodes() -> Dict[str, Any]:
    """List all active nodes and their status."""
    return {
        'active_nodes': list(telemetry_store.keys()),
        'total_nodes': len(telemetry_store),
        'analysis_available': list(analysis_results.keys())
    }


@app.get("/api/status/{node_id}")
async def get_node_status(node_id: str) -> Dict[str, Any]:
    """Get current status of a node."""
    if node_id not in telemetry_store:
        raise HTTPException(status_code=404, detail="Node not found")
    
    telemetry = telemetry_store[node_id]
    result = analysis_results.get(node_id)
    
    return {
        'node_id': node_id,
        'last_telemetry': telemetry.get('timestamp'),
        'position': telemetry.get('location'),
        'alert_level': result.get('alert_level', 'unknown') if result else 'unknown',
        'has_analysis': node_id in analysis_results
    }


@app.post("/api/webhook/notify")
async def webhook_notification(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Webhook endpoint for external notifications.
    (For integration with alerting systems)
    """
    logger.info(f"Webhook notification: {data}")
    return {'status': 'received'}


@app.get("/api/archive/{node_id}")
async def get_audit_trail(node_id: str) -> Dict[str, Any]:
    """Retrieve audit trail for a node from the hash chain."""
    history = archive_manager.get_audit_trail(node_id)
    if not history:
        raise HTTPException(status_code=404, detail="No archive records found for node")
    return {
        'node_id': node_id,
        'record_count': len(history),
        'history': history
    }


@app.get("/api/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get server metrics."""
    return {
        'total_telemetry_received': len(telemetry_store),
        'analysis_results_available': len(analysis_results),
        'uptime_seconds': round(time.time() - server_start_time, 2),
        'models_loaded': model_manager.models_loaded,
        'hash_chain_length': len(archive_manager.hash_chain.chain)
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Cloud Analytics Server...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
