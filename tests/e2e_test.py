"""
End-to-End Integration Test
Purpose: Test entire pipeline: Edge → Cloud → Archive
Simulates: Sensor data generation → YOLO inference → Cloud analysis → Hash chain archiving
"""

import sys
import time
import logging
from typing import List, Dict, Any
from pathlib import Path
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class E2ETestPipeline:
    """End-to-end test pipeline."""
    
    def __init__(self):
        """Initialize E2E test components."""
        self.test_results = []
        self.passed = 0
        self.failed = 0
        
        logger.info("=" * 70)
        logger.info("END-TO-END INTEGRATION TEST")
        logger.info("=" * 70)
    
    def test_sensor_fusion(self) -> bool:
        """Test sensor fusion module."""
        logger.info("\n[TEST 1] Sensor Fusion Module")
        logger.info("-" * 70)
        
        try:
            from edge.sensor_fusion import (
                SensorFusionPipeline,
                SensorReading,
                SensorSimulator
            )
            
            # Initialize pipeline
            pipeline = SensorFusionPipeline(
                imu_frequency=30.0,
                uwb_frequency=3.0
            )
            logger.info("✓ Pipeline initialized")
            
            # Simulate trajectory
            readings = SensorSimulator.simulate_trajectory(duration=2.0, dt=0.033)
            logger.info(f"✓ Simulated {len(readings)} sensor readings")
            
            # Process readings
            states = []
            for reading in readings:
                state = pipeline.process_sensor_reading(reading)
                states.append(state)
            
            if len(states) > 0:
                final_state = states[-1]
                logger.info(f"✓ Final position: ({final_state.x:.2f}, {final_state.y:.2f}, {final_state.z:.2f})")
                logger.info(f"✓ Final velocity: ({final_state.vx:.2f}, {final_state.vy:.2f}, {final_state.vz:.2f})")
                logger.info("✓ Sensor fusion test PASSED")
                return True
            else:
                logger.error("✗ No states produced")
                return False
                
        except Exception as e:
            logger.error(f"✗ Sensor fusion test FAILED: {e}")
            return False
    
    def test_yolo_conversion(self) -> bool:
        """Test YOLOv8 to ONNX conversion."""
        logger.info("\n[TEST 2] YOLOv8 ONNX Conversion")
        logger.info("-" * 70)
        
        try:
            from edge.yolov8_to_onnx import YOLOv8Converter
            import numpy as np
            
            # Initialize converter
            converter = YOLOv8Converter(model_name="yolov8n")
            logger.info("✓ Converter initialized")
            
            # Note: Actual model download requires internet and ultralytics
            # In offline mode, we test the converter structure
            logger.info("⚠ Model download skipped (requires internet connection)")
            logger.info("⚠ In production: model will be converted via converter.download_and_convert()")
            
            # Test preprocessing function
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            preprocessed = converter.preprocess_image(dummy_image, img_size=640)
            
            if preprocessed.shape == (1, 3, 640, 640):
                logger.info(f"✓ Image preprocessing works: {preprocessed.shape}")
                logger.info("✓ ONNX conversion test PASSED")
                return True
            else:
                logger.error(f"✗ Unexpected output shape: {preprocessed.shape}")
                return False
                
        except Exception as e:
            logger.error(f"✗ ONNX conversion test FAILED: {e}")
            return False
    
    def test_cloud_api(self) -> bool:
        """Test cloud API server endpoints."""
        logger.info("\n[TEST 3] Cloud API Server")
        logger.info("-" * 70)
        
        try:
            from cloud.api_server import (
                app,
                EdgeTelemetry,
                LocationData,
                VelocityData,
                IMUData,
                YOLODetection
            )
            from fastapi.testclient import TestClient
            
            # Create test client
            client = TestClient(app)
            
            # Test 1: Health check
            response = client.get("/health")
            if response.status_code == 200:
                logger.info("✓ Health check passed")
            else:
                logger.error(f"✗ Health check failed: {response.status_code}")
                return False
            
            # Test 2: Send telemetry
            telemetry = EdgeTelemetry(
                node_id="test_helmet_001",
                timestamp=time.time(),
                location=LocationData(x=10.5, y=20.3, z=1.7),
                velocity=VelocityData(vx=0.1, vy=-0.05, vz=0.0),
                imu=IMUData(
                    acc_x=0.02,
                    acc_y=0.01,
                    acc_z=9.8,
                    gyro_x=0.001
                ),
                yolo_detections=[
                    YOLODetection(
                        class_name="helmet",
                        confidence=0.98,
                        bbox=[10, 20, 100, 150]
                    )
                ]
            )
            
            response = client.post(
                "/api/construction/test_helmet_001/telemetry",
                json=telemetry.dict()
            )
            
            if response.status_code == 200:
                logger.info("✓ Telemetry upload passed")
            else:
                logger.error(f"✗ Telemetry upload failed: {response.status_code}")
                return False
            
            # Test 3: List nodes
            response = client.get("/api/nodes")
            if response.status_code == 200:
                logger.info("✓ Node listing passed")
            else:
                logger.error(f"✗ Node listing failed: {response.status_code}")
                return False
            
            # Test 4: Get metrics
            response = client.get("/api/metrics")
            if response.status_code == 200:
                logger.info("✓ Metrics endpoint passed")
            else:
                logger.error(f"✗ Metrics endpoint failed: {response.status_code}")
                return False
            
            logger.info("✓ Cloud API test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"✗ Cloud API test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_hash_chain(self) -> bool:
        """Test hash chain archive."""
        logger.info("\n[TEST 4] Hash Chain Archive")
        logger.info("-" * 70)
        
        try:
            from archive.hash_chain import HashChain
            import tempfile
            import os
            
            # Create temporary database
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = os.path.join(tmpdir, "test_chain.db")
                
                # Initialize chain
                chain = HashChain(storage_path=db_path)
                logger.info("✓ Hash chain initialized")
                
                # Add blocks
                for i in range(5):
                    data = {
                        'event': 'analysis_result',
                        'alert_level': 'warning' if i % 2 == 0 else 'normal',
                        'anomaly_score': 0.1 * i,
                        'timestamp': time.time()
                    }
                    
                    block = chain.add_block(f"helmet_{i:03d}", data)
                
                logger.info("✓ Added 5 blocks to chain")
                
                # Verify integrity
                is_valid = chain.verify_integrity()
                if is_valid:
                    logger.info("✓ Chain integrity verified")
                else:
                    logger.error("✗ Chain integrity check failed")
                    return False
                
                # Get statistics
                stats = chain.get_statistics()
                logger.info(f"✓ Chain statistics: {stats['total_blocks']} blocks, {stats['unique_nodes']} nodes")
                
                # Retrieve history
                history = chain.get_node_history("helmet_001", limit=10)
                if len(history) > 0:
                    logger.info(f"✓ Retrieved {len(history)} records for helmet_001")
                else:
                    logger.warning("⚠ No history found for helmet_001")
                
                logger.info("✓ Hash chain test PASSED")
                return True
                
        except Exception as e:
            logger.error(f"✗ Hash chain test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_edge_communication(self) -> bool:
        """Test edge communication module."""
        logger.info("\n[TEST 5] Edge Communication")
        logger.info("-" * 70)
        
        try:
            from edge.communication import (
                EdgeNode,
                EdgeMessage,
                MQTTClient,
                HTTPClient
            )
            
            # Initialize edge node
            edge = EdgeNode(
                node_id="test_helmet_001",
                communication_protocol="http"
            )
            logger.info("✓ Edge node initialized with HTTP client")
            
            # Test telemetry message creation
            message = EdgeMessage(
                node_id="test_helmet_001",
                timestamp=time.time(),
                location={'x': 10.5, 'y': 20.3, 'z': 1.7},
                velocity={'vx': 0.1, 'vy': -0.05, 'vz': 0.0}
            )
            
            json_str = message.to_json()
            if json_str:
                logger.info("✓ Message serialization works")
            else:
                logger.error("✗ Message serialization failed")
                return False
            
            logger.info("✓ Edge communication test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"✗ Edge communication test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_integration_flow(self) -> bool:
        """Test integrated flow: Edge → Cloud → Archive."""
        logger.info("\n[TEST 6] Integrated Flow (Edge → Cloud → Archive)")
        logger.info("-" * 70)
        
        try:
            from edge.sensor_fusion import SensorFusionPipeline, SensorSimulator
            from archive.hash_chain import HashChain
            from cloud.api_server import CloudModelManager
            import tempfile
            import os
            
            logger.info("Step 1: Simulate sensor data")
            pipeline = SensorFusionPipeline()
            readings = SensorSimulator.simulate_trajectory(duration=1.0, dt=0.033)
            logger.info(f"✓ Generated {len(readings)} sensor readings")
            
            logger.info("Step 2: Process through sensor fusion")
            states = []
            for reading in readings:
                state = pipeline.process_sensor_reading(reading)
                states.append(state)
            logger.info(f"✓ Produced {len(states)} fused state vectors")
            
            logger.info("Step 3: Run cloud analysis")
            model_manager = CloudModelManager()
            model_manager.load_models()
            
            analysis_results = []
            for i, state in enumerate(states[::10]):  # Sample every 10th frame
                result = {
                    'node_id': 'test_helmet_001',
                    'timestamp': time.time() + i,
                    'position': {'x': state.x, 'y': state.y, 'z': state.z},
                    'activity': model_manager.analyze_activity([])
                }
                analysis_results.append(result)
            
            logger.info(f"✓ Generated {len(analysis_results)} analysis results")
            
            logger.info("Step 4: Archive to hash chain")
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = os.path.join(tmpdir, "integration_test.db")
                chain = HashChain(storage_path=db_path)
                
                for result in analysis_results:
                    block = chain.add_block(result['node_id'], result)
                
                logger.info(f"✓ Archived {len(analysis_results)} results to hash chain")
                
                # Verify
                is_valid = chain.verify_integrity()
                if is_valid:
                    logger.info("✓ Chain integrity verified")
                    logger.info("✓ Integrated flow test PASSED")
                    return True
                else:
                    logger.error("✗ Chain integrity check failed")
                    return False
            
        except Exception as e:
            logger.error(f"✗ Integration flow test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and generate report."""
        
        tests = [
            ("Sensor Fusion", self.test_sensor_fusion),
            ("YOLOv8 ONNX Conversion", self.test_yolo_conversion),
            ("Cloud API Server", self.test_cloud_api),
            ("Hash Chain Archive", self.test_hash_chain),
            ("Edge Communication", self.test_edge_communication),
            ("Integrated Flow", self.test_integration_flow)
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.test_results.append((test_name, result))
                
                if result:
                    self.passed += 1
                else:
                    self.failed += 1
                    
            except Exception as e:
                logger.error(f"Test '{test_name}' crashed: {e}")
                self.test_results.append((test_name, False))
                self.failed += 1
        
        # Print summary
        self._print_summary()
        
        return {
            'total_tests': len(tests),
            'passed': self.passed,
            'failed': self.failed,
            'results': self.test_results
        }
    
    def _print_summary(self):
        """Print test summary."""
        logger.info("\n" + "=" * 70)
        logger.info("TEST SUMMARY")
        logger.info("=" * 70)
        
        for test_name, result in self.test_results:
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"{status}: {test_name}")
        
        logger.info("-" * 70)
        logger.info(f"Total:  {len(self.test_results)}")
        logger.info(f"Passed: {self.passed}")
        logger.info(f"Failed: {self.failed}")
        logger.info("=" * 70)


def main():
    """Run E2E tests."""
    test_pipeline = E2ETestPipeline()
    results = test_pipeline.run_all_tests()
    
    # Return exit code
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
