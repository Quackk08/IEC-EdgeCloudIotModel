"""
Sensor Fusion: Kalman Filter for Multi-Sensor Integration
Purpose: Fuse IMU, UWB RTLS, and GPS data for accurate location tracking
Output: Clean (x, y, z, vx, vy, vz) state for LSTM trajectory prediction
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    """Sensor measurement data structure."""
    timestamp: float
    uwb_position: Optional[Tuple[float, float, float]] = None  # (x, y, z)
    imu_accel: Optional[Tuple[float, float, float]] = None      # (ax, ay, az)
    imu_gyro: Optional[Tuple[float, float, float]] = None       # (gx, gy, gz)
    gps_position: Optional[Tuple[float, float, float]] = None   # (lat, lon, alt)
    confidence: float = 1.0


@dataclass
class KalmanState:
    """Kalman filter state: position and velocity."""
    x: float  # m
    y: float  # m
    z: float  # m
    vx: float  # m/s
    vy: float  # m/s
    vz: float  # m/s


class KalmanFilter3D:
    """
    3D Kalman Filter for sensor fusion.
    
    State vector: [x, y, z, vx, vy, vz]
    Measurements: UWB position, IMU acceleration
    """
    
    def __init__(self, 
                 dt: float = 0.033,  # 30ms (30 FPS helmet camera)
                 process_noise: float = 0.01,
                 measurement_noise_uwb: float = 0.1,
                 measurement_noise_imu: float = 0.5):
        """
        Initialize Kalman filter.
        
        Args:
            dt: Time step (seconds)
            process_noise: Process noise covariance (Q)
            measurement_noise_uwb: UWB measurement noise (R_uwb)
            measurement_noise_imu: IMU measurement noise (R_imu)
        """
        self.dt = dt
        
        # State vector: [x, y, z, vx, vy, vz]
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix for UWB (position only)
        self.H_uwb = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement matrix for IMU (acceleration, derived from velocity)
        # We measure d²x/dt² = a, so we use state derivative
        self.H_imu = np.array([
            [0, 0, 0, 0, 0, 0],  # IMU doesn't directly measure position
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        
        # State covariance matrix (uncertainty)
        self.P = np.eye(6) * 1.0
        
        # Process noise covariance
        self.Q = np.eye(6) * process_noise
        self.Q[0:3, 0:3] *= 0.1  # Lower process noise for position
        
        # Measurement noise covariance (UWB is more accurate)
        self.R_uwb = np.eye(3) * measurement_noise_uwb
        self.R_imu = np.eye(3) * measurement_noise_imu
        
        # IMU bias (gravity-corrected acceleration offset)
        self.imu_bias = np.array([0.0, 0.0, -9.81])  # Z-axis gravity
        
        logger.info(f"✓ Kalman filter initialized (dt={dt}s)")
    
    def predict(self) -> KalmanState:
        """
        Prediction step: Update state based on motion model.
        
        Returns:
            Current Kalman state
        """
        # Predict state
        self.state = self.F @ self.state
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self._state_to_dataclass()
    
    def update_uwb(self, uwb_position: Tuple[float, float, float]) -> KalmanState:
        """
        Update step: Incorporate UWB RTLS measurement (most accurate).
        
        Args:
            uwb_position: (x, y, z) in meters
            
        Returns:
            Updated Kalman state
        """
        z = np.array(uwb_position)
        
        # Innovation (measurement residual)
        y = z - (self.H_uwb @ self.state)
        
        # Innovation covariance
        S = self.H_uwb @ self.P @ self.H_uwb.T + self.R_uwb
        
        # Kalman gain
        K = self.P @ self.H_uwb.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        self.P = (np.eye(6) - K @ self.H_uwb) @ self.P
        
        return self._state_to_dataclass()
    
    def update_imu(self, imu_accel: Tuple[float, float, float]) -> KalmanState:
        """
        Update step: Incorporate IMU acceleration measurement.
        Note: IMU directly measures acceleration, used for drift correction.
        
        Args:
            imu_accel: (ax, ay, az) in m/s²
            
        Returns:
            Updated Kalman state
        """
        # IMU provides acceleration, which helps correct velocity drift
        # In simple KF, we primarily use UWB for position; IMU for validation
        # Advanced: use IMU to estimate state derivative directly
        
        accel = np.array(imu_accel) - self.imu_bias
        
        # Simple validation: check if acceleration is reasonable
        accel_magnitude = np.linalg.norm(accel)
        if accel_magnitude > 20:  # >2g, likely faulty reading
            logger.warning(f"High IMU acceleration detected: {accel_magnitude:.2f} m/s²")
            return self._state_to_dataclass()
        
        # In production: integrate acceleration into state estimate
        # For MVP: use IMU primarily for drift detection
        
        return self._state_to_dataclass()
    
    def _state_to_dataclass(self) -> KalmanState:
        """Convert state vector to KalmanState dataclass."""
        return KalmanState(
            x=float(self.state[0]),
            y=float(self.state[1]),
            z=float(self.state[2]),
            vx=float(self.state[3]),
            vy=float(self.state[4]),
            vz=float(self.state[5])
        )


class SensorFusionPipeline:
    """
    Multi-sensor fusion pipeline combining UWB, IMU, and GPS.
    """
    
    def __init__(self, 
                 imu_frequency: float = 100.0,  # 100 Hz
                 uwb_frequency: float = 10.0,   # 10 Hz
                 gps_frequency: float = 1.0):   # 1 Hz
        """
        Initialize sensor fusion pipeline.
        
        Args:
            imu_frequency: IMU sampling rate (Hz)
            uwb_frequency: UWB RTLS sampling rate (Hz)
            gps_frequency: GPS sampling rate (Hz)
        """
        self.imu_dt = 1.0 / imu_frequency
        self.uwb_dt = 1.0 / uwb_frequency
        self.gps_dt = 1.0 / gps_frequency
        
        # Kalman filter (updated at IMU frequency)
        self.kf = KalmanFilter3D(dt=self.imu_dt)
        
        # Buffers for sensor data
        self.last_uwb_time = 0.0
        self.last_gps_time = 0.0
        self.last_imu_time = 0.0
        
        logger.info(f"✓ Sensor fusion pipeline initialized")
        logger.info(f"  IMU: {imu_frequency} Hz, UWB: {uwb_frequency} Hz, GPS: {gps_frequency} Hz")
    
    def process_sensor_reading(self, reading: SensorReading) -> KalmanState:
        """
        Process incoming sensor reading and fuse measurements.
        
        Args:
            reading: SensorReading with measurements
            
        Returns:
            Fused state estimate
        """
        # Prediction step (always run)
        state = self.kf.predict()
        
        # UWB update (high priority, most accurate)
        if reading.uwb_position is not None:
            state = self.kf.update_uwb(reading.uwb_position)
            self.last_uwb_time = reading.timestamp
        
        # IMU update (medium priority, for drift correction)
        if reading.imu_accel is not None:
            state = self.kf.update_imu(reading.imu_accel)
            self.last_imu_time = reading.timestamp
        
        # GPS fallback (low priority, used when UWB unavailable)
        if reading.gps_position is not None and \
           (reading.timestamp - self.last_uwb_time) > 2.0:  # UWB gap >2s
            # Convert GPS to local coordinates (requires reference point)
            state = self.kf.update_uwb(reading.gps_position)
            self.last_gps_time = reading.timestamp
        
        return state
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get current state as numpy array.
        
        Returns:
            [x, y, z, vx, vy, vz]
        """
        state = self.kf._state_to_dataclass()
        return np.array([state.x, state.y, state.z, 
                        state.vx, state.vy, state.vz])
    
    def reset(self):
        """Reset filter to initial state."""
        self.kf.state = np.zeros(6)
        self.kf.P = np.eye(6)
        logger.info("Kalman filter reset")


class SensorSimulator:
    """Simulate sensor data for testing."""
    
    @staticmethod
    def simulate_trajectory(duration: float = 10.0, 
                           dt: float = 0.033) -> list:
        """
        Simulate realistic construction worker trajectory.
        
        Args:
            duration: Simulation duration (seconds)
            dt: Time step (seconds)
            
        Returns:
            List of SensorReading objects
        """
        readings = []
        t = 0.0
        
        # Simulate circular path with noise
        radius = 5.0
        angular_velocity = 0.5  # rad/s
        
        uwb_counter = 0
        uwb_period = 10  # UWB every 10 IMU frames (1/30 * 10 ≈ 0.33s)
        
        while t < duration:
            # Ground truth position
            angle = angular_velocity * t
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 1.7  # Head height
            
            # Velocity (tangent to circle)
            vx = -radius * angular_velocity * np.sin(angle)
            vy = radius * angular_velocity * np.cos(angle)
            vz = 0.0
            
            # Acceleration (centripetal)
            ax = -radius * angular_velocity**2 * np.cos(angle)
            ay = -radius * angular_velocity**2 * np.sin(angle)
            az = 9.81  # Gravity
            
            # Add noise
            uwb_pos = None
            if uwb_counter % uwb_period == 0:
                # UWB measurement (relatively accurate)
                uwb_pos = (
                    x + np.random.normal(0, 0.05),
                    y + np.random.normal(0, 0.05),
                    z + np.random.normal(0, 0.02)
                )
            
            imu_accel = (
                ax + np.random.normal(0, 0.1),
                ay + np.random.normal(0, 0.1),
                az + np.random.normal(0, 0.1)
            )
            
            imu_gyro = (
                0.5 + np.random.normal(0, 0.01),
                0.0 + np.random.normal(0, 0.01),
                0.0 + np.random.normal(0, 0.01)
            )
            
            reading = SensorReading(
                timestamp=t,
                uwb_position=uwb_pos,
                imu_accel=imu_accel,
                imu_gyro=imu_gyro,
                confidence=0.95 if uwb_pos else 0.7
            )
            readings.append(reading)
            
            t += dt
            uwb_counter += 1
        
        return readings


def main():
    """Demo sensor fusion pipeline."""
    
    logger.info("=== Sensor Fusion Demo ===")
    
    # Initialize pipeline
    pipeline = SensorFusionPipeline(
        imu_frequency=30.0,  # Helmet camera frequency
        uwb_frequency=3.0,   # UWB every 10 frames
        gps_frequency=1.0
    )
    
    # Simulate sensor readings
    logger.info("Simulating sensor data...")
    readings = SensorSimulator.simulate_trajectory(duration=5.0, dt=0.033)
    
    # Process readings
    states = []
    for reading in readings:
        state = pipeline.process_sensor_reading(reading)
        states.append(state)
    
    # Display results
    logger.info(f"Processed {len(states)} sensor readings")
    
    if len(states) > 0:
        final_state = states[-1]
        logger.info(f"Final position: ({final_state.x:.2f}, {final_state.y:.2f}, {final_state.z:.2f})")
        logger.info(f"Final velocity: ({final_state.vx:.2f}, {final_state.vy:.2f}, {final_state.vz:.2f})")


if __name__ == "__main__":
    main()
