"""
Edge Communication Layer: MQTT Client
Purpose: Send sensor fusion data and YOLO detections to cloud server
Protocol: MQTT 3.1.1 / 5.0 (with fallback to HTTP for edge environments)
"""

import json
import logging
from typing import Callable, Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EdgeMessage:
    """Message structure from edge to cloud."""
    node_id: str
    timestamp: float  # Unix timestamp
    location: Dict[str, float]  # {x, y, z}
    velocity: Dict[str, float]  # {vx, vy, vz}
    imu: Optional[Dict[str, float]] = None  # {acc_x, acc_y, acc_z, gyro_x, ...}
    yolo_detections: Optional[list] = None  # List of detections
    image_frame_hash: Optional[str] = None  # SHA256 of frame
    confidence: float = 1.0
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self), indent=2)


class CommunicationInterface(ABC):
    """Abstract base class for communication protocols."""
    
    @abstractmethod
    def connect(self, server_config: Dict[str, Any]) -> bool:
        """Establish connection to server."""
        pass
    
    @abstractmethod
    def send_message(self, topic: str, message: EdgeMessage) -> bool:
        """Send message to cloud."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Gracefully disconnect."""
        pass


class MQTTClient(CommunicationInterface):
    """
    MQTT client for edge-to-cloud communication.
    Implements reconnection logic and message queue buffering.
    """
    
    def __init__(self, node_id: str = "helmet_001"):
        """
        Initialize MQTT client.
        
        Args:
            node_id: Unique identifier for this edge node
        """
        self.node_id = node_id
        self.client = None
        self.connected = False
        self.message_queue = []
        self.message_lock = threading.Lock()
        
        # Message statistics
        self.messages_sent = 0
        self.messages_failed = 0
        self.last_connection_time = None
        
        logger.info(f"✓ MQTT client initialized (node_id={node_id})")
    
    def connect(self, server_config: Dict[str, Any]) -> bool:
        """
        Connect to MQTT broker.
        
        Args:
            server_config: Dict with 'host', 'port', 'username', 'password'
                Example: {
                    'host': 'broker.example.com',
                    'port': 8883,
                    'username': 'edge_user',
                    'password': 'secure_password',
                    'tls': True
                }
        
        Returns:
            True if connection successful
        """
        try:
            import paho.mqtt.client as mqtt
            
            self.client = mqtt.Client(client_id=self.node_id, protocol=mqtt.MQTTv311)
            
            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_publish = self._on_publish
            self.client.on_message = self._on_message
            
            # Set credentials
            if 'username' in server_config:
                self.client.username_pw_set(
                    server_config['username'],
                    server_config.get('password', '')
                )
            
            # TLS/SSL
            if server_config.get('tls', False):
                self.client.tls_set()
                self.client.tls_insecure_set(False)
            
            # Connect
            host = server_config.get('host', 'localhost')
            port = server_config.get('port', 1883)
            keepalive = server_config.get('keepalive', 60)
            
            logger.info(f"Connecting to MQTT broker: {host}:{port}")
            self.client.connect(host, port, keepalive)
            
            # Start network loop in background thread
            self.client.loop_start()
            
            # Wait for connection
            time.sleep(1)
            
            if self.connected:
                logger.info(f"✓ Connected to MQTT broker")
                self.last_connection_time = time.time()
                return True
            else:
                logger.warning("Connection attempt failed")
                return False
                
        except ImportError:
            logger.error("paho-mqtt not installed. Run: pip install paho-mqtt")
            return False
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def send_message(self, topic: str, message: EdgeMessage) -> bool:
        """
        Publish message to MQTT topic.
        
        Args:
            topic: MQTT topic (e.g., 'construction/helmet_001/telemetry')
            message: EdgeMessage to send
            
        Returns:
            True if message queued/sent successfully
        """
        if not self.client:
            logger.warning("MQTT client not initialized")
            return False
        
        try:
            payload = message.to_json()
            
            # Publish with QoS 1 (at least once delivery)
            result = self.client.publish(topic, payload, qos=1)
            
            if result.rc == 0:
                self.messages_sent += 1
                logger.debug(f"Message sent to {topic}")
                return True
            else:
                self.messages_failed += 1
                logger.warning(f"Publish failed (rc={result.rc})")
                
                # Buffer message for retry
                with self.message_lock:
                    self.message_queue.append((topic, message))
                
                return False
                
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.messages_failed += 1
            return False
    
    def disconnect(self):
        """Gracefully disconnect from broker."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info(f"✓ Disconnected from MQTT broker")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            'connected': self.connected,
            'messages_sent': self.messages_sent,
            'messages_failed': self.messages_failed,
            'queue_size': len(self.message_queue),
            'last_connection_time': self.last_connection_time
        }
    
    # Callback methods
    def _on_connect(self, client, userdata, flags, rc):
        """Called when broker responds to connection request."""
        if rc == 0:
            self.connected = True
            logger.info("MQTT connection established")
            
            # Retry queued messages
            self._retry_queued_messages()
        else:
            self.connected = False
            logger.error(f"Connection failed with code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Called when connection is lost."""
        self.connected = False
        if rc != 0:
            logger.warning(f"Unexpected disconnection (rc={rc}). Reconnecting...")
    
    def _on_publish(self, client, userdata, mid):
        """Called when message is published."""
        logger.debug(f"Message published (mid={mid})")
    
    def _on_message(self, client, userdata, msg):
        """Called when message is received (for RPC commands from cloud)."""
        logger.info(f"Command received: {msg.topic}")
        # Future: handle cloud-to-edge commands
    
    def _retry_queued_messages(self):
        """Retry sending buffered messages."""
        with self.message_lock:
            while self.message_queue:
                topic, message = self.message_queue.pop(0)
                self.send_message(topic, message)


class HTTPClient(CommunicationInterface):
    """
    HTTP client for environments where MQTT is not available.
    Fallback communication method.
    """
    
    def __init__(self, node_id: str = "helmet_001"):
        """Initialize HTTP client."""
        self.node_id = node_id
        self.base_url = None
        self.session = None
        self.connected = False
        
        logger.info(f"✓ HTTP client initialized (node_id={node_id})")
    
    def connect(self, server_config: Dict[str, Any]) -> bool:
        """
        Connect to HTTP server.
        
        Args:
            server_config: Dict with 'host', 'port', 'secure'
        """
        try:
            import requests
            
            host = server_config.get('host', 'localhost')
            port = server_config.get('port', 8000)
            secure = server_config.get('secure', False)
            
            protocol = 'https' if secure else 'http'
            self.base_url = f"{protocol}://{host}:{port}"
            
            self.session = requests.Session()
            
            # Test connection
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                self.connected = True
                logger.info(f"✓ Connected to HTTP server: {self.base_url}")
                return True
            else:
                logger.warning(f"Server health check failed: {response.status_code}")
                return False
                
        except ImportError:
            logger.error("requests not installed. Run: pip install requests")
            return False
        except Exception as e:
            logger.error(f"HTTP connection failed: {e}")
            return False
    
    def send_message(self, topic: str, message: EdgeMessage) -> bool:
        """
        Send message via HTTP POST.
        
        Args:
            topic: Endpoint path (e.g., '/telemetry')
            message: EdgeMessage to send
        """
        if not self.connected:
            logger.warning("HTTP client not connected")
            return False
        
        try:
            url = f"{self.base_url}/api{topic}"
            headers = {'Content-Type': 'application/json'}
            
            response = self.session.post(
                url,
                data=message.to_json(),
                headers=headers,
                timeout=5
            )
            
            if response.status_code in [200, 201]:
                logger.debug(f"HTTP message sent to {url}")
                return True
            else:
                logger.warning(f"HTTP POST failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"HTTP send error: {e}")
            return False
    
    def disconnect(self):
        """Close HTTP session."""
        if self.session:
            self.session.close()
            logger.info("✓ HTTP client disconnected")


class EdgeNode:
    """
    Main edge node class that coordinates all edge components.
    """
    
    def __init__(self, 
                 node_id: str = "helmet_001",
                 communication_protocol: str = "mqtt"):
        """
        Initialize edge node.
        
        Args:
            node_id: Unique node identifier
            communication_protocol: 'mqtt' or 'http'
        """
        self.node_id = node_id
        
        # Initialize communication
        if communication_protocol == "mqtt":
            self.comm = MQTTClient(node_id)
        elif communication_protocol == "http":
            self.comm = HTTPClient(node_id)
        else:
            raise ValueError(f"Unknown protocol: {communication_protocol}")
        
        logger.info(f"✓ Edge node '{node_id}' initialized")
    
    def connect_to_cloud(self, server_config: Dict[str, Any]) -> bool:
        """
        Connect to cloud server.
        
        Args:
            server_config: Configuration dictionary
        """
        return self.comm.connect(server_config)
    
    def send_telemetry(self, 
                      location: Dict[str, float],
                      velocity: Dict[str, float],
                      imu: Optional[Dict[str, float]] = None,
                      detections: Optional[list] = None) -> bool:
        """
        Send telemetry data to cloud.
        
        Args:
            location: {x, y, z} position
            velocity: {vx, vy, vz} velocity
            imu: Optional IMU data
            detections: Optional YOLO detections
        """
        message = EdgeMessage(
            node_id=self.node_id,
            timestamp=time.time(),
            location=location,
            velocity=velocity,
            imu=imu,
            yolo_detections=detections
        )
        
        topic = f"construction/{self.node_id}/telemetry"
        return self.comm.send_message(topic, message)
    
    def disconnect(self):
        """Disconnect from cloud."""
        self.comm.disconnect()


def main():
    """Demo communication pipeline."""
    
    logger.info("=== Edge Communication Demo ===")
    
    # Initialize edge node
    edge = EdgeNode(
        node_id="helmet_001",
        communication_protocol="mqtt"
    )
    
    # Configuration
    server_config = {
        'host': 'localhost',
        'port': 1883,
        'username': 'edge_user',
        'password': 'edge_pass',
        'tls': False
    }
    
    # Connect to cloud
    if edge.connect_to_cloud(server_config):
        logger.info("Connected to cloud")
        
        # Send sample telemetry
        for i in range(3):
            success = edge.send_telemetry(
                location={'x': 10.5 + i*0.1, 'y': 20.3, 'z': 1.7},
                velocity={'vx': 0.1, 'vy': -0.05, 'vz': 0.0},
                imu={
                    'acc_x': 0.02, 'acc_y': 0.01, 'acc_z': 9.8,
                    'gyro_x': 0.001, 'gyro_y': 0.002, 'gyro_z': 0.0
                },
                detections=[
                    {'class': 'helmet', 'confidence': 0.98, 'bbox': [10, 20, 100, 150]},
                    {'class': 'obstacle', 'confidence': 0.85, 'bbox': [200, 100, 350, 300]}
                ]
            )
            
            logger.info(f"Telemetry {i+1}: {'✓' if success else '✗'}")
            time.sleep(0.5)
        
        # Display stats
        if isinstance(edge.comm, MQTTClient):
            stats = edge.comm.get_stats()
            logger.info(f"MQTT Stats: {stats}")
        
        # Disconnect
        edge.disconnect()
    else:
        logger.warning("Failed to connect to cloud (expected in demo without broker)")


if __name__ == "__main__":
    main()
