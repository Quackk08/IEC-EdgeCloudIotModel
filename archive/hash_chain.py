"""
Data Archive: Immutable Hash Chain for Digital Forensics
Purpose: Create tamper-proof record of all analysis results
Technology: SHA-256 hash chain (blockchain-inspired, not true blockchain)
Storage: SQLite or PostgreSQL
"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HashBlock:
    """Single block in hash chain."""
    block_id: int
    timestamp: datetime
    node_id: str
    data_payload: Dict[str, Any]
    prev_hash: str
    data_hash: str
    block_hash: str = field(default="")
    
    def compute_hash(self) -> str:
        """Compute SHA-256 hash of this block."""
        block_data = {
            'block_id': self.block_id,
            'timestamp': self.timestamp.isoformat(),
            'node_id': self.node_id,
            'prev_hash': self.prev_hash,
            'data_hash': self.data_hash
        }
        block_json = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_json.encode()).hexdigest()


class HashChain:
    """
    Append-only hash chain for immutable records.
    """
    
    def __init__(self, storage_path: str = "archive/hash_chain.db"):
        """
        Initialize hash chain with SQLite backend.
        
        Args:
            storage_path: Path to SQLite database file
        """
        self.storage_path = storage_path
        self.chain = []
        self.last_block_hash = "0" * 64  # Initial hash (all zeros)
        
        self._init_database()
        self._load_chain()
        
        logger.info(f"✓ Hash chain initialized (storage: {storage_path})")
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            # Create hash chain table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hash_blocks (
                    block_id INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    block_hash TEXT NOT NULL UNIQUE,
                    data_json TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_node_id ON hash_blocks(node_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON hash_blocks(timestamp)
            """)
            
            # Create metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chain_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("✓ Database schema initialized")
            
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _load_chain(self):
        """Load existing chain from database."""
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT block_id, timestamp, node_id, prev_hash, data_hash, block_hash, data_json
                FROM hash_blocks
                ORDER BY block_id ASC
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            if rows:
                for row in rows:
                    block = HashBlock(
                        block_id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        node_id=row[2],
                        prev_hash=row[3],
                        data_hash=row[4],
                        data_payload=json.loads(row[6])
                    )
                    block.block_hash = row[5]
                    self.chain.append(block)
                
                self.last_block_hash = self.chain[-1].block_hash
                logger.info(f"✓ Loaded {len(self.chain)} blocks from database")
            
        except sqlite3.Error as e:
            logger.warning(f"Failed to load chain: {e}")
    
    def add_block(self, 
                  node_id: str,
                  data_payload: Dict[str, Any]) -> HashBlock:
        """
        Add new block to hash chain.
        
        Args:
            node_id: Edge node identifier
            data_payload: Analysis result or event data
            
        Returns:
            Created HashBlock
        """
        # Compute data hash
        data_json = json.dumps(data_payload, sort_keys=True)
        data_hash = hashlib.sha256(data_json.encode()).hexdigest()
        
        # Create block
        block_id = len(self.chain) + 1
        block = HashBlock(
            block_id=block_id,
            timestamp=datetime.utcnow(),
            node_id=node_id,
            data_payload=data_payload,
            prev_hash=self.last_block_hash,
            data_hash=data_hash
        )
        
        # Compute block hash
        block.block_hash = block.compute_hash()
        
        # Save to database
        self._save_block(block)
        
        # Add to memory chain
        self.chain.append(block)
        self.last_block_hash = block.block_hash
        
        logger.info(f"✓ Block {block_id} added (hash: {block.block_hash[:8]}...)")
        
        return block
    
    def _save_block(self, block: HashBlock):
        """Save block to database."""
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO hash_blocks 
                (block_id, timestamp, node_id, prev_hash, data_hash, block_hash, data_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                block.block_id,
                block.timestamp.isoformat(),
                block.node_id,
                block.prev_hash,
                block.data_hash,
                block.block_hash,
                json.dumps(block.data_payload)
            ))
            
            conn.commit()
            conn.close()
            
        except sqlite3.Error as e:
            logger.error(f"Failed to save block: {e}")
            raise
    
    def verify_integrity(self) -> bool:
        """
        Verify integrity of entire chain.
        Checks that each block correctly references previous block.
        
        Returns:
            True if chain is valid, False otherwise
        """
        if not self.chain:
            return True
        
        # Verify first block has prev_hash of zeros
        if self.chain[0].prev_hash != "0" * 64:
            logger.error("First block prev_hash is not zero")
            return False
        
        # Verify each block
        for i in range(len(self.chain)):
            current = self.chain[i]
            
            # Recompute hash
            computed_hash = current.compute_hash()
            if computed_hash != current.block_hash:
                logger.error(f"Block {i} hash mismatch")
                return False
            
            # Check prev_hash reference
            if i > 0:
                prev_block = self.chain[i-1]
                if current.prev_hash != prev_block.block_hash:
                    logger.error(f"Block {i} prev_hash reference broken")
                    return False
        
        logger.info(f"✓ Chain integrity verified ({len(self.chain)} blocks)")
        return True
    
    def get_node_history(self, node_id: str, limit: int = 100) -> List[HashBlock]:
        """
        Retrieve all blocks for a specific node.
        
        Args:
            node_id: Node identifier
            limit: Maximum number of records to return
            
        Returns:
            List of blocks for the node
        """
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT block_id, timestamp, node_id, prev_hash, data_hash, block_hash, data_json
                FROM hash_blocks
                WHERE node_id = ?
                ORDER BY block_id DESC
                LIMIT ?
            """, (node_id, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            blocks = []
            for row in rows:
                block = HashBlock(
                    block_id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    node_id=row[2],
                    prev_hash=row[3],
                    data_hash=row[4],
                    data_payload=json.loads(row[6])
                )
                block.block_hash = row[5]
                blocks.append(block)
            
            return list(reversed(blocks))  # Return in chronological order
            
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve history: {e}")
            return []
    
    def get_block(self, block_id: int) -> Optional[HashBlock]:
        """
        Retrieve specific block by ID.
        
        Args:
            block_id: Block identifier
            
        Returns:
            HashBlock if found, None otherwise
        """
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT block_id, timestamp, node_id, prev_hash, data_hash, block_hash, data_json
                FROM hash_blocks
                WHERE block_id = ?
            """, (block_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                block = HashBlock(
                    block_id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    node_id=row[2],
                    prev_hash=row[3],
                    data_hash=row[4],
                    data_payload=json.loads(row[6])
                )
                block.block_hash = row[5]
                return block
            
            return None
            
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve block: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get chain statistics."""
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM hash_blocks")
            total_blocks = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT node_id) FROM hash_blocks")
            unique_nodes = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM hash_blocks")
            timestamps = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_blocks': total_blocks,
                'unique_nodes': unique_nodes,
                'first_block_time': timestamps[0] if timestamps[0] else None,
                'last_block_time': timestamps[1] if timestamps[1] else None,
                'chain_valid': self.verify_integrity()
            }
            
        except sqlite3.Error as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def export_node_audit_trail(self, 
                                 node_id: str,
                                 output_file: str) -> bool:
        """
        Export complete audit trail for a node (for legal/compliance).
        
        Args:
            node_id: Node to export
            output_file: Output JSON file path
            
        Returns:
            True if export successful
        """
        try:
            blocks = self.get_node_history(node_id, limit=10000)
            
            audit_trail = {
                'node_id': node_id,
                'export_timestamp': datetime.utcnow().isoformat(),
                'total_records': len(blocks),
                'blocks': [
                    {
                        'block_id': b.block_id,
                        'timestamp': b.timestamp.isoformat(),
                        'data_hash': b.data_hash,
                        'block_hash': b.block_hash,
                        'data': b.data_payload
                    }
                    for b in blocks
                ]
            }
            
            with open(output_file, 'w') as f:
                json.dump(audit_trail, f, indent=2)
            
            logger.info(f"✓ Audit trail exported to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False


# ============================================================================
# Archive Manager (Integration with Cloud Server)
# ============================================================================

class ArchiveManager:
    """Manage archiving of analysis results."""
    
    def __init__(self, storage_path: str = "archive/hash_chain.db"):
        """Initialize archive manager."""
        self.hash_chain = HashChain(storage_path)
    
    def archive_analysis_result(self, 
                                node_id: str,
                                result: Dict[str, Any]) -> str:
        """
        Archive analysis result to immutable hash chain.
        
        Args:
            node_id: Edge node ID
            result: Analysis result dictionary
            
        Returns:
            Block hash (for reference)
        """
        block = self.hash_chain.add_block(node_id, result)
        return block.block_hash
    
    def get_audit_trail(self, node_id: str) -> List[Dict[str, Any]]:
        """Get audit trail for a node."""
        blocks = self.hash_chain.get_node_history(node_id)
        
        return [
            {
                'block_id': b.block_id,
                'timestamp': b.timestamp.isoformat(),
                'data_hash': b.data_hash,
                'block_hash': b.block_hash,
                'data': b.data_payload
            }
            for b in blocks
        ]


# ============================================================================
# Main Demo
# ============================================================================

def main():
    """Demo hash chain functionality."""
    
    logger.info("=== Hash Chain Archive Demo ===")
    
    # Initialize hash chain
    chain = HashChain(storage_path="archive/demo_chain.db")
    
    # Add sample blocks
    logger.info("Adding sample blocks...")
    
    for i in range(3):
        data = {
            'event': 'analysis_result',
            'alert_level': 'warning' if i % 2 == 0 else 'normal',
            'anomaly_score': 0.2 + i * 0.1,
            'collision_risk': 0.1 * i
        }
        
        block = chain.add_block(f"helmet_{i:03d}", data)
    
    # Verify integrity
    logger.info("Verifying chain integrity...")
    is_valid = chain.verify_integrity()
    logger.info(f"Chain valid: {is_valid}")
    
    # Get statistics
    stats = chain.get_statistics()
    logger.info(f"Statistics: {stats}")
    
    # Retrieve history
    logger.info("Retrieving node history...")
    history = chain.get_node_history("helmet_001", limit=10)
    logger.info(f"Found {len(history)} records for helmet_001")
    
    # Export audit trail
    logger.info("Exporting audit trail...")
    chain.export_node_audit_trail("helmet_001", "archive/audit_trail_helmet_001.json")


if __name__ == "__main__":
    main()
