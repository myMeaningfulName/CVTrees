from enum import Enum
from typing import Any, Dict, Optional, Iterator
import numpy as np

class ExecutionMode(Enum):
    TRAIN = "train"
    EVAL = "eval"

class GPContext:
    def __init__(self):
        self.mode = ExecutionMode.TRAIN
        self.models: Dict[str, Any] = {}
        self.node_counter = 0
        self.train_labels: Optional[np.ndarray] = None
        self.batch_size = 32
        
    def reset(self, mode: ExecutionMode = ExecutionMode.TRAIN):
        self.mode = mode
        self.node_counter = 0
        # We do NOT clear models if switching to EVAL, only if starting a new TRAIN session from scratch
        if mode == ExecutionMode.TRAIN:
            self.models = {}
            
    def get_next_node_id(self) -> str:
        nid = f"node_{self.node_counter}"
        self.node_counter += 1
        return nid
        
    def set_train_labels(self, labels: np.ndarray):
        self.train_labels = labels

    def get_model(self, node_id: str) -> Any:
        return self.models.get(node_id)

    def save_model(self, node_id: str, model: Any):
        self.models[node_id] = model

# Global instance
context = GPContext()
