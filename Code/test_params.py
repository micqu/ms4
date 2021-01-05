from enum import Enum
from typing import List

class ModelType(Enum):
    FEDAVG = 1
    CENTRAL = 2
    PFNM = 3

class TestParams(object):
    def __init__(self):
        self.modelType: ModelType
        self.smpc_aggregation: bool = False
        self.dp: bool = False
        self.data_iid: bool = False
        self.data_balanced: bool = True
        self.has_global_model: int = True
        self.binary_classification: bool = False # Change to is_classification

        self.n_nodes: int = 3
        self.epochs: int = 5
        self.rounds: int = 30
        self.batch_size: int = 16 #?

        self.dp_delta = 0.00001

        ## Data partitioning
        # self.min_part_min_size: int = 10
        # self.min_part_max_size: int = 5000
        self.unbalanced_min_part_min_size: float = 0.08
        self.unbalanced_min_part_max_size: float = 0.10
        self.dirichlet_alpha: float = 0.5

        self.save_model: bool = True
        self.lr: float = 0.005
        self.layer_sizes: List[int] = [784, 64, 64, 10]
        self.dropout: float = 0.5
        self.seed: int = 12346
        self.test_batch_size: int = 256
        self.log_interval: int = 1
        self.no_cuda: bool = True
        self.log_dir: str = "results"
        self.test_size = 0.20

        # Not implemented
        self.momentum = 0.5
        self.use_websocket: bool = False