from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from model import Net
from test_params import TestParams
from typing import Callable, Iterator, List, Tuple
import torch as th
from opacus import PrivacyEngine

class Node():
    def __init__(self, model: Net, params: Iterator[Parameter], optimizer: Optimizer):
        self.model = model
        self.params = params
        self.optimizer = optimizer

        self.metric_log: List[List[any]] = []
        self.train_loader: DataLoader = None
        self.test_loader: DataLoader = None
        self.train_size: int = -1
        self.test_size: int = -1

        # For differential privacy
        self.privacy_engine: PrivacyEngine = None

def create_nodes(
    num_nodes: int,
    create_model_func: Callable[[TestParams, th.device], Net],
    create_optimizer_func: Callable[[Iterator[Parameter], TestParams], Optimizer],
    args: TestParams,
    device: th.device) -> List[Node]:

    node_list : List[Node] = list()
    for _ in range(num_nodes):
        model = create_model_func(args, device)
        optimizer = create_optimizer_func(model.parameters(), args)
        params = list(model.parameters())

        node_list.append(Node(model, params, optimizer))

    return node_list

def create_global_node(
        create_model_func: Callable[[TestParams, th.device], Net],
        create_optimizer_func: Callable[[Iterator[Parameter], TestParams], Optimizer],
        args: TestParams, device: th.device) -> Node:

    model = create_model_func(args, device)
    optimizer = create_optimizer_func(model.parameters(), args)
    params = list(model.parameters())
    return Node(model, params, optimizer)

def attach_privacy_engine(node: Node, args: TestParams):
    privacy_engine = PrivacyEngine(
        node.model,
        batch_size = args.batch_size,
        sample_size = node.train_size,
        alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier = 0.2,
        max_grad_norm = 1.0
    )
    privacy_engine.attach(node.optimizer)