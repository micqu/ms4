from pandas.core.frame import DataFrame
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from model import Net
from syft.workers.virtual import VirtualWorker
from test_params import TestParams
import syft as sy
from typing import Callable, Iterator, List, Tuple
import torch as th
# from opacus import PrivacyEngine

class Node():
    def __init__(self, model: Net, params: Iterator[Parameter], optimizer: Optimizer, virtual_worker: VirtualWorker = None):
        self.virtual_worker = virtual_worker
        self.model = model
        self.params = params
        self.optimizer = optimizer

        # Pointers to data contained in virtual worker. (instead of remote dataset variable)
        self.batch_pointers: List[Tuple(any, any)] = []
        self.metric_log: List[List[any]] = []
        self.train_loader: DataLoader = None
        self.test_loader: DataLoader = None
        self.train_size: int = -1
        self.test_size: int = -1

        # For differential privacy
        # self.privacy_engine: PrivacyEngine = None

def create_nodes(
    hook,
    num_nodes: int,
    create_model_func: Callable[[TestParams, th.device], Net],
    create_optimizer_func: Callable[[Iterator[Parameter], TestParams], Optimizer],
    args: TestParams,
    device: th.device) -> List[Node]:

    node_list : List[Node] = list()
    for i in range(num_nodes):
        if args.use_websocket == True:
            virtual_worker = sy.websocket_client(hook, host=f"w{i}", port=3337)
            # classsyft.workers.websocket_client.WebsocketClientWorker(hook, host: str, port: int, secure: bool = False,
            # id: Union[int, str] = 0, is_client_worker: bool = False, log_msgs: bool = False, verbose: bool = False,
            # data: List[Union[torch.Tensor, AbstractTensor]] = None, timeout: int = None)
        else:
            virtual_worker = sy.VirtualWorker(hook, id=f"w{i}")

        model = create_model_func(args, device)
        optimizer = create_optimizer_func(model.parameters(), args)
        params = list(model.parameters())

        node_list.append(Node(model, params, optimizer, virtual_worker))

    return node_list

def create_global_node(
        create_model_func: Callable[[TestParams, th.device], Net],
        create_optimizer_func: Callable[[Iterator[Parameter], TestParams], Optimizer],
        args: TestParams, device: th.device) -> Node:

    model = create_model_func(args, device)
    optimizer = create_optimizer_func(model.parameters(), args)
    params = list(model.parameters())
    return Node(model, params, optimizer)

# def attach_privacy_engine(node: Node, args: TestParams):
#     privacy_engine = PrivacyEngine(
#         node.model,
#         batch_size = 512,
#         sample_size = node.train_size,
#         alphas = range(2,32),
#         noise_multiplier = 1.2,
#         max_grad_norm = 1.0
#     )
#     privacy_engine.attach(node.optimizer)