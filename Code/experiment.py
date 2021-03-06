import os
import torch

from test_params import *
from data_loader import *
from test_params import *
import fed

def main():
    # cut()
    # torch.autograd.set_detect_anomaly(True)

    for i in range(3):
        run_central()
    # run_federated()

    # os.system("shutdown /s /t 1");

def run_federated():
    ### Global, local
    experiments = [
        # ex_g_iid_balanced,
        ex_g_iid_unbalanced,
        # ex_g_non_iid_balanced,
        # ex_g_non_iid_unbalanced,
        # ex_l_iid_balanced,
        # ex_l_iid_unbalanced,
        # ex_l_non_iid_balanced,
        # ex_l_non_iid_unbalanced,
    ]

    # ft = [False, True]
    # ft = [True]
    ft = [False]

    params_list = [
        # E, K, B
        [5, 3, 16],
        # [10, 3, 16],
        # [15, 3, 16],
        # [5, 5, 16],
        # [5, 10, 16],
        # [5, 3, 32],
        # [5, 3, 64]
    ]

    run_safe = False

    for smpc in ft:
        for params in params_list:
            args = TestParams()
            # args, dataset = mnist_net(args)
            args, dataset = completes_net(args)
            args.modelType = ModelType.FEDAVG
            args.smpc_aggregation = smpc
            # args.smpc_aggregation = True
            # args.smpc_aggregation = False
            args.dp = False

            args.rounds = 50
            args.epochs = params[0]
            args.n_nodes = params[1]
            args.batch_size = params[2]

            for experiment in experiments:
                if run_safe:
                    fed.run_experiment_safe(experiment.__name__, experiment(args), dataset)
                else:
                    fed.run_experiment(experiment.__name__, experiment(args), dataset)


def run_central():
    ### Central
    params_list = [
        # B
        # [8],
        [16],
        [32],
        [64]
    ]

    for params in params_list:
        args = TestParams()
        # args, dataset = mnist_net(args)
        args, dataset = completes_net(args)
        args.modelType = ModelType.CENTRAL
        args.dp = False

        args.epochs = 50
        args.batch_size = params[0]
        args.n_nodes = 0
        args.has_global_model = False
        args.smpc_aggregation = False
        
        fed.run_experiment("ex_central", args, dataset, 1)


def mnist_net(args: TestParams):
    args.layer_sizes = [784, 128, 128, 10]
    args.binary_classification = False
    args.lr = 0.01
    args.dropout = 0.5
    return args, load_data_mnist

def boston_net(args: TestParams):
    args.layer_sizes = [13, 128, 128, 1]
    args.binary_classification = False
    return args, load_data_boston

def completes_net(args: TestParams):
    args.layer_sizes = [27, 64, 2]
    args.binary_classification = True
    args.lr = 0.01
    args.dropout = 0.5
    return args, load_data_completes


# Clients with shared model
def ex_g_iid_balanced(args: TestParams):
    args.has_global_model = True
    args.data_iid = True
    args.data_balanced = True
    return args

def ex_g_iid_unbalanced(args: TestParams):
    args.has_global_model = True
    args.data_iid = True
    args.data_balanced = False
    return args

def ex_g_non_iid_balanced(args: TestParams):
    args.has_global_model = True
    args.data_iid = False
    args.data_balanced = True
    return args

def ex_g_non_iid_unbalanced(args: TestParams):
    args.has_global_model = True
    args.data_iid = False
    args.data_balanced = False
    return args


# Clients each have a local model
def ex_l_iid_balanced(args: TestParams):
    args.has_global_model = False
    args.data_iid = True
    args.data_balanced = True
    return args

def ex_l_iid_unbalanced(args: TestParams):
    args.has_global_model = False
    args.data_iid = True
    args.data_balanced = False
    return args

def ex_l_non_iid_balanced(args: TestParams):
    args.has_global_model = False
    args.data_iid = False
    args.data_balanced = True
    args.dirichlet_alpha = 0.95
    return args

def ex_l_non_iid_unbalanced(args: TestParams):
    args.has_global_model = False
    args.data_iid = False
    args.data_balanced = False
    args.dirichlet_alpha = 0.95
    return args

if __name__ == "__main__":
    main()