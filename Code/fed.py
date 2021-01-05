import logging
import os
import time
from typing import Callable

import syft as sy
import torch as th
import torch.optim as optim
from numpy import random

hook = sy.TorchHook(th)

from data_loader import *
from data_partitioner import *
from model import *
from node import *
from test_params import *
from train import *
from test import *


def create_model(args: TestParams, device: th.device):
	model = Net(
		args.layer_sizes[0],
		args.layer_sizes[1:-1],
		args.layer_sizes[-1],
		args.dropout
	).to(device)
	return model

def create_optimizer(params: Iterator[Parameter], args: TestParams):
	# optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, nesterov=True)
	# optimizer = optim.SGD(params, lr=args.lr, momentum=0.9)
	optimizer = optim.SGD(params, lr=args.lr)
	# optimizer = optim.Adam(params, lr=args.lr)
	return optimizer

def run_experiment_safe(experiment_name: str, args: TestParams, load_data_func: Callable, iteration=1):
	try:
		run_experiment(experiment_name, args, load_data_func, iteration)
	except BaseException as e:
		print(f"Oops! Experiment: {experiment_name} failed. Error: {e}")

def run_experiment(experiment_name: str, args: TestParams, load_data_func: Callable, iteration=1):
	print(f"Started experiment: {experiment_name}")
	print(f"Test params used: {vars(args)}")

	if args.modelType == ModelType.CENTRAL:
		args.has_global_model = True

	# Create experiment folder
	if not os.path.exists("results"):
		os.mkdir("results")
	
	# Setup CUDA
	use_cuda = not args.no_cuda and th.cuda.is_available()
	device: th.device = th.device("cuda" if use_cuda else "cpu")
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else{}

	for it in range(iteration):
		seed = args.seed + it
		np.random.seed(seed)
		th.manual_seed(seed)

		t = time.localtime()
		current_time = time.strftime("%H_%M_%S", t)
		ex_name = f"{experiment_name}"
		if args.smpc_aggregation:
			ex_name += "_smpc"
		if args.dp:
			ex_name += "_dp"
		ex_name += f"_{current_time}_s{seed}_e{args.epochs}_k{args.n_nodes}_b{args.batch_size}"
		experiment_path = os.path.join(args.log_dir, ex_name)
		os.mkdir(experiment_path)

		# Setup logging
		logging.basicConfig(
			filename=os.path.join(experiment_path, f"log-s{seed}.txt"),
			format='%(asctime)s %(levelname)-8s %(message)s',
			datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

		logging.debug(f"{ex_name}")
		logging.debug(f"{vars(args)}")
		logging.debug(f"Seed: {seed}")

		# Create nodes
		print(f"Creating {args.n_nodes} nodes...")
		global_node = create_global_node(create_model, create_optimizer, args, device)
		nodes = create_nodes(hook, args.n_nodes, create_model, create_optimizer, args, device)
		w_crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

		# Load all data
		print(f"Loading data...")
		train_data, test_data, X, y, X_test, y_test, all_data = load_data_func()

		# Create data loaders with global train/test data sets
		train_loader = get_data_loader(train_data, args.batch_size, kwargs)
		test_loader = get_data_loader(test_data, args.test_batch_size, kwargs)

		# Distribute data to nodes
		if args.modelType != ModelType.CENTRAL:
			if args.n_nodes > 0:
				print(f"Partitioning data...")
				if args.data_iid:
					data_map = create_partitioning_iid(all_data, args.n_nodes, args.data_balanced, args.unbalanced_min_part_min_size, args.unbalanced_min_part_max_size, args.dirichlet_alpha)
				else:
					data_map = create_partitioning_non_iid(all_data, args.layer_sizes[-1], args.n_nodes, args.data_balanced, args.unbalanced_min_part_min_size, args.unbalanced_min_part_max_size, args.dirichlet_alpha)
					
				train_data_map = {}
				test_data_map = {}
				y_all = all_data[:][1]
				for part_idx, data_idxs in data_map.items():
					part_X, part_X_test = model_selection.train_test_split(data_idxs, test_size=args.test_size, random_state=seed) #, stratify=y_all[data_idxs])
					train_data_map[part_idx] = part_X
					test_data_map[part_idx] = part_X_test
				
				train_partition_stats = get_data_partitioning_stats(y_all, train_data_map)
				logging.debug(f"Train partition stats: {train_partition_stats}")

				test_partition_stats = get_data_partitioning_stats(y_all, test_data_map)
				logging.debug(f"Test partition stats: {test_partition_stats}")


				print(f"Distributing data to nodes... ", end="")
				for i, node in enumerate(nodes):
					print(f"{i}", end=" ")
					local_train_loader = get_data_loader(all_data, args.batch_size, kwargs, train_data_map[i])
					local_test_loader = get_data_loader(all_data, args.test_batch_size, kwargs, test_data_map[i])

					node.train_loader = local_train_loader
					node.train_size = len(train_data_map[i])
					
					# Store local test loader on node for later ...
					node.test_loader = local_test_loader
					node.test_size = len(test_data_map[i])

					for batch_idx, (batch_X, batch_y) in enumerate(node.train_loader):
						batch_X_ptr = batch_X.send(node.virtual_worker)
						batch_y_ptr = batch_y.send(node.virtual_worker)
						node.batch_pointers.append((batch_X_ptr, batch_y_ptr))
			else:
				print("n_nodes is not greater than 0. No nodes available.")
		print()

		# # Add differential privacy engines
		# if args.dp:
		# 	if args.modelType == ModelType.CENTRAL:
		# 		print("Adding privacy engine to global node!")
		# 		attach_privacy_engine(global_node, args)
		# 	else:
		# 		print("Adding privacy engine to all local nodes!")
		# 		for node in nodes:
		# 			attach_privacy_engine(node, args)

		# Train and test
		total_time = 0
		average_iteration_time = 0
		t_start = start_clock()

		if args.modelType == ModelType.CENTRAL:
			# No federation, just classic central learning
			for epoch in range(args.epochs):
				print(f"\nEpoch {epoch}. Batch: ", end="")
				train_loss = train_central(global_node, train_loader, args, device)
				
				train_metric_results = test_with_metrics(global_node.model, train_loader, "train_", args, device,
					acc=True, prec=True, rec=True, auc=True)
				print(f"Global model train metrics: {train_metric_results}\n")
				
				test_metric_results = test_with_metrics(global_node.model, test_loader, "test_", args, device,
					loss=True, acc=True, prec=True, rec=True, auc=True)
				print(f"Global model test metrics: {test_metric_results}\n")

				global_node.metric_log.append([
					train_loss,
					train_metric_results.get("train_accuracy"),
					train_metric_results.get("train_precision"),
					train_metric_results.get("train_recall"),
					train_metric_results.get("train_auc"),
					test_metric_results.get("test_loss"),
					test_metric_results.get("test_accuracy"),
					test_metric_results.get("test_precision"),
					test_metric_results.get("test_recall"),
					test_metric_results.get("test_auc"),
				])

			total_time, average_iteration_time = stop_clock(t_start, args.epochs)
		else:
			# Federated learning
			for r in range(args.rounds):
				print(f"Round {r + 1}:")
				train_losses = train(global_node, nodes, w_crypto_provider, args, device)

				# Test all local models with their own local data
				for node_idx, node in enumerate(nodes):
					train_metric_results = test_with_metrics(node.model, node.train_loader, "train_", args, device,
						acc=True, prec=True, rec=True, auc=True)
					print(f"Node {node_idx} model train metrics: {train_metric_results}\n")

					test_metric_results = test_with_metrics(node.model, node.test_loader, "test_", args, device,
						loss=True, acc=True, prec=True, rec=True, auc=True)
					print(f"Node {node_idx} model test metrics: {test_metric_results}\n")

					node.metric_log.append([
						train_losses[node_idx],
						train_metric_results.get("train_accuracy"),
						train_metric_results.get("train_precision"),
						train_metric_results.get("train_recall"),
						train_metric_results.get("train_auc"),
						test_metric_results.get("test_loss"),
						test_metric_results.get("test_accuracy"),
						test_metric_results.get("test_precision"),
						test_metric_results.get("test_recall"),
						test_metric_results.get("test_auc"),
					])

			total_time, average_iteration_time = stop_clock(t_start, args.rounds)


		# Store timings
		logging.debug(f"Total time: {total_time}")
		logging.debug(f"Average time: {average_iteration_time}")
		

		# Store models and metrics
		if args.modelType == ModelType.CENTRAL:
			if args.save_model:
				th.save(global_node.model.state_dict(), f"{experiment_path}/global_e{args.epochs}_s{seed}.pt")

			df = create_metric_log_dataframe(global_node.metric_log)
			df.to_csv(f"{experiment_path}/metrics_global_node_s{seed}.csv")
		else:
			if args.save_model:
				for node_idx, node in enumerate(nodes):
					th.save(node.model.state_dict(), f"{experiment_path}/node_{node_idx}_r{args.rounds}_s{seed}.pt")

			for log_idx, node in enumerate(nodes):
				df = create_metric_log_dataframe(node.metric_log)
				df.to_csv(f"{experiment_path}/metrics_node_{log_idx}_s{seed}.csv")

		print("\nExperiment completed!\n")


def start_clock():
	return time.perf_counter()


def stop_clock(t_start: float, n_iterations: int):
	total_time = time.perf_counter() - t_start
	print('Total time', round(total_time, 2), 's')

	avg_time = total_time / n_iterations
	print('Average iteration time', round(avg_time, 2), 's')

	return total_time, avg_time

def create_metric_log_dataframe(data):
	df = pd.DataFrame(data=data, columns=[
		"train_loss",
		"train_accuracy",
		"train_precision",
		"train_recall",
		"train_auc",
		"test_loss",
		"test_accuracy",
		"test_precision",
		"test_recall",
		"test_auc"
	])
	return df