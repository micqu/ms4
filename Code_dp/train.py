from typing import Dict
from torch import log, nn

from node import *

def train_central(global_node: Node, train_loader: DataLoader, args: TestParams, device: th.device):
	train_loss = 0
	global_node.model.train()
	for batch_idx, (X, y) in enumerate(train_loader):
		if batch_idx % args.log_interval == 0:
			print(f"{batch_idx}", end=" ")
		loss = update(global_node, X, y, device)
		train_loss += loss * len(X)
	
	train_loss /= len(train_loader)

	epsilon: float = 0
	best_alpha: float = 0
	if args.dp:
		epsilon, best_alpha = global_node.optimizer.privacy_engine.get_privacy_spent(args.dp_delta)
		print(f"(ε = {epsilon:.2f}, δ = {args.dp_delta}) for α = {best_alpha}")
	return float(train_loss), epsilon, best_alpha

def train(global_node: Node, nodes: List[Node], args: TestParams, device: th.device):
	train_losses: List[float] = []
	epsilons: List[float] = [0] * len(nodes)
	best_alphas: List[float] = [0] * len(nodes)

	for node_idx, node in enumerate(nodes):
		print(f"Training node {node_idx} ({args.epochs} epochs, {len(node.train_loader)} batches)", end="")
		
		train_loss = 0
		for epoch in range(args.epochs):
			print(f"\nEpoch {epoch}. Batch: ", end="")

			node.model.train()
			for batch_idx, (X, y) in enumerate(node.train_loader):
				if batch_idx % args.log_interval == 0:
					print(f"{batch_idx}", end=" ")
				loss = update(node, X, y, device)
				train_loss += loss.item() * len(X)

		train_loss /= len(node.train_loader)
		train_loss /= args.epochs
		train_losses.append(float(train_loss))
		
		if args.dp:
			epsilon, best_alpha = node.optimizer.privacy_engine.get_privacy_spent(args.dp_delta)
			epsilons[node_idx] = epsilon
			best_alphas[node_idx] = best_alpha
			print(f"(ε = {epsilon:.2f}, δ = {args.dp_delta}) for α = {best_alpha}")
			
		print()
		
	if args.has_global_model:
		with th.no_grad():
			# Aggregate parameters
			print("Aggregating")
			new_params = aggregate_weights(nodes)

			print("Updating global model")
			update_model(global_node, new_params)

			print("Updating all models")
			send_new_model(nodes, new_params)
	
	return train_losses, epsilons, best_alphas


def update(node: Node, data, target, device: th.device):
	# data, target = data.to(device), target.to(device)

	# Optimize
	node.optimizer.zero_grad()
	pred = node.model(data)
	criterion = nn.NLLLoss()
	loss = criterion(log(pred), target)
	loss.backward()
	node.optimizer.step()
	return loss

def aggregate_weights(nodes: List[Node]):
	new_params: List[int] = list()
	total_sample_count = sum([node.train_size for node in nodes])

	for param_idx in range(len(nodes[0].params)):
		# Averaging weighted by training data sizes on each node
		summed = sum([node.params[param_idx] * node.train_size for node in nodes]) / total_sample_count
		new_params.append(summed)
	return new_params


def send_new_model(nodes: List[Node], new_params: List[int]):
	for node in nodes:
		update_model(node, new_params)


def update_model(node: Node, new_params: List[int]):
	for param_index in range(len(node.params)):
		node.params[param_index].set_(new_params[param_index])