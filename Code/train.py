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

	# epsilon: float = -1
	# best_alpha: float = -1
	# if args.dp:
	# 	epsilon, best_alpha = global_node.optimizer.privacy_engine.get_privacy_spent(args.dp_delta)
	# 	print(f"(ε = {epsilon:.2f}, δ = {args.dp_delta}) for α = {best_alpha}")
	# return epsilon, best_alpha
	return float(train_loss)

def train(global_node: Node, nodes: List[Node], crypto_provider: VirtualWorker, args: TestParams, device: th.device):
	train_losses: List[float] = []
	# epsilons: Dict[int, List[float]] = {}
	# alphas: Dict[int, List[float]] = {}

	for node_idx, node in enumerate(nodes):
		print(f"Training node {node_idx} ({args.epochs} epochs, {len(node.batch_pointers)} batches)", end="")
		
		train_loss = 0
		for epoch in range(args.epochs):
			print(f"\nEpoch {epoch}. Batch: ", end="")

			node.model.train()
			for batch_idx, (X, y) in enumerate(node.batch_pointers):
				if batch_idx % args.log_interval == 0:
					print(f"{batch_idx}", end=" ")
				# Send model to data location
				node.model.send(X.location)
				loss = update(node, X, y, device)
				train_loss += loss.get() * len(X)
				node.model.get()

		train_loss /= len(node.batch_pointers)
		train_loss /= args.epochs
		train_losses.append(float(train_loss))
		
		# if args.dp:
		# 	epsilon, best_alpha = node.optimizer.privacy_engine.get_privacy_spent(args.dp_delta)
		# 	epsilons[node_idx].append(epsilon)
		# 	alphas[node_idx].append(best_alpha)
		# 	print(f"(ε = {epsilon:.2f}, δ = {args.dp_delta}) for α = {best_alpha}")
			
		print()
		
	if args.has_global_model:
		with th.no_grad():
			# Aggregate parameters
			if args.smpc_aggregation:
				print("Aggregating (SMPC)")
				new_params = aggregate_weights_smpc(nodes, crypto_provider)
			else:
				print("Aggregating")
				new_params = aggregate_weights(nodes)

			print("Updating global model")
			update_model(global_node, new_params)

			print("Updating all models")
			send_new_model(nodes, new_params)
	
	# return epsilons, alphas
	return train_losses


def update(node: Node, data, target, device: th.device):
	# data, target = data.to(device), target.to(device)

	# Optimize
	node.optimizer.zero_grad()
	pred = node.model(data)
	criterion = nn.NLLLoss()
	loss = criterion(log(pred), target)
	loss.backward()
	node.optimizer.step()
	
	sy.local_worker.clear_objects()
	return loss

def aggregate_weights(nodes: List[Node]):
	new_params: List[int] = list()
	total_sample_count = sum([node.train_size for node in nodes])

	for param_idx in range(len(nodes[0].params)):
		# Averaging weighted by training data sizes on each node
		summed = sum([node.params[param_idx].copy() * node.train_size for node in nodes]) / total_sample_count
		new_params.append(summed)
	return new_params


def aggregate_weights_smpc(nodes: List[Node], crypto_provider: VirtualWorker):
	# Encrypted aggregation
	virtual_workers = [w.virtual_worker for w in nodes]
	new_params: List[int] = list()
	total_sample_count = sum([node.train_size for node in nodes])

	for param_idx in range(len(nodes[0].params)):
		spdz_params = list()
		for node in nodes:
			param_copy = node.params[param_idx].copy() * node.train_size # Multiply for weighted avg
			param_fixed_precision = param_copy.fix_precision()
			param_encrypted = param_fixed_precision.share(*virtual_workers, crypto_provider=crypto_provider)
			# param = param_encrypted.get()
			spdz_params.append(param_encrypted)

		# Averaging weighted by training data sizes on each node
		new_param = sum(spdz_params).get().float_precision() / total_sample_count
		new_params.append(new_param)
	return new_params


def send_new_model(nodes: List[Node], new_params: List[int]):
	for node in nodes:
		update_model(node, new_params)


def update_model(node: Node, new_params: List[int]):
	for param_index in range(len(node.params)):
		node.params[param_index].set_(new_params[param_index])