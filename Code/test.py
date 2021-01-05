from typing import Dict
from numpy.core.numeric import NaN
from sklearn import metrics
from torch import log, nn

from node import *
from test_params import *
from data_loader import *


def test_with_metrics(
		model: Net, data_loader: DataLoader, metric_name_prefix: str,
		args: TestParams, device: th.device, **metric_params):
	
	y_true, y_pred, y_score, loss = test(model, data_loader, device)

	metric_results: Dict[str, float] = {}

	if metric_params.get("loss") == True:
		metric_results[metric_name_prefix + 'loss'] = loss

	if metric_params.get("acc") == True:
		acc = metrics.accuracy_score(y_true, y_pred)
		metric_results[metric_name_prefix + 'accuracy'] = acc

	if args.binary_classification == True:
		if metric_params.get("prec") == True:
			try:
				prec = metrics.precision_score(y_true, y_pred)
			except Exception:
				prec = NaN
			metric_results[metric_name_prefix + 'precision'] = prec

		if metric_params.get("rec") == True:
			try:
				rec = metrics.recall_score(y_true, y_pred)
			except Exception:
				rec = NaN
			metric_results[metric_name_prefix + 'recall'] = rec

		if metric_params.get("auc") == True:
			try:
				auc = metrics.roc_auc_score(y_true, y_score)
			except Exception:
				auc = NaN
			metric_results[metric_name_prefix + 'auc'] = auc

	return metric_results


def test(model: Net, test_loader: DataLoader, device: th.device):
	model.eval()
	test_loss = 0
	correct = 0
	label_predictions: List[int] = list()
	targets: List[int] = list()
	label_estimates: List[float] = list()

	with th.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			targets.extend([int(t) for t in target])

			pred = model(data)
			criterion = nn.NLLLoss()
			loss = criterion(log(pred), target)
			test_loss += loss * len(data)
			label_prediction = pred.argmax(1, keepdim=True) # get the index of the max log-probability

			for prediction in label_prediction:
				label_predictions.extend([int(p) for p in prediction])

			probabilities = pred
			label_estimates.extend([float(s) for s in probabilities[:, 1]])

			correct += label_prediction.eq(target.view_as(label_prediction)).sum().item()

	test_loss /= len(test_loader.dataset)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))
	
	return targets, label_predictions, label_estimates, float(test_loss)