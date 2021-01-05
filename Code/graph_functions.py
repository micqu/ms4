from enum import Enum
import itertools
import glob
from typing import Tuple
from matplotlib.figure import Figure
import pandas as pd
import torch as th
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

import tikzplotlib

from model import *
from data_loader import *
from fed import test

class Key(Enum):
	TRAIN_LOSS = "train_loss"
	TRAIN_ACC = "train_accuracy"
	TRAIN_PRECISION = "train_precision"
	TRAIN_RECALL = "train_recall"
	TRAIN_AUC = "train_auc"
	TEST_LOSS = "test_loss"
	TEST_ACC = "test_accuracy"
	TEST_PRECISION = "test_precision"
	TEST_RECALL = "test_recall"
	TEST_AUC = "test_auc"
	EPSILON = "epsilon"
	BEST_ALPHA = "best_alpha"
	
class Label(Enum):
	COMM_ROUNDS = "Communication rounds"
	EPOCHS = "Epochs"
	TRAIN_LOSS = "Training loss"
	TRAIN_ACC = "Training accuracy"
	TRAIN_PRECISION = "Training precision"
	TRAIN_RECALL = "Training recall"
	TRAIN_AUC = "Training AUC"
	TEST_LOSS = "Test loss"
	TEST_ACC = "Test accuracy"
	TEST_PRECISION = "Test precision"
	TEST_RECALL = "Test recall"
	TEST_AUC = "Test AUC"
	LOSS = "Loss"
	EPSILON = "Epsilon"
	BEST_ALPHA = "Best alpha"

def plot_data_distribution_cm(fig: Figure, ax, data: np.array):
	d = data / np.sum(data.transpose(), axis=1)
	d = d.transpose()
	d = (np.round(d, decimals=2) * 100).astype(int)

	# Create plot
	#figsize = (5,5))
	ax.set_xticks(range(10))
	ax.set_yticks(range(10))
	ax.set_xticklabels(range(10))
	ax.set_yticklabels(range(10))
	ax.tick_params(left=False, bottom=False)

	color = np.array([
		np.repeat(9, 5),
		np.repeat(8, 5),
		np.repeat(7, 5),
		np.repeat(6, 5),
		np.repeat(5, 5),
		np.repeat(4, 5),
		np.repeat(3, 5),
		np.repeat(2, 5),
		np.repeat(1, 5),
		np.repeat(0, 5),
	])
	
	im = ax.imshow(d)
	annotate_heatmap(im, valfmt="{x:.0f}", textcoloridx=color, threshold=5)
	im = ax.imshow(color, cmap="viridis")

	for _, spine in ax.spines.items():
		spine.set_visible(True)
	
	# plt.ylabel('Label')
	# plt.xlabel('Client')

	fig.tight_layout()
	return fig, ax

def plot_multiple(
		paths: List[Tuple[str, str]],
		keys: List[Key],
		fmt="-",
		marker="", linestyle="-",
		loc="best",
		takeFirst=False,
		**plot_args):
	colors = plt.rcParams['axes.prop_cycle'].by_key()['color'].copy()
	cc = itertools.cycle(colors)

	linewidth = plot_args.get("linewidth", 1)

	legends_colors: List[Tuple[str, any]] = []
	plots_dir = []
	for i, (path, legend) in enumerate(paths):
		all_files = glob.glob(path + "/*.csv")
		
		color_per_dir = next(cc)
		dfs = [pd.read_csv(filename).dropna(axis=1, how='all').dropna(axis=0, how='any') for filename in all_files]
		# dfs = [pd.read_csv(filename) for filename in all_files]
		
		# Calc std deviation
		dfs_concat = pd.concat(dfs)
		by_row_idx = dfs_concat.groupby(dfs_concat.index)
		mean = by_row_idx.mean()
		std = by_row_idx.std(ddof=1)

		mean = mean.dropna(axis=1, how='all').dropna(axis=0, how='any')
		std = std.dropna(axis=1, how='all').dropna(axis=0, how='any')

		plot_keys = []
			# color_per_csv = adjust_lightness(color_per_dir, 0.7 + j * (1.0 - 0.7) / len(all_files))

			# color_per_csv = hex_to_hsv(color_per_dir)
			# color_per_csv[1] = 0.3 + j * (1.0 - 0.3) / len(all_files)
			# color_per_csv[1] = 1 - color_per_csv[1]
			# color_per_csv = hsv_to_hex(color_per_csv)
		
		first_plot_in_dir = True
		for key in keys:
			if plot_args.get("std") == True and std.get(key.value) is not None:
				if plot_args.get("col_per_dir") == True:
					if first_plot_in_dir:
						plot = plt.errorbar(range(len(mean)), mean[key.value], yerr=std[key.value], fmt=fmt, linewidth=linewidth)
						first_plot_in_dir = False
						col = plot[-1][-1].get_color()
						legends_colors.append((legend, col))
					else:
						col = plot_keys[-1][-1].get_color()
						legends_colors.append((legend, col))
						plot = plt.errorbar(range(len(mean)), mean[key.value], yerr=std[key.value], c=col, fmt=fmt, linewidth=linewidth)
				else:
					plot = plt.errorbar(range(len(mean)), mean[key.value], yerr=std[key.value], fmt=fmt, linewidth=linewidth)
				plot_keys.append(plot)
			else:
				if not takeFirst:
					for j, df in enumerate(dfs):
						data = df.get(key.value)
						if data is not None:
							if plot_args.get("col_per_dir") == True:
								if first_plot_in_dir:
									plot = plt.plot(data, marker=marker, linestyle=linestyle, linewidth=linewidth)
									first_plot_in_dir = False
									col = plot[-1].get_color()
									legends_colors.append((legend, col))
								else:
									col = plot_keys[-1][-1].get_color()
									legends_colors.append((legend, col))
									plot = plt.plot(data, c=col, marker=marker, linestyle=linestyle, linewidth=linewidth)
							else:
								plot = plt.plot(data, marker=marker, linestyle=linestyle, linewidth=linewidth)
							plot_keys.append(plot)
				else:
					data = dfs[0].get(key.value)
					if data is not None:
						if plot_args.get("col_per_dir") == True:
							if first_plot_in_dir:
								plot = plt.plot(data, marker=marker, linestyle=linestyle, linewidth=linewidth)
								first_plot_in_dir = False
								col = plot[-1].get_color()
								legends_colors.append((legend, col))
							else:
								col = plot_keys[-1][-1].get_color()
								legends_colors.append((legend, col))
								plot = plt.plot(data, c=col, marker=marker, linestyle=linestyle, linewidth=linewidth)
							# plot = plt.plot(data, c=color_per_dir, marker=marker, linestyle=linestyle)
						else:
							plot = plt.plot(data, marker=marker, linestyle=linestyle, linewidth=linewidth)
						plot_keys.append(plot)
		plots_dir.append(plot_keys)
	
	# 'best'	0
	# 'upper right'	1
	# 'upper left'	2
	# 'lower left'	3
	# 'lower right'	4
	# 'right'	5
	# 'center left'	6
	# 'center right'	7
	# 'lower center'	8
	# 'upper center'	9
	# 'center'	10

	legends = [legend for (_, legend) in paths]
	if len(keys) > 1:
		legends = np.repeat(legends, len(keys))
	# plt.legend(legends, loc=loc)

	return legends, plots_dir, legends_colors

def save(path: str, format: str=".tex", clean: bool=True):
	if clean:
		tikzplotlib.clean_figure()
	tikzplotlib.save(path + format)

def use_pgf():
	matplotlib.use("pgf")
	matplotlib.rcParams.update({
		"pgf.texsystem": "pdflatex",
		'font.family': 'serif',
		'text.usetex': True,
		'pgf.rcfonts': False,
	})

def hex_to_hsv(hex: str):
	rgb = matplotlib.colors.to_rgb(hex)
	hsv = matplotlib.colors.rgb_to_hsv(rgb)
	return hsv

def hsv_to_hex(hsv: List[float]):
	rgb = matplotlib.colors.hsv_to_rgb(hsv)
	hex = matplotlib.colors.to_hex(rgb)
	return hex

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def get_model_predictions(layer_sizes, state_file, load_data_func):
	model = load_model(layer_sizes, state_file)
	kwargs = {}
	train_data, test_data, X, y, X_test, y_test, all_data = load_data_func()
	test_loader = get_data_loader(test_data, 256, kwargs)
	
	device = th.device("cpu")
	targets, label_predictions, label_estimates, test_loss = test(model, test_loader, device)
	return targets, label_predictions, label_estimates, test_loss

def load_model(layer_sizes, state_file):
	model = Net(layer_sizes[0], layer_sizes[1:-1], layer_sizes[-1])
	model.load_state_dict(th.load(state_file))
	return model

def label(x: Label, y: Label):
	plt.xlabel(x.value)
	plt.ylabel(y.value)

def plot_roc(y_true, scores):
	fpr, tpr, thresholds = metrics.roc_curve(y_true, scores)
	# print(f"Thresholds: {thresholds}")
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic')
	return plt

def plot_cm(y_true, y_pred, figsize=(5,5), **args):
	cm = metrics.confusion_matrix(y_true, y_pred)

	# Normalize and filter values
	mask = None
	if not args.get("no_norm"):
		cm = (cm.T / np.sum(cm, axis=1)).T
		mask = np.empty_like(cm, dtype=bool)
		cm = (np.round(cm, decimals=2) * 100).astype(int)

		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			if cm[i, j] < 1:
				mask[i, j] = True
			else:
				mask[i, j] = False

	# Create plot
	fig, ax = plt.subplots(figsize = figsize)
	ax.set_xticks(range(10))
	ax.set_yticks(range(10))
	ax.set_xticklabels(range(10))
	ax.set_yticklabels(range(10))
	ax.tick_params(left=False, bottom=False)

	im = ax.imshow(cm, cmap="Greys")
	
	if mask is None:
		if args.get("invertThreshold") == True:
			annotate_heatmap(im, valfmt="{x:.0f}", invertThreshold=True)
		else:
			annotate_heatmap(im, valfmt="{x:.0f}")
	else:
		if args.get("invertThreshold") == True:
			annotate_heatmap(im, mask=mask, valfmt="{x:.0f}", invertThreshold=True)
		else:
			annotate_heatmap(im, mask=mask, valfmt="{x:.0f}")

	for _, spine in ax.spines.items():
		spine.set_visible(True)
	
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	# fig.tight_layout()
	return fig, ax

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
					 textcolors=["black", "white"], textcoloridx=None,
					 threshold=None, invertThreshold=False, mask=None, **textkw):
	"""
	A function to annotate a heatmap.

	Parameters
	----------
	im
		The AxesImage to be labeled.
	data
		Data used to annotate.  If None, the image's data is used.  Optional.
	valfmt
		The format of the annotations inside the heatmap.  This should either
		use the string format method, e.g. "$ {x:.2f}", or be a
		`matplotlib.ticker.Formatter`.  Optional.
	textcolors
		A list or array of two color specifications.  The first is used for
		values below a threshold, the second for those above.  Optional.
	threshold
		Value in data units according to which the colors from textcolors are
		applied.  If None (the default) uses the middle of the colormap as
		separation.  Optional.
	**kwargs
		All other arguments are forwarded to each call to `text` used to create
		the text labels.
	"""

	if not isinstance(data, (list, np.ndarray)):
		data = im.get_array()

	# Normalize the threshold to the images color range.
	if threshold is not None:
		threshold = im.norm(threshold)
	else:
		threshold = im.norm(data.max())/2.

	# Set default alignment to center, but allow it to be
	# overwritten by textkw.
	kw = dict(horizontalalignment="center",
			  verticalalignment="center")
	kw.update(textkw)

	# Get the formatter in case a string is supplied
	if isinstance(valfmt, str):
		valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

	# Loop over the data and create a `Text` for each "pixel".
	# Change the text's color depending on the data.
	texts = []
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			if textcoloridx is not None:
				if invertThreshold:
					kw.update(color=textcolors[int(im.norm(textcoloridx[i, j]) > threshold)])
				else:
					kw.update(color=textcolors[int(im.norm(textcoloridx[i, j]) < threshold)])
			else:
				if invertThreshold:
					kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
				else:
					kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
			if mask is not None:
				if not mask[i, j]:
					text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
			else:
				text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
				texts.append(text)

	return texts

## Uses seaborn
# def plot_confusion_matrix(y_true, y_pred):
# 	cm = metrics.confusion_matrix(y_true, y_pred)

# 	# # Binary classification
# 	# labels = ["True Neg", "False Pos", "False Neg", "True Pos"]
# 	# labels = np.asarray(labels).reshape(2,2)
# 	# sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')
# 	# sns.make_confusion_matrix(cf_matrix, figsize=(8,6), cbar=False)
# 	# plt.figure(figsize = (10,7))

# 	cm = cm / np.sum(cm, axis=1)
# 	mask = np.empty_like(cm, dtype=bool)
# 	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
# 		if cm[i, j] < 0.005:
# 			mask[i, j] = True
# 		else:
# 			mask[i, j] = False

# 	plt.figure(figsize = (5,5))#, edgecolor='k')
# 	cm = (np.round(cm, decimals=2) * 100).astype(int)

# 	annot = cm.astype(str)

# 	ax = sns.heatmap(cm, annot=annot, fmt="", cmap='Greys', square=True, cbar=False, mask=mask)
# 	for _, spine in ax.spines.items():
# 		spine.set_visible(True)
# 	ax.tick_params(left=False, bottom=False)

# 	plt.yticks(rotation=0)
# 	plt.ylabel('True label')
# 	plt.xlabel('Predicted label')
# 	return plt