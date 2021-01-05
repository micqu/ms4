import numpy as np
import matplotlib.pyplot as plt
import shap

from model import *
from data_loader import *
from graph_functions import *

loop_values = [
	(Key.TRAIN_ACC, Label.TRAIN_ACC),
	(Key.TRAIN_LOSS, Label.TRAIN_LOSS),
	(Key.TEST_ACC, Label.TEST_ACC),
	(Key.TEST_LOSS, Label.TEST_LOSS),
]

def main():
	# plot_confusion_matrix()
	# plot_data_distribution_hist()

	# use_pgf()
	# plot_central()
	# plot_smpc()
	# plot_dp()
	# plot_dp_epsilon()

	# explain_mnist()
	explain_completes()

	## Plot binary ROC curve
	# plt = plot_roc(targets, label_estimates)

	print("Done!")

####
def explain_mnist():
	## Explainability
	layer_sizes = [784, 128, 128, 10]

	state_files = [
		# Central B=16
		r"Code\results\mnist\central\ex_central_16_34_38_s12346_e50_k0_b16\global_e50_s12346.pt",
		## E
		r"Code\results\mnist\no smpc\ex_g_iid_balanced_01_43_22_s12346_e5_k3_b16\node_0_r50_s12346.pt",
		r"Code_dp\results\ex_g_iid_balanced_dp_23_31_51_s12346_e5_k3_b16\node_0_r50_s12346_44.pt",
		## B
		# r"Code\results\mnist\no smpc\ex_g_iid_balanced_15_39_37_s12346_e5_k3_b64\node_0_r50_s12346.pt",
		# r"Code_dp\results\mnist\ex_g_iid_balanced_dp_20_20_43_s12346_e5_k3_b64\node_0_r50_s12346.pt",
		## K
		# r"Code\results\mnist\no smpc\ex_g_iid_balanced_12_10_50_s12346_e5_k10_b16\node_0_r50_s12346.pt",
		# r"Code_dp\results\ex_g_iid_balanced_dp_23_57_09_s12346_e5_k10_b16\node_0_r50_s12346_40.pt",
	]

	seed = 1
	np.random.seed(seed)
	th.manual_seed(seed)

	train_data, test_data, X, y, X_test, y_test, all_data = load_data_mnist()
	test_loader = get_data_loader(test_data, 256, {}, data_idxs=range(103), shuffle=False)
	batch = next(iter(test_loader))
	images, _ = batch

	background = images[:100]
	test_images = images[100:103]

	for i, state_file in enumerate(state_files):
		model = load_model(layer_sizes, state_file)
		e = shap.DeepExplainer(model, background)
		shap_values = e.shap_values(test_images)

		shap_numpy = [s.reshape((-1, 28, 28)) for s in shap_values]
		test_numpy = np.array([t.reshape((-1, 28, 28)) for t in test_images.numpy()]).squeeze()

		# plot the feature attributions
		# shap.image_plot(shap_numpy, test_numpy)
		shap.image_plot(shap_numpy, test_numpy, show=False)
		plt.savefig(f"{i}.png")


def explain_completes():
	layer_sizes = [27, 64, 2]
	state_files = [
		# Central B=16
		(r"Code_dp\results\central\real\ex_central_iid_balanced_17_54_09_s12346_e50_k3_b16\global_e50_s12346.pt", "real_explain_central_e5_k3_b16"),
		
		## E
		(r"Code\results\real\no smpc reduced data\ex_g_iid_balanced_02_36_44_s12346_e5_k3_b16\node_0_r50_s12346.pt", "real_explain_g_iid_balanced_e5_k3_b16"),
		(r"Code_dp\results\ex_g_iid_balanced_dp_00_37_44_s12346_e5_k3_b16\node_0_r50_s12346.pt", "real_explain_g_dp_iid_balanced_e5_k3_b16")

		## B
		# r"Code\results\real\no smpc\ex_g_iid_balanced_00_13_54_s12346_e5_k3_b64\node_0_r50_s12346.pt",
		# r"Code_dp\results\real\ex_g_iid_balanced_dp_02_20_55_s12346_e5_k3_b64\node_0_r50_s12346.pt",

		## K
		# r"Code\results\real\no smpc\ex_g_iid_balanced_23_36_28_s12346_e5_k10_b16\node_0_r50_s12346.pt",
		# r"Code_dp\results\real\ex_g_iid_balanced_dp_01_36_30_s12346_e5_k10_b16\node_0_r50_s12346.pt",
	]

	seed = 12346
	np.random.seed(seed)
	th.manual_seed(seed)

	train_data, test_data, X, y, X_test, y_test, all_data = load_data_completes()
	test_loader = get_data_loader(test_data, 256, {}, data_idxs=range(200), shuffle=False)
	batch = next(iter(test_loader))
	samples, _ = batch

	background = samples[:100]
	test_samples = samples[100:103]

	# plot the feature attributions
	feature_names = get_column_names(r"../data/data_all_2020_12_10/processed/completes_with_embedding_no_zeros.csv")
	r = None
	for i, (state_file, file_name) in enumerate(state_files):
		model = load_model(layer_sizes, state_file)
		e = shap.DeepExplainer(model, background)
		shap_values = e.shap_values(test_samples)

		# plot the feature attributions
		# plt.clf()
		# plt.figure(figsize=(20,5))
		# shap.summary_plot(shap_values[1], feature_names=feature_names[0:-1], max_display=20)
		plt.figure(figsize=(4,5))
		# plt.gcf().subplots_adjust(left=0.50)
		if i == 0:
			r = shap.decision_plot(e.expected_value[1], shap_values[1], feature_names=feature_names[0:-1], auto_size_plot=False, show=False, return_objects=True, xlim=[0, 1], new_base_value=0.5)
		else:
			shap.decision_plot(e.expected_value[1], shap_values[1], feature_names=feature_names[0:-1], auto_size_plot=False, show=False, feature_order=r.feature_idx, xlim=[0, 1], new_base_value=0.5)
		plt.tight_layout()
		plt.savefig(f"{file_name}.png")
		plt.show()

def plot_dp():
	kl = [
		(Key.TRAIN_ACC, Label.TRAIN_ACC),
		(Key.TRAIN_LOSS, Label.TRAIN_LOSS),
		(Key.TRAIN_PRECISION, Label.TRAIN_PRECISION),
		(Key.TRAIN_RECALL, Label.TRAIN_RECALL),
		(Key.TRAIN_AUC, Label.TRAIN_AUC),

		# (Key.TEST_ACC, Label.TEST_ACC),
		# (Key.TEST_LOSS, Label.TEST_LOSS),
		# (Key.TEST_PRECISION, Label.TEST_PRECISION),
		# (Key.TEST_RECALL, Label.TEST_RECALL),
		# (Key.TEST_AUC, Label.TEST_AUC),
	]
	for key, y_label in kl:
		fig, ax = plt.subplots()
		legends = []
		l, _, legends_colors = plot_multiple([
			## Plot central B=16
			## MNIST
			# (r"Code\results\mnist\central\ex_central_16_34_38_s12346_e50_k0_b16", "Central B=16"),
			# (r"Code_dp\results\central\mnist\ex_central_iid_balanced_15_32_45_s12346_e50_k3_b16", "Central B=16"),

			## Real
			# (r"Code\results\real\central\ex_central_15_50_57_s12346_e50_k0_b16", "Central B=16"),
			(r"Code_dp\results\central\real\ex_central_iid_balanced_17_54_09_s12346_e50_k3_b16", "Central B=16"),
		], [key], marker="", std=True, col_per_dir=True, linewidth=2)
		# ], [key], marker="x", std=False, col_per_dir=False)
		legends.extend(l)

		l, _, legends_colors = plot_multiple([
			## MNIST
			## E global IID balanced
			# (r"Code\results\mnist\no smpc\ex_g_iid_balanced_01_43_22_s12346_e5_k3_b16", "FA E=5"),
			# (r"Code\results\mnist\no smpc\ex_g_iid_balanced_05_57_06_s12346_e15_k3_b16", "FA E=15"),
			## E global IID unbalanced
			# (r"Code\results\mnist\no smpc\ex_g_iid_unbalanced_01_53_27_s12346_e5_k3_b16", "FA E=5"),
			# (r"Code\results\mnist\no smpc\ex_g_iid_unbalanced_06_31_47_s12346_e15_k3_b16", "FA E=15"),
			## E global non-IID balanced
			# (r"Code\results\mnist\no smpc\ex_g_non_iid_balanced_02_03_17_s12346_e5_k3_b16", "FA E=5"),
			# (r"Code\results\mnist\no smpc\ex_g_non_iid_balanced_07_05_08_s12346_e15_k3_b16", "FA E=15"),
			## E global non-IID unbalanced
			# (r"Code\results\mnist\no smpc\ex_g_non_iid_unbalanced_02_13_14_s12346_e5_k3_b16", "FA E=5"),
			# (r"Code\results\mnist\no smpc\ex_g_non_iid_unbalanced_07_40_00_s12346_e15_k3_b16", "FA E=15"),


			## E global IID balanced
			# (r"Code\results\mnist\no smpc\ex_g_iid_balanced_01_43_22_s12346_e5_k3_b16", "FA E=5"),
			# (r"Code_dp\results\mnist\ex_g_iid_balanced_dp_01_14_11_s12346_e5_k3_b16", "FA DP E=5"),
			# (r"Code\results\mnist\no smpc\ex_g_iid_balanced_05_57_06_s12346_e15_k3_b16", "FA E=15"),
			# (r"Code_dp\results\mnist\ex_g_iid_balanced_dp_17_03_14_s12346_e15_k3_b16", "FA DP E=15"),
			## E global IID unbalanced
			## E global non-IID balanced
			## E global non-IID unbalanced


			## B
			# (r"Code\results\mnist\no smpc\ex_g_iid_balanced_01_43_22_s12346_e5_k3_b16", "FA B=16"),
			# (r"Code_dp\results\mnist\ex_g_iid_balanced_dp_01_14_11_s12346_e5_k3_b16", "FA DP B=16"),
			# (r"Code\results\mnist\no smpc\ex_g_iid_balanced_15_39_37_s12346_e5_k3_b64", "FA B=64"),
			# (r"Code_dp\results\mnist\ex_g_iid_balanced_dp_20_20_43_s12346_e5_k3_b64", "FA DP B=64"),

			## Real
			## E
			# (r"Code\results\real\no smpc\ex_g_iid_balanced_21_01_37_s12346_e5_k3_b16", "FA E=5"),
			# (r"Code_dp\results\real\ex_g_iid_balanced_dp_00_53_10_s12346_e5_k3_b16", "FA DP E=5"),
			# (r"Code\results\real\no smpc\ex_g_iid_balanced_22_07_47_s12346_e15_k3_b16", "FA E=15"),
			# (r"Code_dp\results\real\ex_g_iid_balanced_dp_01_11_22_s12346_e15_k3_b16", "FA DP E=15"),
			## B
			# (r"Code\results\real\no smpc\ex_g_iid_balanced_21_01_37_s12346_e5_k3_b16", "FA B=16"),
			# (r"Code_dp\results\real\ex_g_iid_balanced_dp_00_53_10_s12346_e5_k3_b16", "FA DP B=16"),
			# (r"Code\results\real\no smpc\ex_g_iid_balanced_00_13_54_s12346_e5_k3_b64", "FA B=64"),
			# (r"Code_dp\results\real\ex_g_iid_balanced_dp_02_20_55_s12346_e5_k3_b64", "FA DP B=64"),
		], [key], fmt="-", std=True, col_per_dir=False, takeFirst=False)
		legends.extend(l)

		# plt.legend(legends, loc="lower center")
		plt.legend(legends, loc="best")
		plt.grid(True, which='both')
		# plt.title("Federated Averaging")
		label(Label.COMM_ROUNDS, y_label)
		# save("mnist_g_dp_iid_balanced_" + key.name.lower(), clean=False)
		# save("mnist_g_dp_iid_balanced_" + key.name.lower() + "_batch", clean=False)

		# save("real_g_dp_iid_balanced_" + key.name.lower(), clean=False)
		save("real_g_dp_iid_balanced_" + key.name.lower() + "_batch", clean=False)
		plt.show()

	# save("mnist_g_dp_iid_balanced_train_accs", clean=False)
	# save("mnist_g_dp_iid_balanced_train_loss", clean=False)
	# save("mnist_g_dp_iid_balanced_test_accs", clean=False)
	# save("mnist_g_dp_iid_balanced_test_loss", clean=False)
	# save("mnist_g_dp_iid_balanced_epsilon")
	# save("mnist_g_dp_iid_balanced_epsilon_batch")
	
	# ax2 = ax.twinx()
	# plot_single((r"results\ex_global_iid_balanced_dp_00_24_28", "FedAvg (DP) epsilon"), [Key.EPSILON], colors=colors[2:], std=True, col_per_dir=True)
	# plt.title("Federated Averaging")
	# plt.show()

	# fig, ax = plt.subplots()
	# plot_multiple([
	# 	(r"results\ex_global_iid_balanced_dp_00_24_28", "FedAvg (DP) E=5, B=16"),
	# ], [Key.TRAIN_ACCURACY, Key.TEST_ACCURACY], fmt="-", loc="lower right", colors=colors, std=True, col_per_dir=True)

	# plt.grid(True, which='both')
	# plt.title("Federated Averaging")
	# label(Label.COMM_ROUNDS, Label.TRAIN_ACCURACY)
	
	# ax2 = ax.twinx()
	# plot_single((r"results\ex_global_iid_balanced_dp_00_24_28", "FedAvg (DP) epsilon"), [Key.EPSILON], colors=colors[2:], std=True, col_per_dir=True)
	# plt.title("Federated Averaging")
	# plt.show()

def plot_dp_epsilon():
	kl = [
		(Key.EPSILON, Label.EPSILON)
	]
	for key, y_label in kl:
		fig, ax = plt.subplots()
		legends = []
		l, _, _ = plot_multiple([
			## MNIST
			## E
			# (r"Code_dp\results\mnist\ex_g_iid_balanced_dp_01_14_11_s12346_e5_k3_b16", "DP E=5"),
			# (r"Code_dp\results\mnist\ex_g_iid_balanced_dp_17_03_14_s12346_e15_k3_b16", "DP E=15"),
			## B
			# (r"Code_dp\results\mnist\ex_g_iid_balanced_dp_01_14_11_s12346_e5_k3_b16", "DP B=16"),
			# (r"Code_dp\results\mnist\ex_g_iid_balanced_dp_20_20_43_s12346_e5_k3_b64", "DP B=64"),
			
			## Real
			## E
			# (r"Code_dp\results\real\ex_g_iid_balanced_dp_00_53_10_s12346_e5_k3_b16", "DP E=5"),
			# (r"Code_dp\results\real\ex_g_iid_balanced_dp_01_11_22_s12346_e15_k3_b16", "DP E=15"),
			## B
			(r"Code_dp\results\real\ex_g_iid_balanced_dp_00_53_10_s12346_e5_k3_b16", "DP B=16"),
			(r"Code_dp\results\real\ex_g_iid_balanced_dp_02_20_55_s12346_e5_k3_b64", "DP B=64"),
		], [key], fmt="-", std=False, col_per_dir=False, takeFirst=True)
		legends.extend(l)

		# plt.legend(legends, loc="lower center")
		plt.legend(legends, loc="best")
		plt.grid(True, which='both')
		plt.title("Federated Averaging")
		label(Label.COMM_ROUNDS, y_label)
		# save("mnist_g_dp_iid_balanced_" + key.name.lower(), clean=True)
		# save("mnist_g_dp_iid_balanced_" + key.name.lower() + "_batch", clean=True)

		# save("real_g_dp_iid_balanced_" + key.name.lower(), clean=True)
		save("real_g_dp_iid_balanced_" + key.name.lower() + "_batch", clean=True)
		plt.show()

def plot_smpc():
	fig, ax = plt.subplots()

	plot_multiple([
		# (r"results\no smpc no dp\ex_global_iid_balanced_16_50_19", "E=5"),
		# (r"results\smpc no dp\ex_global_iid_balanced_smpc_15_17_39x", "SMPC E=5"),
		# (r"results\no smpc no dp\ex_global_iid_balanced_23_28_19", "E=15"),
		# (r"results\smpc no dp\ex_global_iid_balanced_smpc_15_27_43", "SMPC E=15"),

		(r"results\no smpc no dp\ex_global_iid_balanced_11_29_00", "B=64"),
		(r"results\smpc no dp\ex_global_iid_balanced_smpc_16_09_23", "SMPC B=64"),
	], [Key.TEST_ACCURACY], fmt="-", std=True, col_per_dir=False)

	plt.grid(True, which='both')
	plt.title("Federated Averaging")
	label(Label.COMM_ROUNDS, Label.TEST_ACCURACY)

	## Train and Test (Epoch)
	# ax.set_xlim(0, 17.5)
	# ax.set_ylim(0.85, 1.02)
	# save("mnist_global_smpc_iid_balanced_train_accs_batch", clean=False)
	# ax.set_xlim(0, 17.5)
	# ax.set_ylim(0.75, 0.92)
	# save("mnist_global_smpc_iid_balanced_test_accs", clean=False)
	
	## Train and Test (Batch)
	# ax.set_xlim(0, 20.5)
	# ax.set_ylim(0.7, 1.0)
	# save("mnist_global_smpc_iid_balanced_train_accs_batch", clean=False)
	ax.set_xlim(0, 20.5)
	ax.set_ylim(0.6, 0.90)
	save("mnist_global_smpc_iid_balanced_test_accs_batch", clean=False)

	# axins = zoomed_inset_axes(ax, 2, loc=1)
	# axins.set_xlabel("")
	# axins.set_ylabel("")
	# axins.set_xlim(2, 9)
	# axins.set_ylim(0.92, 0.98)
	# plt.xticks(visible=False)
	# plt.yticks(visible=False)
	# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
	plt.show()

def plot_central():
	## Plot central
	metrics = [
		("results\central sgd mnist2\ex_central_15_38_29", "B = 16"),
		("results\central sgd mnist2\ex_central_15_38_46", "B = 32"),
		("results\central sgd mnist2\ex_central_15_38_57", "B = 64")
	]
	plt.figure()
	# plot_multiple(metrics, [Key.TRAIN_ACCURACY])
	# plot_multiple(metrics, [Key.TEST_ACCURACY])
	# plot_multiple(metrics, [Key.TRAIN_LOSS])
	plot_multiple(metrics, [Key.TEST_LOSS])
	plt.title("Central")
	# label(Label.EPOCHS, Label.TRAIN_ACCURACY)
	# label(Label.EPOCHS, Label.TEST_ACCURACY)
	# label(Label.EPOCHS, Label.TRAIN_LOSS)
	label(Label.EPOCHS, Label.TEST_LOSS)
	plt.grid(True, which='both')
	# save("mnist_central_train_accs")
	save("mnist_central_loss")
	plt.show()

def plot_confusion_matrix():
	## Confusion matrix
	layer_sizes = [784, 128, 128, 10]
	state_file = r"results\central sgd mnist\ex_central_15_38_29\global_r30.pt"
	targets, label_predictions, label_estimates, test_loss = get_model_predictions(layer_sizes, state_file)
	
	fig, ax = plot_cm(targets, label_predictions)
	plt.title('')
	plt.xlabel('')
	plt.ylabel('')
	plt.show()
	# plt.savefig("mnist_central_cm.png")

def plot_data_distribution_hist():
	# data = np.array([
	# 	# IID balanced
	# 	[29, 29, 34, 31, 33, 36, 33, 29, 37, 29],
	# 	[34, 33, 25, 38, 29, 31, 31, 32, 37, 30],
	# 	[31, 34, 36, 27, 33, 27, 34, 39, 26, 33],
	# 	[28, 31, 28, 37, 31, 32, 24, 35, 34, 40],
	# 	[38, 33, 37, 27, 34, 34, 38, 25, 26, 28]
	# ])

	# data = np.array([
	# 	# Non-IID balanced
	# 	[53, 0, 52, 77, 7, 27, 0, 71, 5, 32],
	# 	[12, 78, 0, 60, 0, 15, 25, 5, 12, 107],
	# 	[88, 17, 43, 12, 26, 111, 1, 0, 28, 0],
	# 	[0, 44, 61, 0, 12, 2, 38, 66, 80, 18],
	# 	[2, 20, 5, 10, 115, 3, 98, 17, 32, 11]
	# ])

	# data = np.array([
	# 	# IID unbalanced
	# 	[54, 53, 55, 50, 52, 56, 58, 52, 65, 50],
	# 	[76, 79, 71, 84, 79, 73, 75, 92, 73, 78],
	# 	[12, 7, 15, 7, 9, 15, 15, 6, 14, 10],
	# 	[11, 6, 8, 6, 8, 10, 10, 7, 6, 9],
	# 	[11, 8, 10, 8, 6, 10, 6, 9, 7, 7]
	# ])
	data = np.array([
		# nonIID balanced test new logic
		[123, 81, 60, 29, 16, 8, 3, 0, 0, 0],
		[32, 60, 63, 73, 48, 24, 11, 6, 2, 1],
		[2, 14, 29, 53, 66, 64, 45, 22, 19, 6],
		[0, 2, 6, 9, 26, 49, 73, 65, 57, 33],
		[0, 0, 0, 1, 6, 14, 30, 67, 81, 121]
	])

	# data = np.array([
	# 	# Non-IID unbalanced
	# 	[18, 5, 14, 23, 0, 34, 5, 0, 57, 3],
	# 	[70, 1, 1, 10, 45, 82, 0, 68, 62, 2],
	# 	[36, 19, 26, 86, 0, 30, 54, 51, 48, 22],
	# 	[32, 5, 104, 0, 75, 9, 17, 31, 5, 99],
	# 	[1, 127, 8, 34, 45, 6, 83, 9, 0, 35]
	# ])

	## Plot histogram
	fig, ax = plt.subplots(figsize=(4,4))
	datat = data.transpose()[::-1]
	color_map = plt.get_cmap("viridis")
	for i, row in enumerate(datat):
		cidx = round(i * 255 / len(datat))
		if i != 0:
			data_cumsum = sum([datat[r] for r in range(i)])
			plt.bar(range(len(row) + 1)[1:], row, bottom=data_cumsum, width=0.9, color=color_map(cidx))
		else:
			plt.bar(range(len(row) + 1)[1:], row, width=.9, color=color_map(cidx))
	
	ax.tick_params(bottom=False)
	plt.xlabel('')
	plt.ylabel('')
	# save("mnist_data_distribution_iid_balanced_hist")

	# plt.savefig("mnist_data_distribution_iid_balanced_hist.png")
	# plt.savefig("mnist_data_distribution_noniid_balanced_hist.png")
	# plt.savefig("mnist_data_distribution_noniid_unbalanced_hist.png")
	# plt.savefig("mnist_data_distribution_iid_unbalanced_hist.png")
	plt.savefig("mnist_data_distribution_noniid_balanced_hist.png")
	plt.show()

if __name__ == "__main__":
    main()