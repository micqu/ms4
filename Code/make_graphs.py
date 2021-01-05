import numpy as np
import matplotlib.pyplot as plt
import shap
import itertools
import operator

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
	# plot_sample()
	# plot_confusion_matrix()
	# plot_data_distribution_hist()

	# use_pgf()
	# plot_central()
	# plot_global()
	plot_local_real()
	# plot_local_mnist()
	# plot_smpc()

	# explain_mnist()
	# explain_completes()

	print("Done!")

####
def plot_local_real():
	kl = [
		(Key.TRAIN_ACC, Label.TRAIN_ACC),
		(Key.TRAIN_LOSS, Label.TRAIN_LOSS),
		(Key.TRAIN_PRECISION, Label.TRAIN_PRECISION),
		(Key.TRAIN_RECALL, Label.TRAIN_RECALL),
		(Key.TRAIN_AUC, Label.TRAIN_AUC),

		(Key.TEST_ACC, Label.TEST_ACC),
		(Key.TEST_LOSS, Label.TEST_LOSS),
		(Key.TEST_PRECISION, Label.TEST_PRECISION),
		(Key.TEST_RECALL, Label.TEST_RECALL),
		(Key.TEST_AUC, Label.TEST_AUC),
	]

	for key, y_label in kl:
		fig, ax = plt.subplots()
		legends = []
		plots_dirs = []
		l, plots_dir, legends_colors = plot_multiple([
			## Plot central B=16
			# (r"Code\results\real\central\ex_central_15_50_57_s12346_e50_k0_b16", "Central B=16"),

			# (r"Code_dp\results\central\real\ex_central_iid_balanced_17_54_09_s12346_e50_k3_b16", "Central B=16"),
			# (r"Code_dp\results\central\real\ex_central_iid_unbalanced_17_54_19_s12346_e50_k3_b16", "Central B=16"),
			# (r"Code_dp\results\central\real\ex_central_non_iid_balanced_17_54_28_s12346_e50_k3_b16", "Central B=16"),
			# (r"Code_dp\results\ex_central_non_iid_balanced_21_54_54_s12347_e50_k3_b16", "Central B=16"),		# Higher dirichlet
			# (r"Code_dp\results\central\real\ex_central_non_iid_unbalanced_17_54_37_s12346_e50_k3_b16", "Central B=16"),
			(r"Code_dp\results\ex_central_non_iid_unbalanced_21_55_04_s12347_e50_k3_b16", "Central B=16"),		# Higher dirichlet
		], [key], marker="", std=False, col_per_dir=True, linewidth=2)
		legends.extend(l)
		plots_dirs.extend(plots_dir)

		l, plots_dir, legends_colors = plot_multiple([
			## Real
			## E global IID balanced
			# (r"Code\results\real\no smpc\ex_g_iid_balanced_21_01_37_s12346_e5_k3_b16", "FA E=5"),
			# (r"Code\results\real\no smpc\ex_g_iid_balanced_22_07_47_s12346_e15_k3_b16", "FA E=15"),
			## E global IID unbalanced
			# (r"Code\results\real\no smpc\ex_g_iid_unbalanced_21_04_26_s12346_e5_k3_b16", "FA E=5"),
			# (r"Code\results\real\no smpc\ex_g_iid_unbalanced_22_16_05_s12346_e15_k3_b16", "FA E=15"),
			## E global non-IID balanced
			# (r"Code\results\real\no smpc\ex_g_non_iid_balanced_21_07_11_s12346_e5_k3_b16", "FA E=5"),
			# (r"Code\results\real\no smpc\ex_g_non_iid_balanced_22_24_15_s12346_e15_k3_b16", "FA E=15"),
			# (r"Code_dp\results\real_non_iid_redo\ex_g_non_iid_balanced_20_27_31_s12347_e5_k3_b16", "FA E=5"),	# Higher dirichlet
			# (r"Code_dp\results\real_non_iid_redo\ex_g_non_iid_balanced_20_28_45_s12347_e15_k3_b16", "FA E=15"),	# Higher dirichlet
			## E global non-IID unbalanced
			# (r"Code\results\real\no smpc\ex_g_non_iid_unbalanced_21_09_59_s12346_e5_k3_b16", "FA E=5"),
			# (r"Code\results\real\no smpc\ex_g_non_iid_unbalanced_22_32_27_s12346_e15_k3_b16", "FA E=15"),
			(r"Code_dp\results\real_non_iid_redo\ex_g_non_iid_unbalanced_20_27_50_s12347_e5_k3_b16", "FA E=5"),	# Higher dirichlet
			(r"Code_dp\results\real_non_iid_redo\ex_g_non_iid_unbalanced_20_29_29_s12347_e15_k3_b16", "FA E=15"),	# Higher dirichlet

			## B global IID balanced
			# (r"Code\results\real\no smpc\ex_g_iid_balanced_21_01_37_s12346_e5_k3_b16", "FA B=16"),
			# (r"Code\results\real\no smpc\ex_g_iid_balanced_00_13_54_s12346_e5_k3_b64", "FA B=64"),
			## B global IID unbalanced
			# (r"Code\results\real\no smpc\ex_g_iid_unbalanced_21_04_26_s12346_e5_k3_b16", "FA B=16"),
			# (r"Code\results\real\no smpc\ex_g_iid_unbalanced_00_14_46_s12346_e5_k3_b64", "FA B=64"),
			## B global non-IID balanced
			# (r"Code\results\real\no smpc\ex_g_non_iid_balanced_21_07_11_s12346_e5_k3_b16", "FA B=16"),
			# (r"Code\results\real\no smpc\ex_g_non_iid_balanced_00_15_35_s12346_e5_k3_b64", "FA B=64"),
			# (r"Code_dp\results\real_non_iid_redo\ex_g_non_iid_balanced_20_27_31_s12347_e5_k3_b16", "FA B=16"),		# Higher dirichlet
			# (r"Code_dp\results\real_non_iid_redo\ex_g_non_iid_balanced_20_31_48_s12347_e5_k3_b64", "FA B=64"),		# Higher dirichlet
			## B global non-IID unbalanced
			# (r"Code\results\real\no smpc\ex_g_non_iid_unbalanced_21_09_59_s12346_e5_k3_b16", "FA B=16"),
			# (r"Code\results\real\no smpc\ex_g_non_iid_unbalanced_00_16_26_s12346_e5_k3_b64", "FA B=64"),
			# (r"Code_dp\results\real_non_iid_redo\ex_g_non_iid_unbalanced_20_27_50_s12347_e5_k3_b16", "FA B=16"),	# Higher dirichlet
			# (r"Code_dp\results\real_non_iid_redo\ex_g_non_iid_unbalanced_20_31_58_s12347_e5_k3_b64", "FA B=64"),	# Higher dirichlet

		], [key], linestyle="--", std=False, col_per_dir=True)
		legends.extend(l)
		plots_dirs.extend(plots_dir)

		l, plots_dir, legends_colors = plot_multiple([
			## Real
			## E local IID balanced
			# (r"Code\results\real\no smpc\ex_l_iid_balanced_21_12_48_s12346_e5_k3_b16", "Local E=5"),
			# (r"Code\results\real\no smpc\ex_l_iid_balanced_22_40_48_s12346_e15_k3_b16", "Local E=15"),
			## E local IID unbalanced
			# (r"Code\results\real\no smpc\ex_l_iid_unbalanced_21_15_36_s12346_e5_k3_b16", "Local E=5"),
			# (r"Code\results\real\no smpc\ex_l_iid_unbalanced_22_49_05_s12346_e15_k3_b16", "Local E=15"),
			## E local non-IID balanced
			# (r"Code\results\real\no smpc\ex_l_non_iid_balanced_21_18_23_s12346_e5_k3_b16", "Local E=5"),
			# (r"Code\results\real\no smpc\ex_l_non_iid_balanced_22_57_11_s12346_e15_k3_b16", "Local E=15"),
			# (r"Code_dp\results\real_non_iid_redo\ex_l_non_iid_balanced_20_28_08_s12347_e5_k3_b16", "Local E=5"),	# Higher dirichlet
			# (r"Code_dp\results\real_non_iid_redo\ex_l_non_iid_balanced_20_30_15_s12347_e15_k3_b16", "Local E=15"),	# Higher dirichlet
			## E local non-IID unbalanced
			# (r"Code\results\real\no smpc\ex_l_non_iid_unbalanced_21_21_11_s12346_e5_k3_b16", "Local E=5"),
			# (r"Code\results\real\no smpc\ex_l_non_iid_unbalanced_23_05_20_s12346_e15_k3_b16", "Local E=15"),
			(r"Code_dp\results\real_non_iid_redo\ex_l_non_iid_unbalanced_20_28_26_s12347_e5_k3_b16", "Local E=5"),	# Higher dirichlet
			(r"Code_dp\results\real_non_iid_redo\ex_l_non_iid_unbalanced_20_31_00_s12347_e15_k3_b16", "Local E=15"),	# Higher dirichlet

			## B local IID balanced
			# (r"Code\results\real\no smpc\ex_l_iid_balanced_21_12_48_s12346_e5_k3_b16", "Local B=16"),
			# (r"Code\results\real\no smpc\ex_l_iid_balanced_00_17_14_s12346_e5_k3_b64", "Local B=64"),
			## B local IID unbalanced
			# (r"Code\results\real\no smpc\ex_l_iid_unbalanced_21_15_36_s12346_e5_k3_b16", "Local B=16"),
			# (r"Code\results\real\no smpc\ex_l_iid_unbalanced_00_18_05_s12346_e5_k3_b64", "Local B=64"),
			## B local non-IID balanced
			# (r"Code\results\real\no smpc\ex_l_non_iid_balanced_21_18_23_s12346_e5_k3_b16", "Local B=16"),
			# (r"Code\results\real\no smpc\ex_l_non_iid_balanced_00_18_53_s12346_e5_k3_b64", "Local B=64"),
			# (r"Code_dp\results\real_non_iid_redo\ex_l_non_iid_balanced_20_28_08_s12347_e5_k3_b16", "Local B=16"),	# Higher dirichlet
			# (r"Code_dp\results\real_non_iid_redo\ex_l_non_iid_balanced_20_32_07_s12347_e5_k3_b64", "Local B=64"),	# Higher dirichlet
			## B local non-IID unbalanced
			# (r"Code\results\real\no smpc\ex_l_non_iid_unbalanced_21_21_11_s12346_e5_k3_b16", "Local B=16"),
			# (r"Code\results\real\no smpc\ex_l_non_iid_unbalanced_00_19_44_s12346_e5_k3_b64", "Local B=64"),
			# (r"Code_dp\results\real_non_iid_redo\ex_l_non_iid_unbalanced_20_28_26_s12347_e5_k3_b16", "Local B=16"),	# Higher dirichlet
			# (r"Code_dp\results\real_non_iid_redo\ex_l_non_iid_unbalanced_20_32_17_s12347_e5_k3_b64", "Local B=64"),	# Higher dirichlet

		], [key], std=False, col_per_dir=True)
		legends.extend(l)
		plots_dirs.extend(plots_dir)

		lines = [plots_dirs[i][0][0] for i, e in enumerate(plots_dirs)]
		plt.legend(lines, legends, loc="best")
		# plt.legend(lines, legends, loc="lower center")

		plt.grid(True, which='both')
		plt.title("Local Models vs. FA")
		label(Label.COMM_ROUNDS, y_label)
		# save("real_l_iid_balanced_individual_" + key.name.lower(), clean=False)
		# save("real_l_iid_unbalanced_individual_" + key.name.lower(), clean=False)
		# save("real_l_non_iid_balanced_individual_" + key.name.lower(), clean=False)
		# save("real_l_non_iid_unbalanced_individual_" + key.name.lower(), clean=False)

		# save("real_l_iid_balanced_individual_" + key.name.lower() + "_batch", clean=False)
		# save("real_l_iid_unbalanced_individual_" + key.name.lower() + "_batch", clean=False)
		# save("real_l_non_iid_balanced_individual_" + key.name.lower() + "_batch", clean=False)
		# save("real_l_non_iid_unbalanced_individual_" + key.name.lower() + "_batch", clean=False)
		plt.show()

def plot_local_mnist():
	for key, y_label in loop_values:
		# FedAvg, FedAvg with DP, FedAvg with SMPC?
		fig, ax = plt.subplots()
		legends = []
		plots_dirs = []
		l, plots_dir, legends_colors = plot_multiple([
			## Plot central B=16
			# (r"Code\results\mnist\central\ex_central_16_34_38_s12346_e50_k0_b16", "Central B=16"),

			# (r"Code_dp\results\central\ex_central_iid_balanced_15_32_45_s12346_e50_k3_b16", "Central B=16"),
			# (r"Code_dp\results\central\ex_central_iid_unbalanced_15_33_06_s12346_e50_k3_b16", "Central B=16"),
			# (r"Code_dp\results\central\ex_central_non_iid_balanced_15_33_27_s12346_e50_k3_b16", "Central B=16"),
			(r"Code_dp\results\central\ex_central_non_iid_unbalanced_15_33_46_s12346_e50_k3_b16", "Central B=16"),
		], [key], marker="", std=False, col_per_dir=True, linewidth=2)
		legends.extend(l)
		plots_dirs.extend(plots_dir)

		l, plots_dir, legends_colors = plot_multiple([
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
			(r"Code\results\mnist\no smpc\ex_g_non_iid_unbalanced_02_13_14_s12346_e5_k3_b16", "FA E=5"),
			(r"Code\results\mnist\no smpc\ex_g_non_iid_unbalanced_07_40_00_s12346_e15_k3_b16", "FA E=15"),

			## B global IID balanced
			# (r"Code\results\mnist\no smpc\ex_g_iid_balanced_01_43_22_s12346_e5_k3_b16", "FA B=16"),
			# (r"Code\results\mnist\no smpc\ex_g_iid_balanced_15_39_37_s12346_e5_k3_b64", "FA B=64"),
			## B global IID unbalanced
			# (r"Code\results\mnist\no smpc\ex_g_iid_unbalanced_01_53_27_s12346_e5_k3_b16", "FA B=16"),
			# (r"Code\results\mnist\no smpc\ex_g_iid_unbalanced_15_42_28_s12346_e5_k3_b64", "FA B=64"),
			## B global non-IID balanced
			# (r"Code\results\mnist\no smpc\ex_g_non_iid_balanced_02_03_17_s12346_e5_k3_b16", "FA B=16"),
			# (r"Code\results\mnist\no smpc\ex_g_non_iid_balanced_15_45_12_s12346_e5_k3_b64", "FA B=64"),
			## B global non-IID unbalanced
			# (r"Code\results\mnist\no smpc\ex_g_non_iid_unbalanced_02_13_14_s12346_e5_k3_b16", "FA B=16"),
			# (r"Code\results\mnist\no smpc\ex_g_non_iid_unbalanced_15_48_01_s12346_e5_k3_b64", "FA B=64"),

		], [key], linestyle="--", std=False, col_per_dir=True)
		legends.extend(l)
		plots_dirs.extend(plots_dir)

		l, plots_dir, legends_colors = plot_multiple([
			## MNIST
			## E local IID balanced
			# (r"Code\results\mnist\no smpc\ex_l_iid_balanced_02_23_05_s12346_e5_k3_b16", "Local E=5"),
			# (r"Code\results\mnist\no smpc\ex_l_iid_balanced_08_14_36_s12346_e15_k3_b16", "Local E=15"),
			## E local IID unbalanced
			# (r"Code\results\mnist\no smpc\ex_l_iid_unbalanced_02_32_58_s12346_e5_k3_b16", "Local E=5"),
			# (r"Code\results\mnist\no smpc\ex_l_iid_unbalanced_08_48_25_s12346_e15_k3_b16", "Local E=15"),
			## E local non-IID balanced
			# (r"Code\results\mnist\no smpc\ex_l_non_iid_balanced_02_42_32_s12346_e5_k3_b16", "Local E=5"),
			# (r"Code\results\mnist\no smpc\ex_l_non_iid_balanced_09_21_50_s12346_e15_k3_b16", "Local E=15"),
			## E local non-IID unbalanced
			(r"Code\results\mnist\no smpc\ex_l_non_iid_unbalanced_02_52_22_s12346_e5_k3_b16", "Local E=5"),
			(r"Code\results\mnist\no smpc\ex_l_non_iid_unbalanced_09_55_49_s12346_e15_k3_b16", "Local E=15"),

			## B local IID balanced
			# (r"Code\results\mnist\no smpc\ex_l_iid_balanced_02_23_05_s12346_e5_k3_b16", "Local B=16"),
			# (r"Code\results\mnist\no smpc\ex_l_iid_balanced_15_50_46_s12346_e5_k3_b64", "Local B=64"),
			## B local IID unbalanced
			# (r"Code\results\mnist\no smpc\ex_l_iid_unbalanced_02_32_58_s12346_e5_k3_b16", "Local B=16"),
			# (r"Code\results\mnist\no smpc\ex_l_iid_unbalanced_15_53_37_s12346_e5_k3_b64", "Local B=64"),
			## B local non-IID balanced
			# (r"Code\results\mnist\no smpc\ex_l_non_iid_balanced_02_42_32_s12346_e5_k3_b16", "Local B=16"),
			# (r"Code\results\mnist\no smpc\ex_l_non_iid_balanced_15_56_22_s12346_e5_k3_b64", "Local B=64"),
			## B local non-IID unbalanced
			# (r"Code\results\mnist\no smpc\ex_l_non_iid_unbalanced_02_52_22_s12346_e5_k3_b16", "Local B=16"),
			# (r"Code\results\mnist\no smpc\ex_l_non_iid_unbalanced_15_59_13_s12346_e5_k3_b64", "Local B=64"),

		], [key], std=False, col_per_dir=True)
		legends.extend(l)
		plots_dirs.extend(plots_dir)

		lines = [plots_dirs[i][0][0] for i, e in enumerate(plots_dirs)]
		plt.legend(lines, legends, loc="best")

		plt.grid(True, which='both')
		plt.title("Local Models vs. FA")
		label(Label.COMM_ROUNDS, y_label)
		# save("mnist_l_iid_balanced_individual_" + key.name.lower(), clean=False)
		# save("mnist_l_iid_unbalanced_individual_" + key.name.lower(), clean=False)
		# save("mnist_l_non_iid_balanced_individual_" + key.name.lower(), clean=False)
		save("mnist_l_non_iid_unbalanced_individual_" + key.name.lower(), clean=False)

		# save("mnist_l_iid_balanced_individual_" + key.name.lower() + "_batch", clean=False)
		# save("mnist_l_iid_unbalanced_individual_" + key.name.lower() + "_batch", clean=False)
		# save("mnist_l_non_iid_balanced_individual_" + key.name.lower() + "_batch", clean=False)
		# save("mnist_l_non_iid_unbalanced_individual_" + key.name.lower() + "_batch", clean=False)
		plt.show()

def plot_global():
	kl = [
		(Key.TRAIN_ACC, Label.TRAIN_ACC),
		(Key.TRAIN_LOSS, Label.TRAIN_LOSS),
		(Key.TRAIN_PRECISION, Label.TRAIN_PRECISION),
		(Key.TRAIN_RECALL, Label.TRAIN_RECALL),
		(Key.TRAIN_AUC, Label.TRAIN_AUC),

		(Key.TEST_ACC, Label.TEST_ACC),
		(Key.TEST_LOSS, Label.TEST_LOSS),
		(Key.TEST_PRECISION, Label.TEST_PRECISION),
		(Key.TEST_RECALL, Label.TEST_RECALL),
		(Key.TEST_AUC, Label.TEST_AUC),
	]

	for key, y_label in kl:
		# FedAvg, FedAvg with DP, FedAvg with SMPC?
		fig, ax = plt.subplots()
		legends = []
		plots_dirs = []
		l, plots_dir, legends_colors = plot_multiple([
			## Plot central B=16
			# (r"Code\results\mnist\central\ex_central_16_34_38_s12346_e50_k0_b16", "Central B=16"),
			(r"Code\results\real\central\ex_central_15_50_57_s12346_e50_k0_b16", "Central B=16"),
		], [key], marker="x", std=False, col_per_dir=False)
		legends.extend(l)
		plots_dirs.extend(plots_dir)

		l, plots_dir, legends_colors = plot_multiple([
			## E global IID balanced
			# (r"Code\results\mnist\no smpc\ex_g_iid_balanced_01_43_22_s12346_e5_k3_b16", "E=5"),
			# (r"Code\results\mnist\no smpc\ex_g_iid_balanced_05_57_06_s12346_e15_k3_b16", "E=15"),
			## E global IID unbalanced
			# (r"Code\results\mnist\no smpc\ex_g_iid_unbalanced_01_53_27_s12346_e5_k3_b16", "E=5"),
			# (r"Code\results\mnist\no smpc\ex_g_iid_unbalanced_06_31_47_s12346_e15_k3_b16", "E=15"),
			## E global non-IID balanced
			# (r"Code\results\mnist\no smpc\ex_g_non_iid_balanced_02_03_17_s12346_e5_k3_b16", "E=5"),
			# (r"Code\results\mnist\no smpc\ex_g_non_iid_balanced_07_05_08_s12346_e15_k3_b16", "E=15"),
			## E global non-IID unbalanced
			# (r"Code\results\mnist\no smpc\ex_g_non_iid_unbalanced_02_13_14_s12346_e5_k3_b16", "E=5"),
			# (r"Code\results\mnist\no smpc\ex_g_non_iid_unbalanced_07_40_00_s12346_e15_k3_b16", "E=15"),

			## B global IID balanced
			# (r"Code\results\mnist\no smpc\ex_g_iid_balanced_01_43_22_s12346_e5_k3_b16", "B=16"),
			# (r"Code\results\mnist\no smpc\ex_g_iid_balanced_15_39_37_s12346_e5_k3_b64", "B=64"),
			## B global IID unbalanced
			# (r"Code\results\mnist\no smpc\ex_g_iid_unbalanced_01_53_27_s12346_e5_k3_b16", "B=16"),
			# (r"Code\results\mnist\no smpc\ex_g_iid_unbalanced_15_42_28_s12346_e5_k3_b64", "B=64"),
			## B global non-IID balanced
			# (r"Code\results\mnist\no smpc\ex_g_non_iid_balanced_02_03_17_s12346_e5_k3_b16", "B=16"),
			# (r"Code\results\mnist\no smpc\ex_g_non_iid_balanced_15_45_12_s12346_e5_k3_b64", "B=64"),
			## B global non-IID unbalanced
			# (r"Code\results\mnist\no smpc\ex_g_non_iid_unbalanced_02_13_14_s12346_e5_k3_b16", "B=16"),
			# (r"Code\results\mnist\no smpc\ex_g_non_iid_unbalanced_15_48_01_s12346_e5_k3_b64", "B=64"),
			
			## E global IID balanced
			# (r"Code\results\real\no smpc\ex_g_iid_balanced_21_01_37_s12346_e5_k3_b16", "E=5"),
			# (r"Code\results\real\no smpc\ex_g_iid_balanced_22_07_47_s12346_e15_k3_b16", "E=15"),
			## E global IID unbalanced
			# (r"Code\results\real\no smpc\ex_g_iid_unbalanced_21_04_26_s12346_e5_k3_b16", "E=5"),
			# (r"Code\results\real\no smpc\ex_g_iid_unbalanced_22_16_05_s12346_e15_k3_b16", "E=15"),
			## E global non-IID balanced
			# (r"Code\results\real\no smpc\ex_g_non_iid_balanced_21_07_11_s12346_e5_k3_b16", "E=5"),
			# (r"Code\results\real\no smpc\ex_g_non_iid_balanced_22_24_15_s12346_e15_k3_b16", "E=15"),
			## E global non-IID unbalanced
			# (r"Code\results\real\no smpc\ex_g_non_iid_unbalanced_21_09_59_s12346_e5_k3_b16", "E=5"),
			# (r"Code\results\real\no smpc\ex_g_non_iid_unbalanced_22_32_27_s12346_e15_k3_b16", "E=15"),

			## B global IID balanced
			# (r"Code\results\real\no smpc\ex_g_iid_balanced_21_01_37_s12346_e5_k3_b16", "B=16"),
			# (r"Code\results\real\no smpc\ex_g_iid_balanced_00_13_54_s12346_e5_k3_b64", "B=64"),
			## B global IID unbalanced
			# (r"Code\results\real\no smpc\ex_g_iid_unbalanced_21_04_26_s12346_e5_k3_b16", "B=16"),
			# (r"Code\results\real\no smpc\ex_g_iid_unbalanced_00_14_46_s12346_e5_k3_b64", "B=64"),
			## B global non-IID balanced
			# (r"Code\results\real\no smpc\ex_g_non_iid_balanced_21_07_11_s12346_e5_k3_b16", "B=16"),
			# (r"Code\results\real\no smpc\ex_g_non_iid_balanced_00_15_35_s12346_e5_k3_b64", "B=64"),
			## B global non-IID unbalanced
			# (r"Code\results\real\no smpc\ex_g_non_iid_unbalanced_21_09_59_s12346_e5_k3_b16", "B=16"),
			# (r"Code\results\real\no smpc\ex_g_non_iid_unbalanced_00_16_26_s12346_e5_k3_b64", "B=64"),

			##### TEST
			## E global IID balanced
			# (r"Code\results\real\no smpc reduced data\ex_g_iid_balanced_02_36_44_s12346_e5_k3_b16", "E=5"),
			# (r"Code\results\real\no smpc reduced data\ex_g_iid_balanced_03_48_03_s12346_e15_k3_b16", "E=15"),
			## E global IID unbalanced
			# (r"Code\results\real\no smpc reduced data\ex_g_iid_unbalanced_02_39_47_s12346_e5_k3_b16", "E=5"),
			# (r"Code\results\real\no smpc reduced data\ex_g_iid_unbalanced_03_56_49_s12346_e15_k3_b16", "E=15"),
			## E global non-IID balanced
			# (r"Code\results\real\no smpc reduced data\ex_g_non_iid_balanced_02_42_49_s12346_e5_k3_b16", "E=5"),
			# (r"Code\results\real\no smpc reduced data\ex_g_non_iid_balanced_04_05_39_s12346_e15_k3_b16", "E=15"),
			## E global non-IID unbalanced
			# (r"Code\results\real\no smpc reduced data\ex_g_non_iid_unbalanced_02_45_51_s12346_e5_k3_b16", "E=5"),
			# (r"Code\results\real\no smpc reduced data\ex_g_non_iid_unbalanced_04_14_27_s12346_e15_k3_b16", "E=15"),

			## B global IID balanced
			# (r"Code\results\real\no smpc reduced data\ex_g_iid_balanced_02_36_44_s12346_e5_k3_b16", "B=16"),
			# (r"Code\results\real\no smpc reduced data\ex_g_iid_balanced_06_19_07_s12346_e5_k3_b64", "B=64"),
			## B global IID unbalanced
			# (r"Code\results\real\no smpc reduced data\ex_g_iid_unbalanced_02_39_47_s12346_e5_k3_b16", "B=16"),
			# (r"Code\results\real\no smpc reduced data\ex_g_iid_unbalanced_06_20_01_s12346_e5_k3_b64", "B=64"),
			## B global non-IID balanced
			# (r"Code\results\real\no smpc reduced data\ex_g_non_iid_balanced_02_42_49_s12346_e5_k3_b16", "B=16"),
			# (r"Code\results\real\no smpc reduced data\ex_g_non_iid_balanced_06_20_55_s12346_e5_k3_b64", "B=64"),
			## B global non-IID unbalanced
			# (r"Code\results\real\no smpc reduced data\ex_g_non_iid_unbalanced_02_45_51_s12346_e5_k3_b16", "B=16"),
			# (r"Code\results\real\no smpc reduced data\ex_g_non_iid_unbalanced_06_21_49_s12346_e5_k3_b64", "B=64"),
		], [key], std=False, col_per_dir=True)
		legends.extend(l)
		plots_dirs.extend(plots_dir)

		lines = [plots_dirs[i][0][0] for i, e in enumerate(plots_dirs)]
		plt.legend(lines, legends, loc="best")

		plt.grid(True, which='both')
		# plt.title("Federated Averaging")
		label(Label.COMM_ROUNDS, y_label)
		# dataset = "mnist"
		dataset = "real"
		# save(dataset + "_g_iid_balanced_individual_" + key.name.lower(), clean=True)
		# save(dataset + "_g_iid_unbalanced_individual_" + key.name.lower(), clean=True)
		# save(dataset + "_g_non_iid_balanced_individual_" + key.name.lower(), clean=False)
		# save(dataset + "_g_non_iid_unbalanced_individual_" + key.name.lower(), clean=False)

		# save(dataset + "_g_iid_balanced_individual_" + key.name.lower() + "_batch", clean=True)
		# save(dataset + "_g_iid_unbalanced_individual_" + key.name.lower() + "_batch", clean=True)
		# save(dataset + "_g_non_iid_balanced_individual_" + key.name.lower() + "_batch", clean=False)
		# save(dataset + "_g_non_iid_unbalanced_individual_" + key.name.lower() + "_batch", clean=False)
		plt.show()

def plot_smpc():
	kl = [
		# (Key.TRAIN_ACC, Label.TRAIN_ACC),
		# (Key.TRAIN_LOSS, Label.TRAIN_LOSS),
		# (Key.TEST_ACC, Label.TEST_ACC),
		(Key.TEST_LOSS, Label.TEST_ACC)
	]

	for key, y_label in kl:
		fig, ax = plt.subplots()
		legends = []
		# l, _, _ = plot_multiple([
		# 	## Plot central B=16
		# 	# (r"Code\results\mnist\central\ex_central_16_34_38_s12346_e50_k0_b16", "Central B=16"),

		# 	(r"Code_dp\results\central\ex_central_iid_balanced_15_32_45_s12346_e50_k3_b16", "Central B=16"),
		# ], [key], marker="", std=False, col_per_dir=True, linewidth=2)
		# legends.extend(l)

		l, _, _ = plot_multiple([
			# Vary E
			(r"Code\results\mnist\no smpc\ex_g_iid_balanced_01_43_22_s12346_e5_k3_b16", "FA E=5"),
			(r"Code\results\mnist\smpc\ex_g_iid_balanced_smpc_19_05_47_s12346_e5_k3_b16", "FA SMPC E=5"),
			(r"Code\results\mnist\no smpc\ex_g_iid_balanced_05_57_06_s12346_e15_k3_b16", "FA E=15"),
			(r"Code\results\mnist\smpc\ex_g_iid_balanced_smpc_19_17_45_s12346_e15_k3_b16", "FA SMPC E=15"),

			# Vary B
			# (r"Code\results\mnist\no smpc\ex_g_iid_balanced_01_43_22_s12346_e5_k3_b16", "FA B=16"),
			# (r"Code\results\mnist\smpc\ex_g_iid_balanced_smpc_19_05_47_s12346_e5_k3_b16", "FA SMPC B=16"),
			# (r"Code\results\mnist\no smpc\ex_g_iid_balanced_15_39_37_s12346_e5_k3_b64", "FA B=64"),
			# (r"Code\results\mnist\smpc\ex_g_iid_balanced_smpc_19_53_30_s12346_e5_k3_b64", "FA SMPC B=64"),
		], [key], fmt="-", std=True, col_per_dir=False)
		legends.extend(l)

		plt.legend(legends, loc="best")
		plt.grid(True, which='both')
		# plt.title("Federated Averaging")
		label(Label.COMM_ROUNDS, y_label)

		## Train and Test (Epoch)
		# ax.set_xlim(0, 17.5)
		# ax.set_ylim(0.85, 1.02)
		# save("mnist_g_smpc_iid_balanced_train_accs", clean=False)
		# save("mnist_g_smpc_iid_balanced_train_loss", clean=False)

		# ax.set_xlim(0, 17.5)
		# ax.set_ylim(0.75, 0.92)
		# save("mnist_g_smpc_iid_balanced_test_accs", clean=False) #Remember fmt="-o"
		save("mnist_g_smpc_iid_balanced_test_loss", clean=False)
		
		## Train and Test (Batch)
		# ax.set_xlim(0, 20.5)
		# ax.set_ylim(0.7, 1.0)
		# save("mnist_g_smpc_iid_balanced_train_accs_batch", clean=False)

		# ax.set_xlim(0, 20.5)
		# ax.set_ylim(0.6, 0.90)
		# save("mnist_g_smpc_iid_balanced_test_accs_batch", clean=False)


		# # axins = zoomed_inset_axes(ax, 2, loc=1)
		# # axins.set_xlabel("")
		# # axins.set_ylabel("")
		# # axins.set_xlim(2, 9)
		# # axins.set_ylim(0.92, 0.98)
		# # plt.xticks(visible=False)
		# # plt.yticks(visible=False)
		# # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
		plt.show()

def plot_central():
	## Plot central
	kl = [
		(Key.TRAIN_ACC, Label.TRAIN_ACC),
		(Key.TRAIN_LOSS, Label.TRAIN_LOSS),
		(Key.TRAIN_PRECISION, Label.TRAIN_PRECISION),
		(Key.TRAIN_RECALL, Label.TRAIN_RECALL),
		(Key.TRAIN_AUC, Label.TRAIN_AUC),

		(Key.TEST_ACC, Label.TEST_ACC),
		(Key.TEST_LOSS, Label.TEST_LOSS),
		(Key.TEST_PRECISION, Label.TEST_PRECISION),
		(Key.TEST_RECALL, Label.TEST_RECALL),
		(Key.TEST_AUC, Label.TEST_AUC),
	]
	for key, y_label in kl:
		fig, ax = plt.subplots()
		legends = []
		plots_dirs = []
		l, plots_dir, legends_colors = plot_multiple([
			# (r"Code\results\mnist\central\ex_central_16_34_38_s12346_e50_k0_b16", "B = 16"),
			# (r"Code\results\mnist\central\ex_central_16_34_59_s12346_e50_k0_b32", "B = 32"),
			# (r"Code\results\mnist\central\ex_central_16_35_13_s12346_e50_k0_b64", "B = 64"),

			# (r"Code\results\real\central\ex_central_15_50_57_s12346_e50_k0_b16", "B = 16"),
			# (r"Code\results\real\central\ex_central_15_51_06_s12346_e50_k0_b32", "B = 32"),
			# (r"Code\results\real\central\ex_central_15_51_12_s12346_e50_k0_b64", "B = 64"),

			(r"Code\results\ex_central_15_37_35_s12346_e50_k0_b16", "B = 16"),
			(r"Code\results\ex_central_15_37_43_s12346_e50_k0_b32", "B = 32"),
			(r"Code\results\ex_central_15_37_48_s12346_e50_k0_b64", "B = 64"),
		], [key], fmt="-", loc="best", col_per_dir=False)
		label(Label.EPOCHS, y_label)

		# plt.ylim(0, 1)
		plt.legend(l)
		plt.title("Central")
		plt.grid(True, which='both')
		# save("mnist_central_" + key.name.lower(), clean=False)
		# save("completes_central_" + key.name.lower(), clean=True)
		plt.show()

def plot_confusion_matrix():
	## MNIST
	# layer_sizes = [784, 128, 128, 10]
	# state_file = r"Code\results\mnist\central\ex_central_16_34_38_s12346_e50_k0_b16\global_e50_s12346.pt"
	# targets, label_predictions, label_estimates, test_loss = get_model_predictions(layer_sizes, state_file, load_data_mnist)
	# fig, ax = plot_cm(targets, label_predictions, invertThreshold=True)

	## Real
	layer_sizes = [27, 64, 2]
	state_file = r"Code\results\ex_central_19_23_40_s12346_e50_k0_b16\global_e50_s12346.pt"
	targets, label_predictions, label_estimates, test_loss = get_model_predictions(layer_sizes, state_file, load_data_completes)
	fig, ax = plot_cm(targets, label_predictions, figsize=(2,2), invertThreshold=True, no_norm=True)

	plt.title('')
	plt.xlabel('')
	plt.ylabel('')
	# plt.savefig("mnist_central_cm.png")
	plt.savefig("real_central_cm.png")
	plt.show()

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
	# data = np.array([
	# 	# nonIID balanced test new logic
	# 	[123, 81, 60, 29, 16, 8, 3, 0, 0, 0],
	# 	[32, 60, 63, 73, 48, 24, 11, 6, 2, 1],
	# 	[2, 14, 29, 53, 66, 64, 45, 22, 19, 6],
	# 	[0, 2, 6, 9, 26, 49, 73, 65, 57, 33],
	# 	[0, 0, 0, 1, 6, 14, 30, 67, 81, 121]
	# ])
	data = np.array([
		# [34, 17, 63, 54, 24, 1, 12, 15, 100, 0],
		# [3, 12, 14, 6, 79, 24, 8, 13, 21, 140],
		# [30, 118, 34, 22, 8, 2, 25, 43, 23, 15],
		# [75, 9, 0, 27, 18, 122, 4, 65, 0, 0],
		# [12, 4, 57, 51, 31, 9, 110, 29, 7, 10]

		# new IID unbalanced
		[54, 53, 55, 50, 52, 56, 58, 52, 65, 50],
		[76, 79, 71, 84, 79, 73, 75, 92, 73, 78],
		[12, 7, 15, 7, 9, 15, 15, 6, 14, 10],
		[11, 6, 8, 6, 8, 10, 10, 7, 6, 9],
		[11, 8, 10, 8, 6, 10, 6, 9, 7, 7]
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
	# plt.savefig("mnist_data_distribution_iid_unbalanced_hist.png")
	plt.savefig("mnist_data_distribution_noniid_balanced_hist.png")
	# plt.savefig("mnist_data_distribution_noniid_unbalanced_hist.png")
	plt.show()

############################################

def plot_sample():
	train_data, test_data, X, y, X_test, y_test, all_data = load_data_mnist()
	dataloader = get_data_loader(train_data, 512, {}, shuffle=False)

	samples, labels = next(iter(dataloader))

	images = []
	for i in range(10):
		images.append([samples[j] for j in range(len(labels)) if labels[j] == i][0].reshape(28, 28))
	
	fig, ax = show_images(images)
	plt.savefig("mnist_samples")
	plt.show()

def show_images(images: List[np.ndarray]) -> None:
	n: int = len(images)
	f, ax = plt.subplots(figsize=(10,5))
	ax.set_axis_off()
	for i in range(n):
		f.add_subplot(2, n/2, i + 1)
		plt.axis('off')
		# ax.tick_params(left=False, bottom=False)
		# fig.patch.set_visible(False)
		plt.imshow(images[i], cmap="gray")
	return f, ax


if __name__ == "__main__":
	main()