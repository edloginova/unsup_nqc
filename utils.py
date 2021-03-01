import matplotlib.pyplot as plt
import numpy as np

from utils_constants import *


def check_calibration(df, bin_size, output_file=None):
    for idx in np.arange(0, 1.0 / bin_size):
        score_range = (bin_size * idx, bin_size * (idx + 1))
        tmp_df = df[(df[MAX_SCORE] >= score_range[0]) & (df[MAX_SCORE] < score_range[1])]
        len_tmp_df = len(tmp_df)
        if len_tmp_df > 0:
            output_string = "correctness for [%.2f; %.2f[ = %.2f | N. samples = %6d" \
                            % (score_range[0], score_range[1], float(np.mean(tmp_df[CORRECTNESS])), len_tmp_df)
            if output_file is None:
                print(output_string)
            else:
                output_file.write(output_string + "\n")


def plot_calibration(df, bin_size, image_name=None):
    list_correctness = []
    list_counts = []
    list_score_ranges = []
    for idx in np.arange(0, 1.0 / bin_size):
        score_range = (bin_size * idx, bin_size * (idx + 1))
        list_score_ranges.append(score_range)
        tmp_df = df[(df[MAX_SCORE] >= score_range[0]) & (df[MAX_SCORE] < score_range[1])]
        len_tmp_df = len(tmp_df)
        list_counts.append(len_tmp_df)
        if len_tmp_df > 0:
            list_correctness.append(np.mean(tmp_df[CORRECTNESS]))
        else:
            list_correctness.append(0)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    num_bins = len(list_correctness)
    ax[0].bar(range(len(list_correctness)), list_correctness)
    ax[0].set_title('Accuracy per bin')
    ax[0].plot([0, num_bins - 1], [0.0 + bin_size / 2, 1.0 - bin_size / 2], c='r')
    ax[0].set_ylabel('Real Accuracy')

    ax[1].bar(range(len(list_correctness)), list_counts)
    ax[1].set_title('Number of items per bin')
    ax[1].set_ylabel('Number of items')

    for idx in [0, 1]:
        ax[idx].set_xticks(range(len(list_correctness)))
        ax[idx].set_xticklabels(["[%.2f; %.2f[" % (x[0], x[1]) for x in list_score_ranges], rotation=75)
        ax[idx].set_xlabel('Confidence (range)')
    if image_name is None:
        plt.show()
    else:
        plt.savefig(image_name)


def evaluate_level_prediction_high_vs_middle(df, output_file=None):
    if output_file is None:
        # overall
        print("ACC: max_score =       %.4f" % accuracy_with_max_score_hm(df))
        print("ACC: max_others_diff   %.4f" % accuracy_with_max_others_diff_hm(df))
        print("ACC: max_2nd_diff      %.4f" % accuracy_with_max_2nd_diff_hm(df))
        print("ACC: scores_var        %.4f" % accuracy_with_scores_var_hm(df))
        print("ACC: std majority v1 = %.4f" % accuracy_with_std_majority_v1_hm(df))
        print("ACC: std majority v2 = %.4f" % accuracy_with_std_majority_v2_hm(df))
        print("ACC: std majority v3 = %.4f" % accuracy_with_std_majority_v3_hm(df))
        print("ACC: std majority v4 = %.4f" % accuracy_with_std_majority_v4_hm(df))
        # correct predictions
        tmp_df = df[(df[CORRECTNESS + '_m']) & (df[CORRECTNESS + '_h'])]
        print("\nCorrect predictions")
        print("ACC: max_score =       %.4f" % accuracy_with_max_score_hm(tmp_df))
        print("ACC: max_others_diff   %.4f" % accuracy_with_max_others_diff_hm(tmp_df))
        print("ACC: max_2nd_diff      %.4f" % accuracy_with_max_2nd_diff_hm(tmp_df))
        print("ACC: scores_var        %.4f" % accuracy_with_scores_var_hm(tmp_df))
        print("ACC: std majority v1 = %.4f" % accuracy_with_std_majority_v1_hm(tmp_df))
        print("ACC: std majority v2 = %.4f" % accuracy_with_std_majority_v2_hm(tmp_df))
        print("ACC: std majority v3 = %.4f" % accuracy_with_std_majority_v3_hm(tmp_df))
        print("ACC: std majority v4 = %.4f" % accuracy_with_std_majority_v4_hm(tmp_df))
        # wrong predictions
        print("\nWrong predictions")
        tmp_df = df[(~df[CORRECTNESS + '_m']) & (~df[CORRECTNESS + '_h'])]
        print("ACC: max_score =       %.4f" % accuracy_with_max_score_hm(tmp_df))
        print("ACC: max_others_diff   %.4f" % accuracy_with_max_others_diff_hm(tmp_df))
        print("ACC: max_2nd_diff      %.4f" % accuracy_with_max_2nd_diff_hm(tmp_df))
        print("ACC: scores_var        %.4f" % accuracy_with_scores_var_hm(tmp_df))
        print("ACC: std majority v1 = %.4f" % accuracy_with_std_majority_v1_hm(tmp_df))
        print("ACC: std majority v2 = %.4f" % accuracy_with_std_majority_v2_hm(tmp_df))
        print("ACC: std majority v3 = %.4f" % accuracy_with_std_majority_v3_hm(tmp_df))
        print("ACC: std majority v4 = %.4f" % accuracy_with_std_majority_v4_hm(tmp_df))
    else:
        # overall
        output_file.write("ACC: max_score =       %.4f\n" % accuracy_with_max_score_hm(df))
        output_file.write("ACC: max_others_diff   %.4f\n" % accuracy_with_max_others_diff_hm(df))
        output_file.write("ACC: max_2nd_diff      %.4f\n" % accuracy_with_max_2nd_diff_hm(df))
        output_file.write("ACC: scores_var        %.4f\n" % accuracy_with_scores_var_hm(df))
        output_file.write("ACC: std majority v1 = %.4f\n" % accuracy_with_std_majority_v1_hm(df))
        output_file.write("ACC: std majority v2 = %.4f\n" % accuracy_with_std_majority_v2_hm(df))
        output_file.write("ACC: std majority v3 = %.4f\n" % accuracy_with_std_majority_v3_hm(df))
        output_file.write("ACC: std majority v4 = %.4f\n" % accuracy_with_std_majority_v4_hm(df))
        # correct predictions
        tmp_df = df[(df[CORRECTNESS + '_m']) & (df[CORRECTNESS + '_h'])]
        output_file.write("\nCorrect predictions\n")
        output_file.write("ACC: max_score =       %.4f\n" % accuracy_with_max_score_hm(tmp_df))
        output_file.write("ACC: max_others_diff   %.4f\n" % accuracy_with_max_others_diff_hm(tmp_df))
        output_file.write("ACC: max_2nd_diff      %.4f\n" % accuracy_with_max_2nd_diff_hm(tmp_df))
        output_file.write("ACC: scores_var        %.4f\n" % accuracy_with_scores_var_hm(tmp_df))
        output_file.write("ACC: std majority v1 = %.4f\n" % accuracy_with_std_majority_v1_hm(tmp_df))
        output_file.write("ACC: std majority v2 = %.4f\n" % accuracy_with_std_majority_v2_hm(tmp_df))
        output_file.write("ACC: std majority v3 = %.4f\n" % accuracy_with_std_majority_v3_hm(tmp_df))
        output_file.write("ACC: std majority v4 = %.4f\n" % accuracy_with_std_majority_v4_hm(tmp_df))
        # wrong predictions
        output_file.write("\nWrong predictions\n")
        tmp_df = df[(~df[CORRECTNESS + '_m']) & (~df[CORRECTNESS + '_h'])]
        output_file.write("ACC: max_score =       %.4f\n" % accuracy_with_max_score_hm(tmp_df))
        output_file.write("ACC: max_others_diff   %.4f\n" % accuracy_with_max_others_diff_hm(tmp_df))
        output_file.write("ACC: max_2nd_diff      %.4f\n" % accuracy_with_max_2nd_diff_hm(tmp_df))
        output_file.write("ACC: scores_var        %.4f\n" % accuracy_with_scores_var_hm(tmp_df))
        output_file.write("ACC: std majority v1 = %.4f\n" % accuracy_with_std_majority_v1_hm(tmp_df))
        output_file.write("ACC: std majority v2 = %.4f\n" % accuracy_with_std_majority_v2_hm(tmp_df))
        output_file.write("ACC: std majority v3 = %.4f\n" % accuracy_with_std_majority_v3_hm(tmp_df))
        output_file.write("ACC: std majority v4 = %.4f\n" % accuracy_with_std_majority_v4_hm(tmp_df))


def accuracy_with_max_score_hm(df):
    return np.mean(df.apply(lambda r: r[MAX_SCORE + '_h'] < r[MAX_SCORE + '_m'], axis=1))


def accuracy_with_max_others_diff_hm(df):
    return np.mean(df.apply(lambda r: r[MAX_OTH_DIFF + '_h'] < r[MAX_OTH_DIFF + '_m'], axis=1))


def accuracy_with_max_2nd_diff_hm(df):
    return np.mean(df.apply(lambda r: r[MAX_2ND_DIFF + '_h'] < r[MAX_2ND_DIFF + '_m'], axis=1))


def accuracy_with_scores_var_hm(df):
    return np.mean(df.apply(lambda r: r[SCORES_VAR + '_h'] < r[SCORES_VAR + '_m'], axis=1))


def accuracy_with_std_majority_v1_hm(df):
    return np.mean(df.apply(
        lambda r: r[STD_MAX_SCORE + '_h'] + r[STD_MAX_2ND_DIFF + '_h'] + r[STD_SCORES_VAR + '_h']
                  < r[STD_MAX_SCORE + '_m'] + r[STD_MAX_2ND_DIFF + '_m'] + r[STD_SCORES_VAR + '_m'],
        axis=1))


def accuracy_with_std_majority_v2_hm(df):
    return np.mean(df.apply(
        lambda r: r[STD_MAX_SCORE + '_h'] + r[STD_SCORES_VAR + '_h'] < r[STD_MAX_SCORE + '_m'] + r[
            STD_SCORES_VAR + '_m'],
        axis=1))


def accuracy_with_std_majority_v3_hm(df):
    return np.mean(df.apply(
        lambda r: r[STD_MAX_SCORE + '_h'] + r[STD_MAX_2ND_DIFF + '_h'] < r[STD_MAX_SCORE + '_m'] + r[
            STD_MAX_2ND_DIFF + '_m'],
        axis=1))


def accuracy_with_std_majority_v4_hm(df):
    return np.mean(df.apply(
        lambda r: r[STD_MAX_2ND_DIFF + '_h'] + r[STD_SCORES_VAR + '_h'] < r[STD_MAX_2ND_DIFF + '_m'] + r[
            STD_SCORES_VAR + '_m'],
        axis=1))
