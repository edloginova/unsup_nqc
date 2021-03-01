import matplotlib.pyplot as plt
import numpy as np

from utils_constants import MAX_SCORE, CORRECTNESS
from utils_data import create_calibrated_df

random_seed = 42
split = 'test'
# image_name = 'output_figures/rel_diagram.pdf'
image_name = None
bin_size = 0.1

df = create_calibrated_df(['output_bert_seed%d_%s.csv' % (random_seed, split)])

# single model
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
num_bins = len(list_correctness)

fig, ax = plt.subplots(2,1, figsize=(6,6), sharex=True)

ax[0].bar(range(len(list_correctness)), list_correctness, color='#00b176', label='single BERT instance')
ax[0].plot([0, 10], [0.0, 1.0], c='#d02f2d')
ax[0].set_ylabel('Real Accuracy')

ax[0].set_xticks(range(len(list_correctness)))
ax[0].set_xticklabels(["%.2f" % (x[0]+0.05) for x in list_score_ranges])
ax[0].set_ylim([0,1])
ax[0].set_xlim([0,9.5])
ax[0].legend()

# ensemble
random_seeds = [0, 3, 42]
bin_size = 0.1

df = create_calibrated_df(['output_bert_seed%d_%s.csv' % (random_seed, split) for random_seed in random_seeds])
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

num_bins = len(list_correctness)

ax[1].bar(range(len(list_correctness)), list_correctness, color='#ffd74d', label='BERT ensemble')
ax[1].plot([0, 10], [0.0, 1.0], c='#d02f2d')
ax[1].set_ylabel('Real Accuracy')

ax[1].set_xticks(range(len(list_correctness)))
ax[1].set_xticklabels(["%.2f" % (x[0]+0.05) for x in list_score_ranges])
ax[1].set_xlabel('Confidence (range)')
ax[1].set_ylim([0,1])
ax[1].set_xlim([0,9.5])
ax[1].legend()

if image_name is None:
    plt.show()
else:
    plt.savefig(image_name)
