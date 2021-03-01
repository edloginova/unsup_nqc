from netcal.metrics import ECE

from utils_constants import CORRECTNESS, A, B, C, D, LABEL
from utils_data import create_calibrated_df

random_state = 42
split = 'test'
n_bins = 10

for random_seed in [1, 2, 3, 4, 5]:
    df = create_calibrated_df(['output_xlnet_seed_%d_%s.csv' % (random_seed, split)])
    ece = ECE(n_bins)
    uncalibrated_score = ece.measure(df[[A, B, C, D]].values, df[LABEL].values)
    print('XLNET %d: ECE = %.4f' % (random_seed, float(uncalibrated_score)))

for random_seed in [0, 1, 2, 3, 42]:
    df = create_calibrated_df(['output_distilbert_seed%d_%s.csv' % (random_seed, split)])
    ece = ECE(n_bins)
    uncalibrated_score = ece.measure(df[[A, B, C, D]].values, df[LABEL].values)
    print('DistilBERT %d: ECE = %.4f' % (random_seed, float(uncalibrated_score)))

for random_seed in [0, 1, 2, 3, 42]:
    df = create_calibrated_df(['output_bert_seed%d_%s.csv' % (random_seed, split)])
    ece = ECE(n_bins)
    uncalibrated_score = ece.measure(df[[A, B, C, D]].values, df[LABEL].values)
    print('BERT %d: ECE = %.4f' % (random_seed, float(uncalibrated_score)))

# ENSEMBLES

# BERT
df = create_calibrated_df([
    'output_bert_seed0_%s.csv' % split,
    'output_bert_seed3_%s.csv' % split,
    'output_bert_seed42_%s.csv' % split,
])
ece = ECE(n_bins)
uncalibrated_score = ece.measure(df[[A, B, C, D]].values, df[LABEL].values)
print('BERT ENSEMBLE: ECE = %.4f' % (float(uncalibrated_score)))

# XLNet
df = create_calibrated_df([
    'output_xlnet_seed_2_%s.csv' % split,
    'output_xlnet_seed_3_%s.csv' % split,
    'output_xlnet_seed_4_%s.csv' % split,
])
ece = ECE(n_bins)
uncalibrated_score = ece.measure(df[[A, B, C, D]].values, df[LABEL].values)
print('XLNet ENSEMBLE: ECE = %.4f' % (float(uncalibrated_score)))

# DistilBERT
df = create_calibrated_df([
    'output_distilbert_seed1_%s.csv' % split,
    'output_distilbert_seed3_%s.csv' % split,
    'output_distilbert_seed42_%s.csv' % split,
])
ece = ECE(n_bins)
uncalibrated_score = ece.measure(df[[A, B, C, D]].values, df[LABEL].values)
print('DistilBERT ENSEMBLE: ECE = %.4f' % (float(uncalibrated_score)))

# BERT DistilBERT
df = create_calibrated_df([
    'output_bert_seed0_%s.csv' % split,
    'output_bert_seed3_%s.csv' % split,
    'output_bert_seed42_%s.csv' % split,
    'output_distilbert_seed1_%s.csv' % split,
    'output_distilbert_seed3_%s.csv' % split,
    'output_distilbert_seed42_%s.csv' % split,
])
ece = ECE(n_bins)
uncalibrated_score = ece.measure(df[[A, B, C, D]].values, df[LABEL].values)
print('BERT-DistilBERT ENSEMBLE: ECE = %.4f' % (float(uncalibrated_score)))

# BERT XLNet
df = create_calibrated_df([
    'output_bert_seed0_%s.csv' % split,
    'output_bert_seed3_%s.csv' % split,
    'output_bert_seed42_%s.csv' % split,
    'output_xlnet_seed_2_%s.csv' % split,
    'output_xlnet_seed_3_%s.csv' % split,
    'output_xlnet_seed_4_%s.csv' % split,
])
ece = ECE(n_bins)
uncalibrated_score = ece.measure(df[[A, B, C, D]].values, df[LABEL].values)
print('BERT-XLNet ENSEMBLE: ECE = %.4f' % (float(uncalibrated_score)))

# DistilBERT XLNet
df = create_calibrated_df([
    'output_distilbert_seed1_%s.csv' % split,
    'output_distilbert_seed3_%s.csv' % split,
    'output_distilbert_seed42_%s.csv' % split,
    'output_xlnet_seed_2_%s.csv' % split,
    'output_xlnet_seed_3_%s.csv' % split,
    'output_xlnet_seed_4_%s.csv' % split,
])
ece = ECE(n_bins)
uncalibrated_score = ece.measure(df[[A, B, C, D]].values, df[LABEL].values)
print('DistilBERT-XLNet ENSEMBLE: ECE = %.4f' % (float(uncalibrated_score)))

# BERT - DistilBERT - XLNet
df = create_calibrated_df([
    'output_bert_seed0_%s.csv' % split,
    'output_bert_seed3_%s.csv' % split,
    'output_bert_seed42_%s.csv' % split,
    'output_distilbert_seed1_%s.csv' % split,
    'output_distilbert_seed3_%s.csv' % split,
    'output_distilbert_seed42_%s.csv' % split,
    'output_xlnet_seed_2_%s.csv' % split,
    'output_xlnet_seed_3_%s.csv' % split,
    'output_xlnet_seed_4_%s.csv' % split,
])
ece = ECE(n_bins)
uncalibrated_score = ece.measure(df[[A, B, C, D]].values, df[LABEL].values)
print('BERT-DistilBERT-XLNet ENSEMBLE: ECE = %.4f' % (float(uncalibrated_score)))
