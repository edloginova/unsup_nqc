from utils_data import create_calibrated_df
from utils_constants import CORRECTNESS

split = 'test'

# BERT
df = create_calibrated_df([
    'output_bert_seed0_%s.csv' % split,
    'output_bert_seed3_%s.csv' % split,
    'output_bert_seed42_%s.csv' % split,
])
print('BERT: N. Correct = %5d; total N = %5d' % (len(df[df[CORRECTNESS]]), len(df)))

# XLNet
df = create_calibrated_df([
    'output_xlnet_seed_2_%s.csv' % split,
    'output_xlnet_seed_3_%s.csv' % split,
    'output_xlnet_seed_4_%s.csv' % split,
])
print('XLNet: N. Correct = %5d; total N = %5d' % (len(df[df[CORRECTNESS]]), len(df)))

# DistilBERT
df = create_calibrated_df([
    'output_distilbert_seed1_%s.csv' % split,
    'output_distilbert_seed3_%s.csv' % split,
    'output_distilbert_seed42_%s.csv' % split,
])
print('DistilBERT: N. Correct = %5d; total N = %5d' % (len(df[df[CORRECTNESS]]), len(df)))

# BERT DistilBERT
df = create_calibrated_df([
    'output_bert_seed0_%s.csv' % split,
    'output_bert_seed3_%s.csv' % split,
    'output_bert_seed42_%s.csv' % split,
    'output_distilbert_seed1_%s.csv' % split,
    'output_distilbert_seed3_%s.csv' % split,
    'output_distilbert_seed42_%s.csv' % split,
])
print('BERT-DistilBERT: N. Correct = %5d; total N = %5d' % (len(df[df[CORRECTNESS]]), len(df)))

# BERT XLNet
df = create_calibrated_df([
    'output_bert_seed0_%s.csv' % split,
    'output_bert_seed3_%s.csv' % split,
    'output_bert_seed42_%s.csv' % split,
    'output_xlnet_seed_2_%s.csv' % split,
    'output_xlnet_seed_3_%s.csv' % split,
    'output_xlnet_seed_4_%s.csv' % split,
])
print('BERT-XLNet: N. Correct = %5d; total N = %5d' % (len(df[df[CORRECTNESS]]), len(df)))

# DistilBERT XLNet
df = create_calibrated_df([
    'output_distilbert_seed1_%s.csv' % split,
    'output_distilbert_seed3_%s.csv' % split,
    'output_distilbert_seed42_%s.csv' % split,
    'output_xlnet_seed_2_%s.csv' % split,
    'output_xlnet_seed_3_%s.csv' % split,
    'output_xlnet_seed_4_%s.csv' % split,
])
print('DistilBERT-XLNet: N. Correct = %5d; total N = %5d' % (len(df[df[CORRECTNESS]]), len(df)))

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
print('BERT - DistilBERT - XLNet: N. Correct = %5d; total N = %5d' % (len(df[df[CORRECTNESS]]), len(df)))

# single models
model_names = [
    'output_bert_seed0_%s.csv' % split,
    'output_bert_seed3_%s.csv' % split,
    'output_bert_seed42_%s.csv' % split,
    'output_distilbert_seed1_%s.csv' % split,
    'output_distilbert_seed3_%s.csv' % split,
    'output_distilbert_seed42_%s.csv' % split,
    'output_xlnet_seed_2_%s.csv' % split,
    'output_xlnet_seed_3_%s.csv' % split,
    'output_xlnet_seed_4_%s.csv' % split,
]
for model_name in model_names:
    df = create_calibrated_df([model_name])
    print('%s: N. Correct = %5d; total N = %5d' % (model_name, len(df[df[CORRECTNESS]]), len(df)))
