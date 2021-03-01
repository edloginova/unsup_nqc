from utils_data import create_calibrated_df, prepare_dataset_for_high_vs_middle_prediction
from utils import plot_calibration, check_calibration, evaluate_level_prediction_high_vs_middle
import numpy as np
from utils_constants import CORRECTNESS

random_state = 42
split = 'test'

# BERT
df = create_calibrated_df([
    'output_bert_seed0_%s.csv' % split,
    'output_bert_seed3_%s.csv' % split,
    'output_bert_seed42_%s.csv' % split,
])
print('BERT: QA TEST ACCURACY = %.4f' % float(np.mean(df[CORRECTNESS])))
output_filename = 'output/bert_ensemble_%s.txt' % split
output_file = open(output_filename, "w")
plot_calibration(df, 0.1, image_name='output_figures/bert_ensemble_%s.pdf' % split)
check_calibration(df, 0.1, output_file=output_file)
df = prepare_dataset_for_high_vs_middle_prediction(df, output_file=output_file, random_state=random_state)
evaluate_level_prediction_high_vs_middle(df, output_file=output_file)
output_file.close()

# XLNet
df = create_calibrated_df([
    'output_xlnet_seed_2_%s.csv' % split,
    'output_xlnet_seed_3_%s.csv' % split,
    'output_xlnet_seed_4_%s.csv' % split,
])
print('XLNet: QA TEST ACCURACY = %.4f' % float(np.mean(df[CORRECTNESS])))
output_filename = 'output/xlnet_ensemble_%s.txt' % split
output_file = open(output_filename, "w")
plot_calibration(df, 0.1, image_name='output_figures/xlnet_ensemble_%s.pdf' % split)
check_calibration(df, 0.1, output_file=output_file)
df = prepare_dataset_for_high_vs_middle_prediction(df, output_file=output_file, random_state=random_state)
evaluate_level_prediction_high_vs_middle(df, output_file=output_file)
output_file.close()

# DistilBERT
df = create_calibrated_df([
    'output_distilbert_seed1_%s.csv' % split,
    'output_distilbert_seed3_%s.csv' % split,
    'output_distilbert_seed42_%s.csv' % split,
])
print('DistilBERT: QA TEST ACCURACY = %.4f' % float(np.mean(df[CORRECTNESS])))
output_filename = 'output/distilbert_ensemble_%s.txt' % split
output_file = open(output_filename, "w")
plot_calibration(df, 0.1, image_name='output_figures/distilbert_ensemble_%s.pdf' % split)
check_calibration(df, 0.1, output_file=output_file)
df = prepare_dataset_for_high_vs_middle_prediction(df, output_file=output_file, random_state=random_state)
evaluate_level_prediction_high_vs_middle(df, output_file=output_file)
output_file.close()

# BERT DistilBERT
df = create_calibrated_df([
    'output_bert_seed0_%s.csv' % split,
    'output_bert_seed3_%s.csv' % split,
    'output_bert_seed42_%s.csv' % split,
    'output_distilbert_seed1_%s.csv' % split,
    'output_distilbert_seed3_%s.csv' % split,
    'output_distilbert_seed42_%s.csv' % split,
])
print('BERT-DistilBERT: QA TEST ACCURACY = %.4f' % float(np.mean(df[CORRECTNESS])))
output_filename = 'output/bert_distilbert_ensemble_%s.txt' % split
output_file = open(output_filename, "w")
plot_calibration(df, 0.1, image_name='output_figures/bert_distilbert_ensemble_%s.pdf' % split)
check_calibration(df, 0.1, output_file=output_file)
df = prepare_dataset_for_high_vs_middle_prediction(df, output_file=output_file, random_state=random_state)
evaluate_level_prediction_high_vs_middle(df, output_file=output_file)
output_file.close()

# BERT XLNet
df = create_calibrated_df([
    'output_bert_seed0_%s.csv' % split,
    'output_bert_seed3_%s.csv' % split,
    'output_bert_seed42_%s.csv' % split,
    'output_xlnet_seed_2_%s.csv' % split,
    'output_xlnet_seed_3_%s.csv' % split,
    'output_xlnet_seed_4_%s.csv' % split,
])
print('BERT-XLNet: QA TEST ACCURACY = %.4f' % float(np.mean(df[CORRECTNESS])))
output_filename = 'output/bert_xlnet_ensemble_%s.txt' % split
output_file = open(output_filename, "w")
plot_calibration(df, 0.1, image_name='output_figures/bert_xlnet_ensemble_%s.pdf' % split)
check_calibration(df, 0.1, output_file=output_file)
df = prepare_dataset_for_high_vs_middle_prediction(df, output_file=output_file, random_state=random_state)
evaluate_level_prediction_high_vs_middle(df, output_file=output_file)
output_file.close()

# DistilBERT XLNet
df = create_calibrated_df([
    'output_distilbert_seed1_%s.csv' % split,
    'output_distilbert_seed3_%s.csv' % split,
    'output_distilbert_seed42_%s.csv' % split,
    'output_xlnet_seed_2_%s.csv' % split,
    'output_xlnet_seed_3_%s.csv' % split,
    'output_xlnet_seed_4_%s.csv' % split,
])
print('DistilBERT-XLNet: QA TEST ACCURACY = %.4f' % float(np.mean(df[CORRECTNESS])))
output_filename = 'output/distilbert_xlnet_ensemble_%s.txt' % split
output_file = open(output_filename, "w")
plot_calibration(df, 0.1, image_name='output_figures/distilbert_xlnet_ensemble_%s.pdf' % split)
check_calibration(df, 0.1, output_file=output_file)
df = prepare_dataset_for_high_vs_middle_prediction(df, output_file=output_file, random_state=random_state)
evaluate_level_prediction_high_vs_middle(df, output_file=output_file)
output_file.close()

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
print('BERT-DistilBERT-XLNet: QA TEST ACCURACY = %.4f' % float(np.mean(df[CORRECTNESS])))
output_filename = 'output/bert_distilbert_xlnet_ensemble_%s.txt' % split
output_file = open(output_filename, "w")
plot_calibration(df, 0.1, image_name='output_figures/bert_distilbert_xlnet_ensemble_%s.pdf' % split)
check_calibration(df, 0.1, output_file=output_file)
df = prepare_dataset_for_high_vs_middle_prediction(df, output_file=output_file, random_state=random_state)
evaluate_level_prediction_high_vs_middle(df, output_file=output_file)
output_file.close()
