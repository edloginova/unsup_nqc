from utils_data import create_calibrated_df, prepare_dataset_for_high_vs_middle_prediction
from utils import plot_calibration, check_calibration, evaluate_level_prediction_high_vs_middle
from utils_constants import CORRECTNESS
import numpy as np

random_state = 42
split = 'test'

for random_seed in [1, 2, 3, 4, 5]:
    df = create_calibrated_df(['output_xlnet_seed_%d_%s.csv' % (random_seed, split)])
    print('XLNET %d: QA TEST ACCURACY = %.4f' % (random_seed, float(np.mean(df[CORRECTNESS]))))
    output_filename = 'output/xlnet_%d_test.txt' % random_seed
    output_file = open(output_filename, "w")
    plot_calibration(df, 0.1, image_name='output_figures/xlnet_%d_test.pdf' % random_seed)
    check_calibration(df, 0.1, output_file=output_file)
    df = prepare_dataset_for_high_vs_middle_prediction(df, output_file=output_file, random_state=random_state)
    evaluate_level_prediction_high_vs_middle(df, output_file=output_file)
    output_file.close()

for random_seed in [0, 1, 2, 3, 42]:
    df = create_calibrated_df(['output_distilbert_seed%d_%s.csv' % (random_seed, split)])
    print('DistilBERT %d: QA TEST ACCURACY = %.4f' % (random_seed, float(np.mean(df[CORRECTNESS]))))
    output_filename = 'output/distilbert%d_%s.txt' % (random_seed, split)
    output_file = open(output_filename, "w")
    plot_calibration(df, 0.1, image_name='output_figures/distilbert%d_%s.pdf' % (random_seed, split))
    check_calibration(df, 0.1, output_file=output_file)
    df = prepare_dataset_for_high_vs_middle_prediction(df, output_file=output_file, random_state=random_state)
    evaluate_level_prediction_high_vs_middle(df, output_file=output_file)
    output_file.close()

for random_seed in [0, 1, 2, 3, 42]:
    df = create_calibrated_df(['output_bert_seed%d_%s.csv' % (random_seed, split)])
    print('BERT %d: QA TEST ACCURACY = %.4f' % (random_seed, float(np.mean(df[CORRECTNESS]))))
    output_filename = 'output/bert%d_%s.txt' % (random_seed, split)
    output_file = open(output_filename, "w")
    plot_calibration(df, 0.1, image_name='output_figures/bert%d_%s.pdf' % (random_seed, split))
    check_calibration(df, 0.1, output_file=output_file)
    df = prepare_dataset_for_high_vs_middle_prediction(df, output_file=output_file, random_state=random_state)
    evaluate_level_prediction_high_vs_middle(df, output_file=output_file)
    output_file.close()
