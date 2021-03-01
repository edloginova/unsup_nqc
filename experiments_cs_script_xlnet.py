import pandas as pd

from utils_data import create_calibrated_df
from utils_mturk import get_list_id_within_doc, prepare_df_for_evaluation, perform_evaluation

# data preparation
df_results_mturk = pd.read_csv('data/pairwise_race_cs.csv')

df_predictions = create_calibrated_df([
    'output_xlnet_seed_2_test.csv',
    'output_xlnet_seed_3_test.csv',
    'output_xlnet_seed_4_test.csv'
])
list_id_within_doc = get_list_id_within_doc(df_predictions)
df_predictions['id'] = list_id_within_doc
df_for_evaluation = prepare_df_for_evaluation(df_results_mturk, df_predictions)
output_filename = 'output/pairwise_race_cs_xlnet_ensemble_test.txt'
output_file = open(output_filename, "w")
perform_evaluation(df_for_evaluation, output_file=output_file)
output_file.close()

# Single models
for random_seed in [2, 3, 4]:
    df_predictions = create_calibrated_df(['output_xlnet_seed_%d_test.csv' % random_seed])
    list_id_within_doc = get_list_id_within_doc(df_predictions)
    df_predictions['id'] = list_id_within_doc
    df_for_evaluation = prepare_df_for_evaluation(df_results_mturk, df_predictions)
    output_filename = 'output/pairwise_race_cs_xlnet_%d_test.txt' % random_seed
    output_file = open(output_filename, "w")
    perform_evaluation(df_for_evaluation, output_file=output_file)
    output_file.close()
