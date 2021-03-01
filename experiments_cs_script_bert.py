import pandas as pd

from utils_data import create_calibrated_df
from utils_mturk import get_list_id_within_doc, prepare_df_for_evaluation, perform_evaluation

# data preparation
df_results_mturk = pd.read_csv('data/pairwise_race_cs.csv')

# Single models
for random_seed in [0, 3, 42]:
    df_predictions = create_calibrated_df(['output_bert_seed%d_test.csv' % random_seed])
    list_id_within_doc = get_list_id_within_doc(df_predictions)
    df_predictions['id'] = list_id_within_doc
    df_for_evaluation = prepare_df_for_evaluation(df_results_mturk, df_predictions)
    output_filename = 'output/pairwise_race_cs_bert_%d_test.txt' % random_seed
    output_file = open(output_filename, "w")
    perform_evaluation(df_for_evaluation, output_file=output_file)
    output_file.close()

# ensemble
df_predictions = create_calibrated_df([
    'output_bert_seed0_test.csv', 'output_bert_seed3_test.csv', 'output_bert_seed42_test.csv'])
list_id_within_doc = get_list_id_within_doc(df_predictions)
df_predictions['id'] = list_id_within_doc
df_for_evaluation = prepare_df_for_evaluation(df_results_mturk, df_predictions)
output_filename = 'output/pairwise_race_cs_bert_ensemble_test.txt'
output_file = open(output_filename, "w")
perform_evaluation(df_for_evaluation, output_file=output_file)
output_file.close()
