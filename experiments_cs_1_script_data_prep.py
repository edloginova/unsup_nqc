from utils_mturk import get_race_lines, get_mturk_results_dataframe_raw_mturk_and_race_lines

lines = get_race_lines("data/race")
df_results_mturk = get_mturk_results_dataframe_raw_mturk_and_race_lines('data/output_mturk.csv', race_lines=lines)
df_results_mturk.to_csv('data/pairwise_race_cs.csv', index=False)
