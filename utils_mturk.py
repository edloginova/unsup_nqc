import os
import json
import tqdm
import glob
import numpy as np
import pandas as pd
from utils_constants import *


def perform_evaluation(df_for_evaluation, output_file=None):
    df = df_for_evaluation.copy()
    df['pred_max_2nd_diff'] = df.apply(lambda r: 1 if r['max_2nd_diff_q1'] < r['max_2nd_diff_q2'] else 2, axis=1)
    df['pred_max_oth_diff'] = df.apply(lambda r: 1 if r['max_others_diff_q1'] < r['max_others_diff_q2'] else 2, axis=1)
    df['pred_scores_var'] = df.apply(lambda r: 1 if r['scores_var_q1'] < r['scores_var_q2'] else 2, axis=1)
    df['pred_max_score'] = df.apply(lambda r: 1 if r['max_score_q1'] < r['max_score_q2'] else 2, axis=1)

    output_str = ""
    output_str += "GENERAL QUESTIONS\n"
    output_str += _return_output_string(df)
    output_str += "\n"
    output_str += "QUESTIONS WITH AGREEMENT == 3\n"
    tmp_df = df[df.agreement == 3]
    output_str += _return_output_string(tmp_df)
    output_str += "\n"
    output_str += "ONLY CORRECTLY ANSWERED QUESTIONS\n"
    tmp_df = df[(df[CORRECTNESS+'_q1'])&(df[CORRECTNESS+'_q2'])]
    output_str += _return_output_string(tmp_df)
    output_str += "\n"
    output_str += "ONLY CORRECTLY ANSWERED QUESTIONS WITH AGREEMENT == 3\n"
    tmp_df = tmp_df[tmp_df.agreement == 3]
    output_str += _return_output_string(tmp_df)
    if output_file is None:
        print(output_str)
    else:
        output_file.write(output_str)


def _return_output_string(df):
    output_str = '%d QUESTIONS\n' % len(df)
    output_str += "pred_max_2nd_diff %.4f\n" % np.mean(df.apply(lambda r: r[LABEL] == r['pred_max_2nd_diff'], axis=1))
    output_str += "pred_max_oth_diff %.4f\n" % np.mean(df.apply(lambda r: r[LABEL] == r['pred_max_oth_diff'], axis=1))
    output_str += "pred_scores_var   %.4f\n" % np.mean(df.apply(lambda r: r[LABEL] == r['pred_scores_var'], axis=1))
    output_str += "pred_max_score    %.4f\n" % np.mean(df.apply(lambda r: r[LABEL] == r['pred_max_score'], axis=1))
    return output_str


def prepare_df_for_evaluation(df_results_mturk, df):
    df_for_evaluation = pd.merge(
        df_results_mturk,
        df[[LEVEL, DOCUMENT_ID, ID, A, B, C, D, MAX_2ND_DIFF, MAX_OTH_DIFF, SCORES_VAR, MAX_SCORE, CORRECTNESS]],
        left_on=['idx_q1', LEVEL, DOCUMENT_ID],
        right_on=[ID, LEVEL, DOCUMENT_ID]
    ).rename(columns={
        ID: 'id_q1', A: 'A_q1', B: 'B_q1', C: 'C_q1', D: 'D_q1', MAX_2ND_DIFF: 'max_2nd_diff_q1',
        MAX_OTH_DIFF: 'max_others_diff_q1', SCORES_VAR: 'scores_var_q1', MAX_SCORE: 'max_score_q1',
        CORRECTNESS: CORRECTNESS+'_q1'
    })
    # shape (80 x 22)

    df_for_evaluation = pd.merge(
        df_for_evaluation,
        df[[LEVEL, DOCUMENT_ID, ID, A, B, C, D, MAX_2ND_DIFF, MAX_OTH_DIFF, SCORES_VAR, MAX_SCORE, CORRECTNESS]],
        left_on=['idx_q2', LEVEL, DOCUMENT_ID],
        right_on=[ID, LEVEL, DOCUMENT_ID]
    ).rename(columns={
        ID: 'id_q2', A: 'A_q2', B: 'B_q2', C: 'C_q2', D: 'D_q2', MAX_2ND_DIFF: 'max_2nd_diff_q2',
        MAX_OTH_DIFF: 'max_others_diff_q2', SCORES_VAR: 'scores_var_q2', MAX_SCORE: 'max_score_q2',
        CORRECTNESS: CORRECTNESS + '_q2'
    })
    return df_for_evaluation


def get_list_id_within_doc(df):
    list_id_within_doc = []
    previous_doc_id = None
    cnt = 0
    for document_id in df[DOCUMENT_ID].values:
        if document_id != previous_doc_id:
            cnt = 0
        else:
            cnt += 1
        previous_doc_id = document_id
        list_id_within_doc.append(cnt)
    return list_id_within_doc


def get_race_lines(race_data_dir):
    input_dir = os.path.join(race_data_dir, "test/high")
    lines = []
    files = glob.glob(input_dir + "/*txt")
    for file in tqdm.tqdm(files, desc="read files"):
        with open(file, "r", encoding="utf-8") as fin:
            data_raw = json.load(fin)
            data_raw["race_id"] = file
            lines.append(data_raw)
    input_dir = os.path.join(race_data_dir, "test/middle")
    files = glob.glob(input_dir + "/*txt")
    for file in tqdm.tqdm(files, desc="read files"):
        with open(file, "r", encoding="utf-8") as fin:
            data_raw = json.load(fin)
            data_raw["race_id"] = file
            lines.append(data_raw)
    return lines


def get_mturk_results_dataframe_raw_mturk_and_race_lines(df_raw_mturk_path, race_lines=None):

    df_raw_mturk = pd.read_csv(df_raw_mturk_path)[[DOCUMENT_ID, 'question_1', 'question_2', 'auth1', 'auth2', 'Turker']]

    df_raw_mturk['sum'] = df_raw_mturk.apply(lambda r: r['auth1'] + r['auth2'] + r['Turker'], axis=1)
    df_raw_mturk[LABEL] = df_raw_mturk.apply(lambda r: 1 if (r['sum'] == 3 or r['sum'] == 4) else 2, axis=1)
    df_raw_mturk['agreement'] = df_raw_mturk.apply(lambda r: 3 if (r['sum'] == 3 or r['sum'] == 6) else 2, axis=1)

    # lines = get_race_lines()
    filtered_lines = [x for x in race_lines if x[ID] in df_raw_mturk[DOCUMENT_ID].values]

    list_q1 = []
    list_q2 = []
    for line in filtered_lines:
        for idx, question in enumerate(line[QUESTIONS]):
            if question in df_raw_mturk['question_1'].values:
                if line[ID][0] == 'h':
                    level = HIGH
                    document_id = line[ID][4:]
                else:
                    level = MIDDLE
                    document_id = line[ID][6:]
                list_q1.append({
                      IDX: idx,
                      ID: line[ID],
                      LEVEL: level,
                      DOCUMENT_ID: document_id,
                      ARTICLE: line[ARTICLE],
                      QUESTION: question
                })
            if question in df_raw_mturk['question_2'].values:
                if line[ID][0] == 'h':
                    level = HIGH
                    document_id = line[ID][4:]
                else:
                    level = MIDDLE
                    document_id = line[ID][6:]
                list_q2.append({
                    IDX: idx,
                    ID: line[ID],
                    LEVEL: level,
                    DOCUMENT_ID: document_id,
                    ARTICLE: line[ARTICLE],
                    QUESTION: question
                })

    df_q1 = pd.DataFrame({
        LEVEL: [x[LEVEL] for x in list_q1],
        DOCUMENT_ID: [x[DOCUMENT_ID] for x in list_q1],
        'tmp_doc_id': [x[LEVEL]+x[DOCUMENT_ID] for x in list_q1],
        IDX: [x[IDX] for x in list_q1],
        QUESTION: [x[QUESTION] for x in list_q1],
    })

    df_q2 = pd.DataFrame({
        LEVEL: [x[LEVEL] for x in list_q2],
        DOCUMENT_ID: [x[DOCUMENT_ID] for x in list_q2],
        'tmp_doc_id': [x[LEVEL]+x[DOCUMENT_ID] for x in list_q2],
        IDX: [x[IDX] for x in list_q2],
        QUESTION: [x[QUESTION] for x in list_q2],
    })

    out_df = pd.merge(df_raw_mturk, df_q1, left_on=['question_1', DOCUMENT_ID], right_on=[QUESTION, 'tmp_doc_id'])\
        .drop(['tmp_doc_id', QUESTION], axis=1)\
        .rename(columns={'document_id_x': 'aggr_document_id', 'document_id_y': DOCUMENT_ID, IDX: 'idx_q1'})

    out_df = pd.merge(
        out_df,
        df_q2,
        left_on=['question_2', 'aggr_document_id', LEVEL, DOCUMENT_ID],
        right_on=[QUESTION, 'tmp_doc_id', LEVEL, DOCUMENT_ID]
    ).drop(['tmp_doc_id', QUESTION], axis=1).rename(columns={IDX: 'idx_q2'})

    out_df = out_df[
        [LEVEL, DOCUMENT_ID, 'aggr_document_id', 'question_1', 'idx_q1', 'question_2', 'idx_q2',
         'auth1', 'auth2', 'Turker', 'sum', LABEL, 'agreement']
    ]

    return out_df
