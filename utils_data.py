import os

import numpy as np
import pandas as pd

from utils_constants import *
from utils_math import softmax


def create_calibrated_df(list_filenames, data_dir='data', to_compute_additional_cols=True):
    df = pd.read_csv(os.path.join(data_dir, list_filenames[0]))

    # two different behaviors depending on whether I only have to softmax one model or the ensemble of several models
    if len(list_filenames) == 1:
        df = df.rename(columns={'score %s' % label: '%s' % label for label in LABELS})
        df[TMP_PRED] = df.apply(lambda r: softmax([r['%s' % label] for label in LABELS]), axis=1)
        for idx, label in enumerate(LABELS):
            df['%s' % label] = df.apply(lambda r: r[TMP_PRED][idx], axis=1)
    else:
        df = df.rename(columns={'score %s' % label: '%s %d' % (label, 0) for label in LABELS})
        df[TMP_PRED] = df.apply(lambda r: softmax([r['%s %d' % (label, 0)] for label in LABELS]), axis=1)
        for idx, label in enumerate(LABELS):
            df['%s %d' % (label, 0)] = df.apply(lambda r: r[TMP_PRED][idx], axis=1)
        # add the predictions of the other seed (already performed softmax)
        for idx, filename in enumerate(list_filenames[1:]):
            tmp_df = pd.read_csv(os.path.join(data_dir, filename))
            tmp_df = tmp_df.rename(columns={'score %s' % label: '%s %d' % (label, idx + 1) for label in LABELS})
            tmp_df[TMP_PRED] = tmp_df.apply(lambda r: softmax([r['%s %d' % (label, idx + 1)] for label in LABELS]),
                                            axis=1)
            for idx2, label in enumerate(LABELS):
                tmp_df['%s %d' % (label, idx + 1)] = tmp_df.apply(lambda r: r[TMP_PRED][idx2], axis=1)
            for label in LABELS:
                df['%s %d' % (label, idx + 1)] = tmp_df['%s %d' % (label, idx + 1)]
        # average the score of each label (between the different seed)
        for label in LABELS:
            df[label] = df.apply(lambda r: np.mean([r['%s %d' % (label, idx3)] for idx3 in range(len(list_filenames))]),
                                 axis=1)
    # keep only the columns I want
    df = df[[IDX, LEVEL, DOCUMENT_ID, LABEL, A, B, C, D]]
    # perform the prediction
    df[PREDICTION] = df.apply(lambda r: np.argmax([r['%s' % label] for label in LABELS]), axis=1)
    # compute the additional columns
    df[CORRECTNESS] = df.apply(lambda r: r[LABEL] == r[PREDICTION], axis=1)
    df[SCORES] = df.apply(lambda r: [r[A], r[B], r[C], r[D]], axis=1)
    df[MAX_2ND_DIFF] = df.apply(lambda r: np.max(r[SCORES]) - np.sort(r[SCORES])[-2], axis=1)
    df[MAX_OTH_DIFF] = df.apply(lambda r: np.max(r[SCORES]) - (np.sum(r[SCORES]) - np.max(r[SCORES])) / 3.0, axis=1)
    df[SCORES_VAR] = df.apply(lambda r: np.var(r[SCORES]), axis=1)
    df[MAX_SCORE] = df.apply(lambda r: np.max(r[SCORES]), axis=1)
    # compute the standardized columns
    for column in [MAX_2ND_DIFF, MAX_OTH_DIFF, SCORES_VAR, MAX_SCORE]:
        df['standardized_' + column] = (df[column].values - df[column].mean()) / df[column].std()
    return df[COLUMNS]


def prepare_dataset_for_high_vs_middle_prediction(df, max_len=2000, output_file=None, random_state=None):
    if output_file is None:
        print("Num. high questions", len(df[df[LEVEL] == HIGH]))
        print("Num. middle questions", len(df[df[LEVEL] == MIDDLE]))
    else:
        output_file.write("Num. high questions " + str(len(df[df[LEVEL] == HIGH])) + "\n")
        output_file.write("Num. middle questions " + str(len(df[df[LEVEL] == MIDDLE])) + "\n")

    df_high = df[df[LEVEL] == HIGH].copy()[PREDICTION_COLUMNS]
    df_high = df_high.rename(columns={x: x + '_h' for x in df_high.columns})
    df_high['key'] = 1

    df_middle = df[df[LEVEL] == MIDDLE].copy()[PREDICTION_COLUMNS]
    df_middle = df_middle.rename(columns={x: x + '_m' for x in df_middle.columns})
    df_middle['key'] = 1

    length = min(len(df_high), len(df_middle), max_len)
    if output_file is None:
        print("Considered %d items for each level" % length)
    else:
        output_file.write("Considered %d items for each level\n" % length)
    return pd.merge(
        df_high.sample(length, random_state=random_state), df_middle.sample(length, random_state=random_state), on='key'
    )
