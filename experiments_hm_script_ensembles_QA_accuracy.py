from utils_data import create_calibrated_df
import numpy as np
from utils_constants import CORRECTNESS

splits = ['eval', 'test']

for split in splits:
    # BERT
    df = create_calibrated_df([
        'output_bert_seed0_%s.csv' % split,
        'output_bert_seed3_%s.csv' % split,
        'output_bert_seed42_%s.csv' % split,
    ])
    print('BERT: QA %s ACCURACY = %.4f' % (split, float(np.mean(df[CORRECTNESS]))))

    # XLNet
    df = create_calibrated_df([
        'output_xlnet_seed_2_%s.csv' % split,
        'output_xlnet_seed_3_%s.csv' % split,
        'output_xlnet_seed_4_%s.csv' % split,
    ])
    print('XLNet: QA %s ACCURACY = %.4f' % (split, float(np.mean(df[CORRECTNESS]))))

    # DistilBERT
    df = create_calibrated_df([
        'output_distilbert_seed1_%s.csv' % split,
        'output_distilbert_seed3_%s.csv' % split,
        'output_distilbert_seed42_%s.csv' % split,
    ])
    print('DistilBERT: QA %s ACCURACY = %.4f' % (split, float(np.mean(df[CORRECTNESS]))))


    # BERT DistilBERT
    df = create_calibrated_df([
        'output_bert_seed0_%s.csv' % split,
        'output_bert_seed3_%s.csv' % split,
        'output_bert_seed42_%s.csv' % split,
        'output_distilbert_seed1_%s.csv' % split,
        'output_distilbert_seed3_%s.csv' % split,
        'output_distilbert_seed42_%s.csv' % split,
    ])
    print('BERT-DistilBERT: QA %s ACCURACY = %.4f' % (split, float(np.mean(df[CORRECTNESS]))))

    # BERT XLNet
    df = create_calibrated_df([
        'output_bert_seed0_%s.csv' % split,
        'output_bert_seed3_%s.csv' % split,
        'output_bert_seed42_%s.csv' % split,
        'output_xlnet_seed_2_%s.csv' % split,
        'output_xlnet_seed_3_%s.csv' % split,
        'output_xlnet_seed_4_%s.csv' % split,
    ])
    print('BERT-XLNet: QA %s ACCURACY = %.4f' % (split, float(np.mean(df[CORRECTNESS]))))

    # DistilBERT XLNet
    df = create_calibrated_df([
        'output_distilbert_seed1_%s.csv' % split,
        'output_distilbert_seed3_%s.csv' % split,
        'output_distilbert_seed42_%s.csv' % split,
        'output_xlnet_seed_2_%s.csv' % split,
        'output_xlnet_seed_3_%s.csv' % split,
        'output_xlnet_seed_4_%s.csv' % split,
    ])
    print('DistilBERT-XLNet: QA %s ACCURACY = %.4f' % (split, float(np.mean(df[CORRECTNESS]))))

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
    print('BERT-DistilBERT-XLNet: QA %s ACCURACY = %.4f' % (split, float(np.mean(df[CORRECTNESS]))))
