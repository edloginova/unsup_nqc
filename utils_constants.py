IDX = 'idx'
LEVEL = 'level'
DOCUMENT_ID = 'document_id'
LABEL = 'label'
A = 'A'
B = 'B'
C = 'C'
D = 'D'
PREDICTION = 'prediction'
CORRECTNESS = 'correctness'
SCORES = 'scores'
MAX_2ND_DIFF = 'max_2nd_diff'
MAX_OTH_DIFF = 'max_others_diff'
SCORES_VAR = 'scores_var'
MAX_SCORE = 'max_score'
STD_MAX_2ND_DIFF = 'standardized_max_2nd_diff'
STD_MAX_OTH_DIFF = 'standardized_max_others_diff'
STD_SCORES_VAR = 'standardized_scores_var'
STD_MAX_SCORE = 'standardized_max_score'

LABELS = [A, B, C, D]
COLUMNS = [
    IDX, LEVEL, DOCUMENT_ID, LABEL, A, B, C, D, PREDICTION, CORRECTNESS, SCORES, MAX_2ND_DIFF, MAX_OTH_DIFF, SCORES_VAR,
    MAX_SCORE, STD_MAX_2ND_DIFF, STD_MAX_OTH_DIFF, STD_SCORES_VAR, STD_MAX_SCORE
]
PREDICTION_COLUMNS = [
    IDX, CORRECTNESS, MAX_2ND_DIFF, MAX_OTH_DIFF, SCORES_VAR, MAX_SCORE,
    STD_MAX_2ND_DIFF, STD_MAX_OTH_DIFF, STD_SCORES_VAR, STD_MAX_SCORE
]

TMP_PRED = 'tmp_predictions'

HIGH = 'high'
MIDDLE = 'middle'

ID = 'id'
QUESTIONS = 'questions'
ARTICLE = 'article'
QUESTION = 'question'
