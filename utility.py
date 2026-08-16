import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

VALUE_MAP = {
    (-1.0, -1.0): 3, (1.0, 1.0): 3,
    (-1.0, -0.7): 1, (0.7, 1.0): 1, (-0.7, -0.7): 1, (0.7, 0.7): 1, (0.0, 0.0): 1,
    (-1.0, 0.0): 0,  (1.0, 0.0): 0, (-0.7, 0.0): 0,  (0.7, 0.0): 0,
    (-1.0, 0.7): -2, (-0.7, 1.0): -2, (-0.7, 0.7): -1, 
    (-1.0, 1.0): -5 
}

def get_score_point(vec1, vec2):
    return sum(VALUE_MAP.get(tuple(sorted((x, y))), 0) for x, y in zip(vec1, vec2))

def _get_pair_score(val1: float, val2: float) -> int:
    return VALUE_MAP.get(tuple(sorted((val1, val2))), 0)

def get_interest_point(vec1, vec2):
    score_12 = sum(VALUE_MAP.get(tuple(sorted((x, y))), 0) for x, y in zip(vec1, vec2))
    score_11 = sum(VALUE_MAP.get((x, x), 0) for x in vec1)
    score_22 = sum(VALUE_MAP.get((y, y), 0) for y in vec2)
    denominator = np.sqrt(score_11 * score_22)
    if denominator == 0:
        return 0.0
    return score_12 / denominator

def edit_df(df: pd.DataFrame, quest: str, ans: float, step: int = 1) -> pd.DataFrame:
    if '_score' not in df.columns:
        df = df.copy()
        df['_score'] = 0.0

    score_map = {val: _get_pair_score(val, ans) for val in (-1.0, -0.7, 0.0, 0.7, 1.0)}
    df['_score'] += df[quest].map(score_map).fillna(0)

    max_score = df['_score'].max()
    
    if max_score <= 3:
        threshold = 0.0
    else:
        dynamic_ratio = min(0.90, 0.40 + 0.05 * step)
        threshold = max_score * dynamic_ratio

    return df[df['_score'] >= threshold].drop(columns=[quest])

def retrain_model(df):
    X = df.drop(['Character', '_score'], axis=1, errors='ignore')
    if X.empty or len(df['Character'].unique()) <= 1:
        return pd.Series(dtype='float64')
    names = df['Character']
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X, names)
    return pd.Series(model.feature_importances_, index=X.columns)

def get_question(fi_df: pd.DataFrame, df: pd.DataFrame, remaining_chars: list) -> str:
    if 0 < len(remaining_chars) <= 30:
        sub_df = df[df['Character'].isin(remaining_chars)]
        feature_cols = [c for c in sub_df.columns if c not in ['Character', '_score']]
        if feature_cols:
            variances = sub_df[feature_cols].var()
            if not variances.empty:
                return variances.idxmax()
                
    top_quest = np.array(fi_df.index[:3])
    return np.random.choice(top_quest)

def get_user_answer(question_text):
    answers_map = {
        'yes': 1.0,
        'mb yes': 0.7,
        'idk': 0.0,
        'mb no': -0.7,
        'no': -1.0
    }
    print(f"\nВопрос: {question_text}")
    print("Варианты ответа: yes, mb yes, idk, mb no, no")
    while True:
        user_input = input("Ваш ответ: ").strip().lower()
        if user_input in answers_map:
            return answers_map[user_input]
        print("Неверный ввод. Пожалуйста, выберите из: yes, mb yes, idk, mb no, no")