import warnings
import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")

VALUE_MAP = {
    (-1.0, -1.0): 3, (1.0, 1.0): 3,
    (-1.0, -0.7): 1, (0.7, 1.0): 1, (-0.7, -0.7): 1, (0.7, 0.7): 1, (0.0, 0.0): 1,
    (-1.0, 0.0): 0,  (1.0, 0.0): 0, (-0.7, 0.0): 0,  (0.7, 0.0): 0,
    (-1.0, 0.7): -2, (-0.7, 1.0): -2, (-0.7, 0.7): -1,
    (-1.0, 1.0): -5  
}

DOMAIN_RULES = {
    "Является ли он синигами (проводником душ)?": [
        "Достиг ли этот персонаж банкая?", 
        "Является ли он материализованным духом занпакто?"
    ],
    "Съел ли персонаж Дьявольский фрукт?": [
        "Относится ли Дьявольский фрукт к типу Парамеция?", 
        "Относится ли Дьявольский фрукт к типу Логия?", 
        "Относится ли Дьявольский фрукт к типу Обычный Зоан?", 
        "Относится ли Дьявольский фрукт к типу Древний Зоан?", 
        "Относится ли Дьявольский фрукт к типу Мифический Зоан?", 
        "Является ли Дьявольский фрукт Искусственным Зоаном (Смайлом)?"
    ]
}

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
    
    if ans <= -0.7 and quest in DOMAIN_RULES:
        cols_to_drop = [c for c in DOMAIN_RULES[quest] if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

    max_score = df['_score'].max()
    
    if max_score <= 3:
        threshold = 0.0
    else:
        dynamic_ratio = min(0.90, 0.40 + 0.05 * step)
        threshold = max_score * dynamic_ratio

    return df[df['_score'] >= threshold].drop(columns=[quest], errors='ignore')

def retrain_model(df):
    X = df.drop(['Character', '_score'], axis=1, errors='ignore')
    if X.empty or len(df['Character'].unique()) <= 1:
        return pd.Series(dtype='float64')
    names = df['Character']
    
    if HAS_CATBOOST:
        model = CatBoostClassifier(iterations=50, thread_count=-1, logging_level='Silent', random_state=42)
        model.fit(X, names)
        return pd.Series(model.get_feature_importance(), index=X.columns).sort_values(ascending=False)
    else:
        model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
        model.fit(X, names)
        return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

def calculate_entropy(series):
    counts = series.value_counts()
    probabilities = counts / len(series)
    return entropy(probabilities, base=2)

def get_question(fi_df: pd.Series, df: pd.DataFrame, remaining_chars: list) -> str:
    if len(df) > 100:
        global_boosters = [
            "Является ли он синигами (проводником душ)?",
            "Съел ли персонаж Дьявольский фрукт?",
            "Ваш персонаж женского пола?",
            "Является ли он преступником(пиратом разбойником)"
        ]
        for booster in global_boosters:
            if booster in fi_df.index and booster in df.columns:
                sub_vals = df[booster].unique()
                if len(sub_vals) > 1:
                    return booster

    if 0 < len(remaining_chars) <= 30:
        sub_df = df[df['Character'].isin(remaining_chars)]
        feature_cols = [c for c in sub_df.columns if c not in ['Character', '_score']]
        if feature_cols:
            entropies = sub_df[feature_cols].apply(calculate_entropy)
            if not entropies.empty and entropies.max() > 0:
                return entropies.idxmax()
                
    top_quest = list(fi_df.index[:3])
    if top_quest:
        return np.random.choice(top_quest)
    return ""

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