import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

def active_cosine_similarity(user_vec, matrix):
    mask = ~np.isnan(user_vec)
    
    if mask.sum() == 0:
        return np.zeros(len(matrix), dtype=float)
        
    u_active = user_vec[mask]
    m_active = matrix[:, mask]
    
    dot = np.dot(m_active, u_active)
    u_norm = np.linalg.norm(u_active)
    m_norms = np.linalg.norm(m_active, axis=1)
    
    denom = u_norm * m_norms
    return np.divide(dot, denom, out=np.zeros(len(matrix), dtype=float), where=denom!=0)

def edit_df(df, ans, current_quest):
    if ans == 1.0:
        df = df[df[current_quest] >= -0.3]
    elif ans == -1.0:
        df = df[df[current_quest] <= 0.3]
        
    return df.drop(current_quest, axis='columns')

def retrain_model(df):
    X = df.drop('Character', axis=1)
    
    if X.empty or len(df['Character'].unique()) <= 1:
        return pd.Series(dtype='float64')
        
    names = df['Character']
    
    model = RandomForestClassifier(n_estimators=70, random_state=42)
    model.fit(X, names)
    
    return pd.Series(model.feature_importances_, index=X.columns)

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