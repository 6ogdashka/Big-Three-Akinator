import pandas as pd
import numpy as np
from utility import edit_df, retrain_model, get_user_answer, active_cosine_similarity

importance = pd.read_csv('demo-feature-importance.csv', index_col=0).squeeze('columns')
df = pd.read_csv('demo-Blich-data-augmented.csv')
df_copy = df.copy()

X = df.drop('Character', axis=1)
matrix_full = df_copy.drop('Character', axis=1).values
user_responses = pd.Series(np.nan, index=X.columns)

min_questions = 6

while len(importance) > 0:
    user_vec = user_responses.values
    scores = active_cosine_similarity(user_vec, matrix_full)
    current_similarity = scores.max()
    
    answered_count = (~np.isnan(user_vec)).sum()
    
    if answered_count >= min_questions and current_similarity >= 0.95:
        if (scores >= 0.95).sum() == 1:
            break

    top_features = importance.nlargest(3).index.tolist()
    current_quest = np.random.choice(top_features)
    
    ans = get_user_answer(current_quest)
    user_responses[current_quest] = ans

    df = edit_df(df, ans, current_quest)
    importance = retrain_model(df)

user_vec = user_responses.values
scores = active_cosine_similarity(user_vec, matrix_full)
max_score = scores.max()

if max_score == 0.0:
    print("Ни один персонаж не подошел")
else:
    best_match = scores.argmax()
    best_character = df_copy['Character'].iloc[best_match]
    print(f"\nВаш персонаж: {best_character} (Сходство: {max_score:.2%})")