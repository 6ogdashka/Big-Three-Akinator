import pandas as pd
import numpy as np

VALUE_MAP = {
    (-1.0, -1.0): 3, (1.0, 1.0): 3,
    (-1.0, -0.7): 1, (0.7, 1.0): 1, (-0.7, -0.7): 1, (0.7, 0.7): 1, (0.0, 0.0): 1,
    (-1.0, 0.0): 0,  (1.0, 0.0): 0, (-0.7, 0.0): 0,  (0.7, 0.0): 0,
    (-1.0, 0.7): -1, (-0.7, 1.0): -1, (-0.7, 0.7): -1,
    (-1.0, 1.0): -3
}

def _get_pair_score(val1: float, val2: float) -> int:
    return VALUE_MAP.get(tuple(sorted((val1, val2))), 0)


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
        dynamic_ratio = min(0.85, 0.30 + 0.025 * step)
        threshold = max_score * dynamic_ratio

    return df[df['_score'] >= threshold].drop(columns=[quest])


def get_next_best_question(df: pd.DataFrame, fi_df: pd.DataFrame, top_k: int = 3) -> str:
    available_questions = list(set(df.columns) - {'Character', '_score'})
    if not available_questions:
        return None

    if 2 <= len(df) <= 5:
        top_candidates = df.sort_values(by='_score', ascending=False).head(2)
        v1 = top_candidates.iloc[0][available_questions]
        v2 = top_candidates.iloc[1][available_questions]
        
        diffs = (v1 - v2).abs()
        valid_diffs = diffs[diffs > 0]
        
        if not valid_diffs.empty:
            best_dueling_candidates = valid_diffs.nlargest(top_k).index.tolist()
            return np.random.choice(best_dueling_candidates)

    feature_col = fi_df.columns[0]
    for col in fi_df.columns:
        if col.lower() in ['feature', 'question', 'признак', 'вопрос', 'unnamed: 0']:
            feature_col = col
            break

    if len(fi_df.columns) > 1:
        score_col = fi_df.columns[1] if fi_df.columns[0] == feature_col else fi_df.columns[0]
        fi_sorted = fi_df.sort_values(by=score_col, ascending=False)
    else:
        fi_sorted = fi_df

    top_features = []
    for feat in fi_sorted[feature_col]:
        if feat in available_questions:
            top_features.append(feat)
            if len(top_features) == top_k:
                break

    if top_features:
        return np.random.choice(top_features)

    return np.random.choice(available_questions)


def get_user_answer(question_text: str) -> float:
    answers_map = {'yes': 1.0, 'mb yes': 0.7, 'idk': 0.0, 'mb no': -0.7, 'no': -1.0}
    print(f"\nВопрос: {question_text}")
    print("Варианты ответа: yes, mb yes, idk, mb no, no")
    
    while True:
        user_input = input("Ваш ответ: ").strip().lower()
        if user_input in answers_map:
            return answers_map[user_input]
        print("Неверный ввод. Выберите из: yes, mb yes, idk, mb no, no")


def main():
    TOTAL_QUESTIONS = 20

    data_path = "demo-Blich-data-augmented.csv"
    fi_path = "demo-feature-importance.csv"

    try:
        df = pd.read_csv(data_path)
        fi_df = pd.read_csv(fi_path)
    except FileNotFoundError as e:
        print(f"Ошибка загрузки файлов: {e}")
        return

    print(f"Загружен датасет: {len(df)} персонажей.")

    for step in range(1, TOTAL_QUESTIONS + 1):
        best_question = get_next_best_question(df, fi_df, top_k=3)
        
        if not best_question:
            print("\nВопросы закончились!")
            break

        print(f"\n--- Шаг {step} из {TOTAL_QUESTIONS} ---")
        ans = get_user_answer(best_question)
        
        df = edit_df(df, best_question, ans, step=step)
        print(f"Осталось персонажей: {len(df)}")

        if len(df) <= 1:
            print("\nОстался 1 явный лидер!")
            break

    if '_score' in df.columns:
        df = df.sort_values(by='_score', ascending=False)

    output_path = "test_filtered_result.csv"
    df.to_csv(output_path, index=False)
    print(f"\nТест завершен. Итоговая таблица: '{output_path}'")


if __name__ == "__main__":
    main()