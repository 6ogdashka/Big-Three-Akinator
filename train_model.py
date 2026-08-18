import pandas as pd
from utility import retrain_model, get_question, get_user_answer, edit_df, get_interest_point

class AkinatorEngine:
    def __init__(self, original_df):
        self.original_df = original_df.copy()
        self.reset_game()
        
    def reset_game(self):
        self.df = self.original_df.copy()
        self.step = 1
        self.asked_questions = []
        self.user_answers = []
        self.is_finished = False
        self.result_message = ""
        
    def play_step(self):
        if self.is_finished:
            return
            
        # Передаем step в retrain_model для гибридной оптимизации
        fi_series = retrain_model(self.df, step=self.step)
        
        if fi_series.empty or len(fi_series) == 0:
            self.is_finished = True
            self.result_message = "\nХмм... Кажется, я исчерпал все вопросы или не смог сузить круг персонажей."
            return
            
        remaining_characters = self.df['Character'].unique() if 'Character' in self.df.columns else []
        
        question = get_question(fi_series, self.df, remaining_characters)
        if not question:
            self.is_finished = True
            self.result_message = "\nНе осталось вопросов для разделения."
            return
            
        answer = get_user_answer(question)
        self.asked_questions.append(question)
        self.user_answers.append(answer)
        
        self.df = edit_df(self.df, question, answer, step=self.step)
        remaining_characters = self.df['Character'].unique() if 'Character' in self.df.columns else []
        
        print(f"[Шаг {self.step}] Осталось возможных персонажей: {len(remaining_characters)}")
        
        self.check_win_condition(remaining_characters)
        self.step += 1
        
    def check_win_condition(self, remaining_characters):
        if len(remaining_characters) == 0:
            self.is_finished = True
            self.result_message = "\n❌ Что-то пошло не так, в базе не осталось подходящих персонажей под ваши ответы."
            return
            
        if len(remaining_characters) == 1:
            self.is_finished = True
            self.result_message = f"\n🎉 Я уверен! Это персонаж: **{remaining_characters[0]}**"
            return

        if len(remaining_characters) < 10:
            similarities = []
            for char in remaining_characters:
                char_row = self.original_df[self.original_df['Character'] == char]
                if not char_row.empty:
                    char_vec = char_row[self.asked_questions].values[0]
                    sim = get_interest_point(char_vec, self.user_answers)
                    similarities.append((char, sim))
                    
            similarities.sort(key=lambda x: x[1], reverse=True)
            best_char, max_sim = similarities[0]
            
            if len(similarities) > 1:
                second_sim = similarities[1][1]
                delta = max_sim - second_sim
                
                if delta > 0.30 and max_sim >= 0.95:
                    self.is_finished = True
                    self.result_message = f"\n🎉 Я разгадал! Это персонаж: **{best_char}**\n(Сходство: {max_sim:.2f}, Отрыв от остальных: {delta:.2f})"
                    return

        if self.step >= 20:
            self.is_finished = True
            print("\n🤔 Достигнут лимит вопросов (20 шагов). Выбираю лучшего из оставшихся...")
            self._finalize_best_match(remaining_characters)

    def _finalize_best_match(self, remaining_characters):
        similarities = []
        for char in remaining_characters:
            char_row = self.original_df[self.original_df['Character'] == char]
            if not char_row.empty:
                char_vec = char_row[self.asked_questions].values[0]
                sim = get_interest_point(char_vec, self.user_answers)
                similarities.append((char, sim))
        
        if similarities:
            best_char, max_sim = max(similarities, key=lambda x: x[1])
            self.result_message = f"Наиболее подходящий персонаж из оставшихся: **{best_char}** (Сходство: {max_sim:.2f})"
        else:
            self.result_message = "Подходящих персонажей не осталось."

def play_game(original_df):
    print("\n--- Новая игра Акинатора запущена! Загадайте персонажа ---")
    engine = AkinatorEngine(original_df)
    
    while not engine.is_finished:
        engine.play_step()
        
    print(engine.result_message)

if __name__ == '__main__':
    data_file = 'Onepiece_Blich-data_augmented.csv'
    try:
        initial_df = pd.read_csv(data_file)
        print(f"Успешно загружено записей: {len(initial_df)} из {data_file}")
    except FileNotFoundError:
        print(f"Внимание: {data_file} не найден. Пробую загрузить оригинальный...")
        data_file = 'Onepiece_Blich-data.csv'
        try:
            initial_df = pd.read_csv(data_file)
            print(f"Загружен {data_file}. Сначала запустите скрипт аугментации для лучшей работы!")
        except FileNotFoundError:
            print("Ошибка: Файлы датасетов не найдены.")
            exit(1)
            
    while True:
        play_game(initial_df)
        
        restart = input("\nХотите сыграть еще раз? (yes/no): ").strip().lower()
        if restart != 'yes':
            print("Спасибо за игру! До свидания.")
            break
