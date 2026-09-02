import sys
import os
import random
import pandas as pd
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QStackedWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtGui import QPixmap, QMovie
from PyQt6 import uic

from utility import retrain_model, get_question, edit_df, get_interest_point

DEPENDENCY_RULES = {
    "Съел ли персонаж дьявольский фрукт?": [
        "Относится ли дьявольский фрукт к типу Парамеция?",
        "Относится ли дьявольский фрукт к типу Логия?",
        "Относится ли дьявольский фрукт к типу Обычный Зоан?",
        "Относится ли дьявольский фрукт к типу Древний Зоан?",
        "Относится ли дьявольский фрукт к типу Мифический Зоан?"
    ],
    "Является ли он синигами (проводником душ)?": [
        "Достиг ли этот персонаж банкая?",
        "Является ли он материализованным духом занпакто?"
    ],
    "У вашего персонажа есть врожденная глазная техника (Додзюцу)?": [
        "У вашего персонажа имеется легендарный Шаринган?",
        "У вашего персонажа пробужден Риннеган?",
        "У вашего персонажа есть клановый Бьякуган?"
    ],
    "Состоит в официальной организации(служитель закона)?": [
        "Был ли персонаж Ситибукаем (Морским Лордом)?",
        "Был ли персонаж офицером-агентом (Барок Воркс)?",
        "Был ли персонаж агентом Фронтира (Барок Воркс)?"
    ],
    "Является ли он преступником(пиратом разбойником)?": [
        "Является ли персонаж Синъути (Звездным исполнителем Пиратов Зверей)?",
        "Является ли персонаж Старшей Звездой (Бедствием Пиратов Зверей)?",
        "Является ли персонаж Генералом-Сладостью (Пиратов Большой Мамочки)?"
    ]
}

class AkinatorEngine:
    def __init__(self, original_df):
        self.original_df = original_df.copy()
        self.reset_game()
        
    def reset_game(self):
        self.df = self.original_df.copy(deep=True)
        self.step = 1
        self.asked_questions = []
        self.user_answers = []
        self.is_finished = False
        self.result_message = ""
        self.guessed_character = None
        
        self.current_core_question = None
        self.current_display_question = None
        self.is_verification = False
        self.history = [] 
        self.emotion = "standart" # Стартовая эмоция

    def generate_next_question(self):
        if self.is_finished:
            return None
            
        fi_series = retrain_model(self.df, step=self.step)
        
        if fi_series.empty or len(fi_series) == 0:
            self.is_finished = True
            self.result_message = "\nХмм... Кажется, я исчерпал все вопросы или не смог сузить круг персонажей."
            return None
            
        remaining_characters = self.df['Character'].unique() if 'Character' in self.df.columns else []
        
        raw_question = get_question(fi_series, self.df, remaining_characters)
        if not raw_question:
            self.is_finished = True
            self.result_message = "\nНе осталось вопросов для разделения."
            return None

        self.is_verification = raw_question.startswith("[Уточнение]")
        self.current_display_question = raw_question
        self.current_core_question = raw_question.replace("[Уточнение] Вы уверены в ответе на вопрос: ", "") if self.is_verification else raw_question
            
        return self.current_display_question

    def process_answer(self, answer: float):
        if self.is_finished or not self.current_core_question:
            return
            
        prev_char_count = len(self.df['Character'].unique()) if 'Character' in self.df.columns else 0

        # Используем deepcopy, чтобы при откате назад данные гарантированно не были изменены по ссылке
        state_snapshot = (
            self.df.copy(deep=True), 
            self.step, 
            list(self.asked_questions), 
            list(self.user_answers),
            self.current_core_question, 
            self.current_display_question, 
            self.is_verification,
            self.is_finished, 
            self.result_message, 
            self.guessed_character,
            self.emotion
        )
        self.history.append(state_snapshot)

        if self.is_verification and self.current_core_question in self.asked_questions:
            idx = self.asked_questions.index(self.current_core_question)
            self.user_answers[idx] = answer
        else:
            self.asked_questions.append(self.current_core_question)
            self.user_answers.append(answer)
        
        if answer <= -0.7 and self.current_core_question in DEPENDENCY_RULES:
            cols_to_drop = [col for col in DEPENDENCY_RULES[self.current_core_question] if col in self.df.columns]
            if cols_to_drop:
                self.df = self.df.drop(columns=cols_to_drop)
                print(f"[Оптимизация] Автоматически исключено нерелевантных вопросов: {len(cols_to_drop)}")
                
        self.df = edit_df(self.df, self.current_core_question, answer, step=self.step)
        remaining_characters = self.df['Character'].unique() if 'Character' in self.df.columns else []
        
        print(f"[Шаг {self.step}] Осталось возможных персонажей: {len(remaining_characters)}")
        
        self.check_win_condition(remaining_characters, prev_char_count)
        self.step += 1

    def undo(self) -> bool:  #все ещё ужасно работает
        if not self.history:
            return False 
            
        state = self.history.pop()
        self.df = state[0].copy(deep=True)
        self.step = state[1]
        self.asked_questions = list(state[2])
        self.user_answers = list(state[3])
        self.current_core_question = state[4]
        self.current_display_question = state[5]
        self.is_verification = state[6]
        self.is_finished = state[7]
        self.result_message = state[8]
        self.guessed_character = state[9]
        self.emotion = state[10]
        
        return True

    def check_win_condition(self, remaining_characters, prev_char_count):
        curr_char_count = len(remaining_characters)
        diff = prev_char_count - curr_char_count
        
        # Логика смены эмоций на основе того, как сильно сократился список персонажей
        if diff <= 2: 
            self.emotion = "hard"
        elif diff >= 50 or (prev_char_count > 0 and curr_char_count <= prev_char_count * 0.6): # Если убрали 50+ или больше 40%
            self.emotion = "happy"
        else:
            self.emotion = random.choice(["standart", "think"])

        if len(remaining_characters) == 0:
            self.is_finished = True
            self.result_message = "\n❌ Что-то пошло не так, в базе не осталось подходящих персонажей под ваши ответы."
            return
            
        if len(remaining_characters) == 1:
            self.is_finished = True
            self.guessed_character = remaining_characters[0]
            self.result_message = f"\n🎉 Я уверен! Это персонаж: **{self.guessed_character}**"
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
                
                if delta > 0.45 and max_sim >= 0.95:
                    self.is_finished = True
                    self.guessed_character = best_char
                    self.result_message = f"\n🎉 Я разгадал! Это персонаж: **{self.guessed_character}**\n(Сходство: {max_sim:.2f}, Отрыв от остальных: {delta:.2f})"
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
            self.guessed_character = best_char
            self.result_message = f"Наиболее подходящий персонаж из оставшихся: **{self.guessed_character}** (Сходство: {max_sim:.2f})"
        else:
            self.result_message = "Подходящих персонажей не осталось."


class StartPage(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        
        layout = QVBoxLayout(self)
        label = QLabel("Добро пожаловать в Акинатор!\n(Это временная StartPage)", self)
        btn = QPushButton("Начать игру", self)
        
        btn.clicked.connect(self.start_new_game)
        layout.addWidget(label)
        layout.addWidget(btn)

    def start_new_game(self):
        self.main_window.current_df = self.main_window.global_df.copy()
        self.main_window.show_game_page()


class GamePage(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        ui_path = os.path.join(os.path.dirname(__file__), "gamepage.ui")
        uic.loadUi(ui_path, self)

        self.movie = QMovie("photo/ezgif-83f587d9ce2dc45e.gif")
        self.BACKGROUND.setScaledContents(True)
        self.BACKGROUND.setMovie(self.movie)
        self.movie.start()

        me_pixmap = QPixmap("photo/standart.png")
        self.WE.setScaledContents(True)
        self.WE.setPixmap(me_pixmap)

        self.YES.clicked.connect(lambda: self.handle_answer(1.0))
        self.MBYES.clicked.connect(lambda: self.handle_answer(0.7))
        self.IDK.clicked.connect(lambda: self.handle_answer(0.0))
        self.MBNO.clicked.connect(lambda: self.handle_answer(-0.7))
        self.NO.clicked.connect(lambda: self.handle_answer(-1.0))
        self.back.clicked.connect(self.on_back_clicked)

        self.engine = None

    def handle_answer(self, value):
        if not self.engine or self.engine.is_finished:
            return 
        self.engine.process_answer(value)
        self.update_ui()

    def update_ui(self, use_existing_question=False):
        if self.engine.is_finished:
            self._handle_finish()
            return

        if not use_existing_question:
            question = self.engine.generate_next_question()
        
        if self.engine.is_finished:
            self._handle_finish()
        else:
            display_text = f"Вопрос {self.engine.step}:\n{self.engine.current_display_question}"
            self.QUESTION.setPlainText(display_text)
            
            # Обновление картинки-реакции на основе текущей эмоции
            if hasattr(self.engine, 'emotion'):
                emotion_pixmap = QPixmap(f"photo/{self.engine.emotion}.png")
                self.WE.setPixmap(emotion_pixmap)

            self._set_buttons_enabled(True)

    def _handle_finish(self):
        if self.engine.guessed_character:
            self.main_window.show_end_page(self.engine.guessed_character, self.engine.step)
        else:
            self.QUESTION.setPlainText(self.engine.result_message.strip())
            self._set_buttons_enabled(False)

    def _set_buttons_enabled(self, state: bool):
        self.YES.setEnabled(state)
        self.MBYES.setEnabled(state)
        self.IDK.setEnabled(state)
        self.MBNO.setEnabled(state)
        self.NO.setEnabled(state)

    def on_back_clicked(self):
        if self.engine.undo():
            self.update_ui(use_existing_question=True)


class EndPage(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        ui_path = os.path.join(os.path.dirname(__file__), "endpage.ui")
        uic.loadUi(ui_path, self)

        self.movie = QMovie("photo/ezgif-83f587d9ce2dc45e.gif")
        self.BACKGROUND.setScaledContents(True)
        self.BACKGROUND.setMovie(self.movie)
        self.movie.start()

        winner_pixmap = QPixmap("photo/Naruto_newshot.png")
        self.WINNER.setScaledContents(True)
        self.WINNER.setPixmap(winner_pixmap)

        self.pushButton.clicked.connect(self.on_yes_clicked)
        self.pushButton_2.clicked.connect(self.on_no_clicked)

        self.guessed_character = ""
        self.step = 0
        self.state = 0

    def setup_page(self, character, step):
        self.guessed_character = character
        self.step = step
        self.state = 0
        self.textEdit.setPlainText(f"Угадан персонаж: {self.guessed_character}.\nЭто ваш персонаж?")

    def on_yes_clicked(self):
        if self.state == 0:
            with open("history.txt", "a", encoding="utf-8") as f:
                f.write(f"Угадан персонаж: {self.guessed_character} Ход: {self.step}\n")
            self.main_window.show_start_page()
            
        elif self.state == 1:
            self.main_window.restart_game_without(self.guessed_character)

    def on_no_clicked(self):
        if self.state == 0:
            self.state = 1
            self.textEdit.setPlainText("Желаете ли вы продолжить поиск?")
            
        elif self.state == 1:
            self.main_window.show_start_page()


class MainWindow(QMainWindow):
    def __init__(self, original_df, parent=None):
        super().__init__(parent)
        self.global_df = original_df.copy()
        self.current_df = self.global_df.copy()

        self.stackedWidget = QStackedWidget(self)
        self.setCentralWidget(self.stackedWidget)

        self.setFixedSize(500, 640)
        self.setWindowTitle("Big Three Akinator")

        self.start_page = StartPage(self)
        self.game_page = GamePage(self)
        self.end_page = EndPage(self)

        self.stackedWidget.addWidget(self.start_page)
        self.stackedWidget.addWidget(self.game_page)
        self.stackedWidget.addWidget(self.end_page)

        self.stackedWidget.setCurrentWidget(self.start_page)

    def show_start_page(self):
        self.stackedWidget.setCurrentWidget(self.start_page)

    def show_game_page(self):
        self.game_page.engine = AkinatorEngine(self.current_df)
        self.game_page.update_ui()
        self.stackedWidget.setCurrentWidget(self.game_page)

    def show_end_page(self, character, step):
        self.end_page.setup_page(character, step)
        self.stackedWidget.setCurrentWidget(self.end_page)

    def restart_game_without(self, character):
        self.current_df = self.current_df[self.current_df['Character'] != character]
        self.show_game_page()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    data_file = 'Naruto-Blich-Onepiece-data.csv'
    try:
        initial_df = pd.read_csv(data_file)
        print(f"Успешно загружено записей: {len(initial_df)}")
    except FileNotFoundError:
        print(f"Ошибка: Файл {data_file} не найден. Проверьте путь.")
        initial_df = pd.DataFrame() 

    window = MainWindow(original_df=initial_df)
    window.show()
    sys.exit(app.exec())