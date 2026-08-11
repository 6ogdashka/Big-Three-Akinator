import pandas as pd
from utility import retrain_model, get_question, get_user_answer, edit_df

def play_game(original_df):
    df = original_df.copy()
    step = 1
    
    print("\n--- Новая игра Акинатора запущена! Загадайте персонажа ---")
    
    while True:
        fi_series = retrain_model(df)
        
        if fi_series.empty or len(fi_series) == 0:
            print("\nХмм... Кажется, я исчерпал все вопросы или не смог сузить круг персонажей.")
            break
            
        question = get_question(fi_series)
        
        answer = get_user_answer(question)
        
        df = edit_df(df, question, answer, step=step)
        
        remaining_characters = df['Character'].unique() if 'Character' in df.columns else []
        
        print(f"[Шаг {step}] Осталось возможных персонажей: {len(remaining_characters)}")
        
        if len(remaining_characters) == 1:
            guessed_char = remaining_characters[0]
            print(f"\n🎉 Я угадал! Это персонаж: **{guessed_char}**")
            break
        elif len(remaining_characters) == 0:
            print("\n❌ Что-то пошло не так, в базе не осталось подходящих персонажей под ваши ответы.")
            break
        elif step >= 25:
            print(f"\n🤔 Похоже, у меня слишком много вариантов ({len(remaining_characters)}), но лимит вопросов исчерпан.")
            if len(remaining_characters) > 0:
                print(f"Возможные варианты: {list(remaining_characters[:5])}")
            break
            
        step += 1

if __name__ == '__main__':
    data_file = 'Onepiece_Blich_augmented.csv'
    try:
        initial_df = pd.read_csv(data_file)
        print(f"Успешно загружено записей: {len(initial_df)} из {data_file}")
    except FileNotFoundError:
        print(f"Ошибка: Файл {data_file} не найден. Сначала запустите скрипт аугментации.")
        exit(1)
        
    while True:
        play_game(initial_df)
        
        restart = input("\nХотите сыграть еще раз? (yes/no): ").strip().lower()
        if restart != 'yes':
            print("Спасибо за игру! До свидания.")
            break