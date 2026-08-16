import pandas as pd
import numpy as np

def augment_dataset(input_file='Onepiece_Blich-data.csv', 
                    output_file='Onepiece_Blich_augmented.csv', 
                    n_variations=2, 
                    random_state=42):
    """
    Проводит аугментацию датасета акинатора с заданными вероятностями:
    - 40% оставить оригинальное значение (-1 / 1)
    - 40% заменить на вероятностное (-0.7 / 0.7)
    - 20% заменить на 0 (не знаю)
    """
    np.random.seed(random_state)
    
    # Загружаем исходную таблицу
    print(f"Загрузка файла {input_file}...")
    df = pd.read_csv(input_file)
    
    augmented_rows = []
    feature_cols = df.columns[1:] # Все колонки с признаками, кроме имени персонажа
    
    print(f"Исходное количество персонажей: {len(df)}")
    print("Выполнение аугментации...")
    
    for _, row in df.iterrows():
        # 1. Обязательно сохраняем оригинальную строку
        augmented_rows.append(row.copy())
        
        # 2. Создаем n_variations аугментированных копий для каждого персонажа
        for _ in range(n_variations):
            new_row = row.copy()
            for col in feature_cols:
                val = row[col]
                # Применяем аугментацию только к определенным ответам (-1 и 1)
                if val in [1, -1]:
                    # Случайный выбор действия по заданным вероятностям:
                    # 40% - keep, 40% - soft (0.7 / -0.7), 20% - zero (0)
                    choice = np.random.choice(['keep', 'soft', 'zero'], p=[0.4, 0.4, 0.2])
                    
                    if choice == 'keep':
                        new_row[col] = val
                    elif choice == 'soft':
                        new_row[col] = 0.7 if val == 1 else -0.7
                    elif choice == 'zero':
                        new_row[col] = 0.0
            
            augmented_rows.append(new_row)
            
    # Собираем новый датафрейм
    augmented_df = pd.DataFrame(augmented_rows)
    
    # Сохраняем в новый CSV-файл
    augmented_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Аугментация завершена! Новый файл сохранен как: {output_file}")
    print(f"Итоговое количество строк (образцов) в датасете: {len(augmented_df)}")
    
    return augmented_df

if __name__ == '__main__':
    augment_dataset()