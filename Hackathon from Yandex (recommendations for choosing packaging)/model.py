import pandas as pd
import numpy as np
import pickle
import joblib


def predict(data):
    #Загрузка датасета с объемом упаковок:
    carton = pd.read_csv('carton.csv')
    carton['carton_volume'] = carton['LENGTH']*carton['WIDTH']*carton['HEIGHT']

    # Загрузка модели 
    model_file = "file_item.pcl"
    with open(model_file, "rb") as file:
        item_model = pickle.load(file)

    # Загрузка энкодера
    encoder = 'encoder.pkl'
    with open(encoder, "rb") as file:
        encoder = joblib.load(file)

    # Преобразование словарей в датафреймы
    items = data.get("items", [])
    if not items:
        return 'пустой список items'

    df = pd.DataFrame(items)
    df["orderId"] = data.get("orderId", "default_orderId")
    df = df[["orderId", "sku", "count", "size1", "size2", "size3", "weight", "type", "description"]]
    numeric_columns = ["size1", "size2", "size3", "weight"]
    df[numeric_columns] = df[numeric_columns].astype(float)

    # Создаем новые признаки:
    df['volume_sku'] = df['size1']*df['size2']*df['size3']
    df['item_count'] = df.groupby('orderId')['sku'].transform('nunique')
    df['total_volume_sku'] = df['volume_sku']*df['count']
    df['total_wght'] = df['weight']*df['count']
    
    # Добавляем столбцы с признаками из 'description'
    df[['упаковка в пленку', 'не требует упаковки', 'сыпучее', 'пачкает', 'готовое блюда', 'хрупкое', 'интим товар',
        'цена высокая', 'пахучий', 'мнется', 'цена низкая', 'опасный', 'электроника',
        'цена средняя', 'товары для авто и мототехники', 'впитывает запах', 'продукты питания',
        'средства личной гигиены', 'товары для животных', 
        'химикаты', 'аптека', 'одежда, обувь, аксессуары', 'меркурий', 'прочие']] = df['description'].apply(
    lambda x: pd.Series([1 if 'упаковка в пленку' in x else 0,
                         1 if 'не требует упаковки' in x else 0,
                         1 if 'сыпучее' in x else 0,
                         1 if 'пачкает' in x else 0,
                         1 if 'готовое блюда' in x else 0,
                         1 if 'хрупкое' in x else 0,
                         1 if 'интим товар' in x else 0,
                         1 if 'цена высокая' in x else 0,
                         1 if 'пахучий' in x else 0,
                         1 if 'мнется' in x else 0,
                         1 if 'цена низкая' in x else 0,
                         1 if 'опасный' in x else 0,
                         1 if 'электроника' in x else 0,
                         1 if 'цена средняя' in x else 0,
                         1 if 'товары для авто и мототехники' in x else 0,
                         1 if 'впитывает запах' in x else 0, 
                         1 if 'продукты питания' in x else 0,
                         1 if 'средства личной гигиены' in x else 0,
                         1 if 'товары для животных' in x else 0,
                         1 if 'химикаты' in x else 0,
                         1 if 'аптека' in x else 0,
                         1 if 'одежда, обувь, аксессуары' in x else 0,
                         1 if 'меркурий' in x else 0,
                         1 if not any(val in x for val in ['упаковка в пленку', 'не требует упаковки', 'сыпучее', 'пачкает', 'готовое блюда', 
                                                           'хрупкое', 'интим товар','цена высокая', 'пахучий', 'мнется', 'цена низкая', 'опасный', 
                                                           'электроника', 'цена средняя', 'товары для авто и мототехники', 'впитывает запах', 
                                                           'продукты питания','средства личной гигиены', 'товары для животных', 
                                                           'химикаты', 'аптека', 'одежда, обувь, аксессуары', 'меркурий']) else 0])) 
    
    # Удаляем значения в виде списков:
    df.drop(['type', 'description'], axis=1, inplace=True)
    
    # РАСЧЕТ ИНДИВИДУАЛЬНОЙ УПАКОВКИ 
    # Создаем пустой список для хранения выбранных упаковок
    selected_packs = []

    # Итерируемся по строкам датасета
    for index, row in df.iterrows():
        total_volume = row['total_volume_sku']
        selected_pack = None
        min_volume_diff = float('inf')
    
        for pack_index, pack_row in carton.iterrows():
            pack_volume = pack_row['carton_volume']
            volume_diff = pack_volume - total_volume
        
            if volume_diff >= 0 and volume_diff < min_volume_diff:
                min_volume_diff = volume_diff
                selected_pack = pack_row['CARTONTYPE']
    
        selected_packs.append(selected_pack)

    # Добавляем столбец с выбранными упаковками в датасет
    df['box_name_sku'] = selected_packs

    # Создаем функцию для фильтрации значений 'NONPACK' и 'STRETCH'
    def filter_box_name(row):
        if (row['интим товар'] == 0) and (row['химикаты'] == 0) and (row['хрупкое'] == 0) and (row['не требует упаковки'] == 1):
            return 'NONPACK'
        elif (row['интим товар'] == 0) and (row['химикаты'] == 0) and (row['хрупкое'] == 0) and (row['упаковка в пленку'] == 1):
            return 'STRETCH'
        else:
            return row['box_name_sku']

    # Применяем функцию к столбцу 'box_name'
    df['box_name_sku'] = df.apply(filter_box_name, axis=1)
    
    # Делаем агрегацию по уникальному заказу
    df_agg = df.groupby('orderId').agg({
    'weight': ['min', 'max'],
    'size1': ['mean', 'min', 'max'],
    'size2': ['mean', 'min', 'max'],
    'size3': ['mean', 'min', 'max'],
    'total_volume_sku': ['mean', 'min', 'max', 'sum'],
    'total_wght':  'max',	
    'volume_sku': ['mean', 'min', 'max'],
    'count': 'mean',
    'item_count': 'mean',
    'упаковка в пленку': 'mean',
    'не требует упаковки': 'mean',
    'хрупкое': 'mean',
    'цена низкая': 'mean',
    'продукты питания': 'mean',
    'химикаты': 'mean'
    })

    df_agg.columns = ['_'.join(col) for col in df_agg.columns]
    df_agg = df_agg.reset_index()
  
    # Удаляем лишние столбцы
    df_agg.drop('orderId', axis=1, inplace=True)
   
    # ПРЕДСКАЗАНИЕ МОДЕЛИ
    y_pred = item_model.predict_proba(df_agg)

    # Раскодировка предсказания:
    predicted_labels = np.argmax(y_pred, axis=1)
    y_pred_norm = encoder.inverse_transform(predicted_labels.reshape(-1, 1))
    df_agg['box_name_model'] = y_pred_norm 

    # ОПРЕДЕЛЕНИЕ ОПТИМАЛЬНОЙ УПАКОВКИ ДЛЯ ЗАКАЗА
    # Создаем пустой список для хранения выбранных упаковок
    selected_packs = []

    #  Итерируемся по строкам первого датасета
    for index, row in df_agg.iterrows():
        # Получаем объем товара для текущей строки
        total_volume = row['total_volume_sku_sum']
    
        # Инициализируем переменные для выбранной упаковки и ее разницы в объеме
        selected_pack = None
        min_volume_diff = float('inf')
    
        # Итерируемся по упаковкам из второго датасета
        for pack_index, pack_row in carton.iterrows():
            # Получаем объем упаковки
            pack_volume = pack_row['carton_volume']
        
            # Вычисляем разницу в объеме между упаковкой и товаром
            volume_diff = pack_volume - total_volume
        
            # Проверяем, является ли текущая упаковка наиболее оптимальной
            if volume_diff >= 0 and volume_diff < min_volume_diff:
                min_volume_diff = volume_diff
                selected_pack = pack_row['CARTONTYPE']
    
        # Добавляем выбранную упаковку в список
        selected_packs.append(selected_pack)

    # Добавляем столбец с выбранными упаковками в первый датасет
    df_agg['box_name_volume'] = selected_packs
    
    # Результат
    output = ""
    output += f"рекомендованные упаковки: {df_agg.box_name_model.values}, {df_agg.box_name_volume.values}, "
    output += f"оптимальный объем: {df_agg.box_name_volume.values}, "
    for index, row in df.iterrows():
        output += f"рекомендации по индивидуальной упаковке товаров: {row['sku']} - {row['box_name_sku']}, "
    
    return output