# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 19:10:27 2023

@author: user
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import seaborn as sns
import pickle 
from catboost import CatBoostClassifier
import io
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import plotly.graph_objs as go 

st.title('Определение музыкального жанра')

dat = pd.read_csv('train.csv')
dat = dat.drop('instance_id', axis=1)
dat.dropna(inplace=True)

# Создаем словарь с медианными значениями duration_ms для каждого жанра
median_duration_ms_by_genre = dat.groupby('music_genre')['duration_ms'].median().to_dict()

# Заменяем отрицательные значения duration_ms на медианные значения по жанрам
dat['duration_ms'] = dat.apply(
    lambda row: median_duration_ms_by_genre[row['music_genre']] if row['duration_ms'] < 0 else row['duration_ms'],
    axis=1)
dat = dat[dat['duration_ms'] <= 420000.0]
durations_ms = np.unique(dat.duration_ms)

music = dat['music_genre'].unique().tolist()
mode = dat['mode'].unique().tolist()
acc = np.unique(dat.acousticness)
#dance = np.unique(dat.danceability)
#ener = np.unique(dat.energy)
instr = np.unique(dat.instrumentalness)
#live = np.unique(dat.liveness)
#loud = np.unique(dat.loudness)
#speech = np.unique(dat.speechiness)
#temp = np.unique(dat.tempo)
val = np.unique(dat.valence)


st.sidebar.markdown('## Данные для прогноза')
select_event_2 = st.sidebar.selectbox('Музыкальный жанр', music)

start_dur, stop_dur = st.sidebar.select_slider('Длительность трека', 
                                         options = durations_ms, 
                                         value=(min(durations_ms), max(durations_ms)))

start_acc, end_acc = st.sidebar.select_slider('Аккустичность трека', options = acc,
                                              value=(min(acc), max(acc)))
#select_event_5 = st.sidebar.select_slider('Танцевальность трека', options = dance)
#select_event_6 = st.sidebar.select_slider('Энергичность трека', options = ener)
start_instr, end_instr = st.sidebar.select_slider('Инструментальность трека', options = instr,
                                                  value=(min(instr), max(instr)))
#select_event_8 = st.sidebar.select_slider('Живучесть трека', options = live)
#select_event_9 = st.sidebar.select_slider('Громкость трека', options = loud)
#select_event_10 = st.sidebar.select_slider('Выразительность трека', options = speech)
#select_event_11 = st.sidebar.select_slider('Темп трека', options = temp)
start_val, end_val = st.sidebar.select_slider('Привлекательность для пользователя', 
                                           options = val,
                                           value=(min(val), max(val)))
select_event_3 = st.sidebar.selectbox('Модальность трека', ['Все'] + mode)


st.header('К какому жанру оносится трек: '+select_event_2)

# Фильтрация данных с учетом модальности
if select_event_3 == 'Все':
    # Если выбран вариант "Все", игнорируйте модальность
    filtered_data = dat[
        (dat['music_genre'] == select_event_2) &
        (dat['duration_ms'] >= start_dur) &
        (dat['duration_ms'] <= stop_dur) &
        (dat['acousticness'] >= start_acc) &
        (dat['acousticness'] <= end_acc) &
        (dat['instrumentalness'] >= start_instr) &
        (dat['instrumentalness'] <= end_instr) &
        (dat['valence'] >= start_val) &
        (dat['valence'] <= end_val)
    ]
else:
    # В противном случае учитывайте выбранную модальность
    filtered_data = dat[
        (dat['music_genre'] == select_event_2) &
        (dat['duration_ms'] >= start_dur) &
        (dat['duration_ms'] <= stop_dur) &
        (dat['acousticness'] >= start_acc) &
        (dat['acousticness'] <= end_acc) &
        (dat['instrumentalness'] >= start_instr) &
        (dat['instrumentalness'] <= end_instr) &
        (dat['valence'] >= start_val) &
        (dat['valence'] <= end_val) &
        (dat['mode'] == select_event_3)
    ]
# Отобразите отфильтрованные данные
st.write(filtered_data.head(10))
       
st.header('Статистические характеристики музыкального жанра: '+select_event_2)
st.write(filtered_data.describe())

corr_matrix = filtered_data.corr()  
fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, 
                                y=corr_matrix.columns))
st.header('Интерактивная тепловая карта для корреляции')
st.plotly_chart(fig)

fig = px.scatter_matrix(filtered_data, dimensions=["acousticness", "instrumentalness", 
                                                   "loudness", "energy"], color="mode")
st.header('Матрица диаграмм рассеяния')

fig.update_layout(
    width=800,  # Укажите ширину
    height=600  # Укажите высоту
)
st.plotly_chart(fig)

# Загрузите encoder из файла
with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)
    
# Загрузите вашу модель до основного кода Streamlit
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)     
               
# Определите функцию для выполнения предсказаний
def make_prediction(model, data):
    # Ваш код для выполнения предсказаний с использованием модели
    prediction = model.predict(data)
    return prediction

# Определите функцию для предобработки данных (ваша функция pres)
def pres(data):

  columns_to_drop = ['track_name', 'instance_id', 'obtained_date']
  data.drop(columns=columns_to_drop, axis=1, inplace=True)

  numeric_features = data.select_dtypes(include=['number'])
  categorical_features = data.select_dtypes(include=['object'])

  numeric_imputer = IterativeImputer(max_iter=10, random_state=0)
  categorical_imputer = SimpleImputer(strategy="most_frequent")

  numeric_filled = numeric_imputer.fit_transform(numeric_features)
  categorical_filled = categorical_imputer.fit_transform(categorical_features)

  numeric_filled_df = pd.DataFrame(numeric_filled, columns=numeric_features.columns)
  categorical_filled_df = pd.DataFrame(categorical_filled, columns=categorical_features.columns)

  data = pd.concat([numeric_filled_df, categorical_filled_df], axis=1)

  data = pd.get_dummies(data, columns=categorical_features.columns)

  return data
           
# Функция для создания ссылки для скачивания CSV файла
def get_csv_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Преобразуйте CSV в кодировку base64
    href = f'<a href="data:file/csv;base64,{b64}" download="results.csv">Скачать CSV файл</a>'
    return href

uploaded_files = st.file_uploader('Добовить файл CSV', accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)

    if st.button('Хочу ПРОГНОЗ'):
        # Вызов функции pres для предобработки данных
        data = pres(pd.read_csv(io.BytesIO(bytes_data)))  # Используйте io.BytesIO для чтения байтовых данных
        # Вызов функции make_prediction для выполнения прогнозов
        prediction = make_prediction(model, data)
        pred_decoded = encoder.inverse_transform(prediction)

        # После выполнения предсказаний
        predicted_df = pd.DataFrame({'Prediction': pred_decoded.flatten()})  # Преобразуйте в одномерный массив
        original_df = pd.read_csv(io.BytesIO(bytes_data))  # Загрузка исходных данных

        # Если нет уникального идентификатора, нужно убедиться, что данные в одном и том же порядке
        result_df = pd.concat([original_df['track_name'], predicted_df], axis=1)

        st.write("Результаты предсказания:")
        st.write(result_df.head())

        if st.button('Скачать результаты в CSV'):
            result_df.to_csv('results.csv', index=False)
            st.success("Результаты успешно сохранены в файл results.csv. Нажмите на ссылку ниже, чтобы скачать.")
            st.markdown(get_csv_download_link(result_df), unsafe_allow_html=True)









            
            




