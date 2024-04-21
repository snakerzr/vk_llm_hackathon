import streamlit as st
import pandas as pd
from inference import infer
from llm import pipeline

def display_results(title, result, is_list=False):
    st.write(title)
    if is_list:
        if result:
            # Преобразование списка в строку с элементами, разделенными запятыми
            formatted_tags = "Тэги: " + ", ".join(result)
            st.text(formatted_tags)
        else:
            st.text("Нет подходящих тэгов")
    else:
        # Проверяем, есть ли в DataFrame достаточно столбцов
        if result.shape[1] >= 2:
            # Тэги из двух последних столбцов
            last_two_columns = result.iloc[:, -2:]
            tags = pd.concat([last_two_columns[col].dropna() for col in last_two_columns]).unique()
            if len(tags) > 0:
                formatted_tags = "Тэги: " + ", ".join(tags)
                st.text(formatted_tags)
            else:
                st.text("Нет подходящих тэгов")

            # Подтэги из второго столбца
            sub_tags = result.iloc[:, 1].dropna().unique()  # Второй столбец для подтэгов
            if len(sub_tags) > 0:
                formatted_sub_tags = "Подтэги: " + ", ".join(sub_tags)
                st.text(formatted_sub_tags)
            else:
                st.text("Нет подходящих подтэгов")
        else:
            st.text("Недостаточно данных для отображения тэгов")

st.title("Загрузка и обработка данных")

text_input = st.text_area("Введите текст здесь")

if text_input:
    st.write("Введенный текст:")
    st.write(text_input)

    if st.button("Применить модель inference"):
        result = infer([text_input])  # предполагается, что infer возвращает список
        display_results("Результат работы модели inference:", result, is_list=True)

    if st.button("Применить модель llm"):
        small_result, full_result = pipeline([text_input])  # предполагается, что pipeline возвращает два DataFrame
        display_results("Результат работы модели llm:", small_result, is_list=False)
        # Полный результат можно вывести при необходимости
        # display_results("Полный результат:", full_result)

st.caption("Введите текст для обработки.")
