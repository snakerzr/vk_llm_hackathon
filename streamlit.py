from main import pipeline
import streamlit as st
import pandas as pd


st.title("Загрузка и обработка данных")

uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")
text_input = st.text_area("Или введите текст здесь")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if "articles" in data.columns or "Article Text" in data.columns:
        st.write("CSV файл успешно загружен и содержит необходимый столбец 'articles'.")
        try:
            small_result, full_result = pipeline(data["articles"].tolist())
        except KeyError as k_e:
            small_result, full_result = pipeline(data["Article Text"].tolist())
        st.write("Краткий результат:")
        st.dataframe(small_result)
        st.write("Полный результат:")
        st.dataframe(full_result)
    else:
        st.error("В файле отсутствует необходимый столбец 'articles'.")
elif text_input:
    st.write("Введен текст:")
    st.write(text_input)
    small_result, full_result = pipeline([text_input])

    st.write("Главный тэг вар1:")
    st.header(small_result["tag_with_tags"].values[0])

    st.write("Главный тэг вар2:")
    st.header(small_result["tag_with_text"].values[0])

    st.write("Уточняющие тэги:")
    st.text("\n".join(small_result["tags"].astype(str)))
    st.write("Полный результат:")
    st.dataframe(full_result)

st.caption("Загрузите файл или введите текст для обработки.")
