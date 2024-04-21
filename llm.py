import os

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    UserMessage,
)
from mistral_common.tokens.instruct.normalize import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

_ = load_dotenv(find_dotenv())

LLM_ENDPOINT = os.environ["LLM_ENDPOINT"]

advanced_tag_list = {
    "Политика": "Политика - власть, стратегия, дипломатия",
    "Экономика": "Экономика - финансы, рынки, инвестиции",
    "Общество": "Общество - население, культура, социальные вопросы",
    "Закон и право": "Закон и право - юстиция, законодательство, правопорядок",
    "Кино": "Кино - фильмы, режиссура, актеры",
    "Телевидение": "Телевидение - шоу, программы, каналы",
    "Персоны": "Персоны - известные люди, биографии, влияние",
    "События": "События - происшествия, мероприятия, анонсы",
    "Бренды": "Бренды - компании, товары, реклама",
    "Наука": "Наука - исследования, открытия, теории",
    "Гаджеты": "Гаджеты - устройства, инновации, электроника",
    "Соцсети": "Соцсети - интернет, общение, платформы",
    "Технологии": "Технологии - прогресс, новшества, приложения",
    "Опросы": "Опросы - исследования, мнения, статистика",
    "Головоломки": "Головоломки - задачи, размышления, интеллект",
    "Дом": "Дом - жилье, интерьер, уют",
    "Транспорт": "Транспорт - движение, транспортные средства, логистика",
    "Погода": "Погода - климат, прогнозы, условия",
    "Рецепты": "Рецепты - кулинария, блюда, ингредиенты",
    "Мода": "Мода - стиль, тренды, одежда",
    "Красота": "Красота - уход, здоровье, внешность",
}


def get_linear_probs(ai_response: AIMessage) -> float:
    if (
        ai_response.response_metadata["logprobs"]["content"][0]["token"].strip()
        == "True"
    ):
        log_prob = ai_response.response_metadata["logprobs"]["content"][0]["logprob"]
        return np.round(np.exp(log_prob), 3)
    else:
        return 0


def tagging(
    news_articles: list[str],
    tags_dict: dict[str, str],
    temperature: float = None,
) -> pd.DataFrame:
    base_url = LLM_ENDPOINT

    if not temperature:
        llm = ChatOpenAI(
            api_key="<key>",
            model="tgi",
            openai_api_base=base_url,
        ).bind(logprobs=True)

    else:
        llm = ChatOpenAI(
            api_key="<key>",
            model="tgi",
            openai_api_base=base_url,
            temperature=temperature,
        ).bind(logprobs=True)

    SYSTEM_PROMPT = "Ты новостной эксперт высшего класса. И можешь определить тэг новости без ошибок. Ты отвечаешь только True или False и ничего больше."

    INSTRUCTION_TEMPLATE = """Верни True, если статья принадлежит тэгу. Или False, если не принадлежит. Больше ничего не пиши.

    > СТАТЬЯ
    >>>>>
    {article}
    >>>>>

    > ТЭГ
    >>>>>
    {tag}
    >>>>>"""

    prompt = ChatPromptTemplate.from_template(INSTRUCTION_TEMPLATE)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", INSTRUCTION_TEMPLATE),
        ]
    )

    chain = prompt | llm

    tokenizer_v3 = MistralTokenizer.v3()
    result = []

    for article in news_articles:
        intermediate_result = {}
        intermediate_result["article"] = article
        for tag in tags_dict:
            # Считаем токены в инпуте
            user_message = INSTRUCTION_TEMPLATE.format(
                article=article, tag=tags_dict[tag]
            )
            messages = ChatCompletionRequest(
                messages=[
                    AssistantMessage(content=SYSTEM_PROMPT),
                    UserMessage(content=user_message),
                ]
            )
            token_n = len(tokenizer_v3.encode_chat_completion(messages).tokens)

            if token_n - 4000 > 0:
                article = article[:-1000]

            answer = chain.invoke({"article": article, "tag": tags_dict[tag]})

            if answer.content == "True":
                prob = get_linear_probs(answer)
                intermediate_result[tag] = prob
            else:
                intermediate_result[tag] = 0

        result.append(intermediate_result)

    df = pd.DataFrame(result)

    df["highest_tag"] = df.drop("article", axis=1).idxmax(axis=1)

    return df


async def atagging(
    news_articles: list[str],
    tags_dict: dict[str, str],
    temperature: float = None,
) -> pd.DataFrame:
    base_url = LLM_ENDPOINT

    if not temperature:
        llm = ChatOpenAI(
            api_key="<key>",
            model="tgi",
            openai_api_base=base_url,
        ).bind(logprobs=True)

    else:
        llm = ChatOpenAI(
            api_key="<key>",
            model="tgi",
            openai_api_base=base_url,
            temperature=temperature,
        ).bind(logprobs=True)

    SYSTEM_PROMPT = "Ты новостной эксперт высшего класса. И можешь определить тэг новости без ошибок. Ты отвечаешь только True или False и ничего больше."

    INSTRUCTION_TEMPLATE = """Верни True, если статья принадлежит тэгу. Или False, если не принадлежит. Больше ничего не пиши.

    > СТАТЬЯ
    >>>>>
    {article}
    >>>>>

    > ТЭГ
    >>>>>
    {tag}
    >>>>>"""

    prompt = ChatPromptTemplate.from_template(INSTRUCTION_TEMPLATE)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", INSTRUCTION_TEMPLATE),
        ]
    )

    chain = prompt | llm

    tokenizer_v3 = MistralTokenizer.v3()
    result = []

    async_requests = []
    for article in news_articles:
        for tag in tags_dict:
            user_message = INSTRUCTION_TEMPLATE.format(
                article=article, tag=tags_dict[tag]
            )
            messages = ChatCompletionRequest(
                messages=[
                    AssistantMessage(content=SYSTEM_PROMPT),
                    UserMessage(content=user_message),
                ]
            )
            token_n = len(tokenizer_v3.encode_chat_completion(messages).tokens)
            if token_n - 4000 > 0:
                article = article[:-1000]
            async_requests.append({"article": article, "tag": tags_dict[tag]})
    answers = await chain.abatch(async_requests)

    answers = [get_linear_probs(answer) for answer in answers]

    result = pd.DataFrame(async_requests)
    result["tag"] = result["tag"].map({tags_dict[x]: x for x in tags_dict})
    result["probs"] = answers
    result = result.pivot(index="article", columns="tag", values="probs").reset_index()

    result["highest_tag"] = result.drop("article", axis=1).idxmax(axis=1)

    return result


def tags_creation(news_articles: list[str]) -> pd.DataFrame:
    base_url = LLM_ENDPOINT

    llm = ChatOpenAI(
        api_key="<key>", model="tgi", openai_api_base=base_url, temperature=0.1
    )

    SYSTEM_PROMPT = "Ты новостной эксперт высшего класса. Ты безупречно определяешь тэги новостной статьи."

    INSTRUCTION_TEMPLATE = """Напиши 10 тэгов для статьи, который превосходно отражают суть статьи и по которым сразу становится понятно общее содержание статьи. Напиши только тэги и больше ничего. Напиши тэги через запятую. Каждый тэг одно слово.

    > СТАТЬЯ
    >>>>>
    {article}
    >>>>>
    """

    prompt = ChatPromptTemplate.from_template(INSTRUCTION_TEMPLATE)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", INSTRUCTION_TEMPLATE),
        ]
    )

    chain = prompt | llm

    tokenizer_v3 = MistralTokenizer.v3()
    result = []

    for article in news_articles:
        intermediate_result = {}
        intermediate_result["article"] = article
        # Считаем токены в инпуте
        user_message = INSTRUCTION_TEMPLATE.format(article=article)
        messages = ChatCompletionRequest(
            messages=[
                AssistantMessage(content=SYSTEM_PROMPT),
                UserMessage(content=user_message),
            ]
        )
        token_n = len(tokenizer_v3.encode_chat_completion(messages).tokens)

        if token_n - 4000 > 0:
            article = article[:-1000]

        answer = chain.invoke({"article": article})

        intermediate_result["tags"] = answer.content

        result.append(intermediate_result)

    return pd.DataFrame(result)


def pipeline(news_articles: list[str]) -> pd.DataFrame:
    # get tags
    created_tags = tags_creation(news_articles)

    # classify tags
    # Классификация тэгов происходит с температурой 0.15
    classified_tags = tagging(created_tags["tags"].tolist(), advanced_tag_list, 0.15)

    classified_tags.columns = ["tags"] + classified_tags.columns[1:].tolist()

    result = created_tags.merge(classified_tags, on="tags")

    # classify raw text with advanced tags
    raw_text_tags = tagging(news_articles, advanced_tag_list, 0.15)

    result = result.merge(raw_text_tags, on="article", suffixes=["_tags", "_raw"])

    small_result = result[["article", "tags", "highest_tag_tags", "highest_tag_raw"]]
    small_result.columns = ["article", "tags", "tag_with_tags", "tag_with_text"]

    return small_result, result
