from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

candidate_labels = [
    "Политика",
    "Экономика",
    "Общество",
    "Закон и право",
    "Кино",
    "Телевидение",
    "Персоны",
    "События",
    "Бренды",
    "Наука",
    "Гаджеты",
    "Соцсети",
    "Технологии",
    "Опросы",
    "Головоломки",
    "Дом",
    "Транспорт",
    "Погода",
    "Рецепты",
    "Мода",
    "Красота",
]


model_id = "infernalfox"

text = "Cxu mi povas iri trinki teon"



model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def infer(text):

    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    output = model(**inputs)

    logits = output["logits"]

    tags = [bool(logit > 0.5) for logit in logits[0].tolist()]

    TAGS = [candidate_labels[index] for index, tag in enumerate(tags) if tag]

    return TAGS


if __name__ == '__main__':
    print(infer(text))
