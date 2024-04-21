from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference import infer
from llm import advanced_tag_list, pipeline, tagging, tags_creation

app = FastAPI()


class TextData(BaseModel):
    text: str


class Articles(BaseModel):
    articles: List[str]


@app.post("/infer/")
async def perform_inference(text_data: TextData):
    try:
        tags = infer(text_data.text)
        return {"tags": tags}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-tags/")
async def generate_tags(article_data: Articles):
    try:
        tags_df = tags_creation(article_data.articles)
        return tags_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify-tags/")
async def classify_tags(article_data: Articles):
    try:
        classified_tags_df = tagging(article_data.articles, advanced_tag_list, 0.15)
        return classified_tags_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-articles/")
async def process_articles(article_data: Articles):
    try:
        small_result, full_result = pipeline(article_data.articles)
        return {
            "small_result": small_result.to_dict(orient="records"),
            "full_result": full_result.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
