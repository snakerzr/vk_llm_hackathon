# -*- coding: utf-8 -*-
import requests
import json

url = "http://localhost:8000"

data_articles = {
    "articles": [
        "Если ребенок верит в сказку хотя бы до десяти лет, он вырастет хорошим человеком. Однажды на новогоднем представлении случилась совсем не новогодняя история: Дед Мороз потерял голос. А впереди у него была целая сцена. И шесть с половиной тысяч зрителей в зале. Повезло, что в этот момент за кулисами стоял цирковой режиссёр Коля Челноков. Он быстро переоделся, я ему так же быстро наговорил текст. И хотя у нашего нового Деда Мороза был совсем не драматический сказочный баритон, а, наоборот, тоненький тенор, он блестяще вышел из ситуации. Половину роли сыграл своими словами, но ни на секунду не остановил действие. И ему аплодировали все — и зрители, и артисты. Он нас выручил, сыграв три спектакля в этот день. К счастью, дети способны не заметить наших взрослых ошибок, а артисты умеют заполнить паузу и вытащить ситуацию, как будто ничего и не было. Вообще, готовить спектакли для детей — очень благодарная история. Взрослого человека даже самым гениальным произведением поменять сложно, у него уже сформировано свое мировоззрение. С детьми история иная: у ребенка в голове после спектакля что-то может очень сильно поменяться. Марина Цветаева говорила: «Ребенка надо заклясть!» И я пытаюсь так и сделать. Пусть он сразу даже не все поймет и осознает, но пройдет лет десять — и он вспомнит. Крупицы нравственности закладываются именно в детстве. Если человек верит в сказку, живет в сказке, скажем, хотя бы до десяти лет, то я уверен, что он вырастет хорошим человеком."
    ]
}

data_text = {
    "text": "Если ребенок верит в сказку хотя бы до десяти лет, он вырастет хорошим человеком."
}


def test_api_endpoints():
    try:
        response_infer = requests.post(f"{url}/infer/", json=data_text)
        print("Response from /infer/:")
        print(json.dumps(response_infer.json(), indent=2, ensure_ascii=False))

        response_tags = requests.post(f"{url}/generate-tags/", json=data_articles)
        print("Response from /generate-tags/:")
        print(json.dumps(response_tags.json(), indent=2, ensure_ascii=False))

        response_classify = requests.post(f"{url}/classify-tags/", json=data_articles)
        print("Response from /classify-tags/:")
        print(json.dumps(response_classify.json(), indent=2, ensure_ascii=False))

        response_process = requests.post(f"{url}/process-articles/", json=data_articles)
        print("Response from /process-articles/:")
        print(json.dumps(response_process.json(), indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"Failed to test API endpoints: {str(e)}")


test_api_endpoints()
