from regex import F, P
import requests
import json
from data_loader import FullMultiOutputDataset, PartialMultiOutputDataset

API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJ6engiLCJVc2VyTmFtZSI6Inp6eCIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxOTA4NzcwODI0Njc2OTgzNjAzIiwiUGhvbmUiOiIxODk2NDg5MzY0OSIsIkdyb3VwSUQiOiIxOTA4NzcwODI0NjcyNzg5Mjk5IiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiIiwiQ3JlYXRlVGltZSI6IjIwMjUtMDQtMDggMTc6NTg6NTkiLCJUb2tlblR5cGUiOjEsImlzcyI6Im1pbmltYXgifQ.ruNDDJoLEdENj_TnGuqjLXVKO1q4nC_YVYVKg3l8rfoppp-jgidDRX6kp0T1rqn_ihvRJtO80gkNsu8gzKFqHPBiB-WxNzqmrsmm2-Hw3BQLT1NAjjWvBM0UAdu8PmcFhpL7oJEmMUGRb7FmcAx6kCA4MzEm9XZsr3bjejhgy7bvci3l7ihmizBv-E_Sh1qCaZt3OpuWAOgZ7tXiVcqrsn6GicW_k0FUkNsPWE_p3ThfIT4d75ZrnqYF2jpI1mcPj9DhLUVduHcIYOnv8-xuHLPvsXnOTX7j26sEwJSo-XI1P-ilPXF3nM0D8kMBwPzAJPtHYj9xlw5J8upReEJDfQ"
GROUP_ID = "1908770824672789299"
MODEL_NAME = "MiniMax-Text-01"

def call_minimax(prompt):
    url = "https://api.minimax.chat/v1/text/chatcompletion_v2"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        print(result)
        return result["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

def extract_keywords(dataset):
    results = []

    for sample in dataset:
        question = sample['question']
        requirement = sample['requirement']
        outputs = sample['outputs']

        result = {
            "question": question,
            "requirements": requirement,
            "summarized_keywords": []
        }

        for i, output in enumerate(outputs):
            prompt = f"""
You are given a user's phone purchasing requirements and a recommendation text.

Requirements:
{requirement}

Recommendation:
{output}

Your task is to extract concise keywords that correspond to each of the above requirements.
Please return the results as:
- Requirement 1: [...]
- Requirement 2: [...]
- ...
If a requirement is not clearly addressed, mark it as "Not mentioned".
"""
            keywords = call_minimax(prompt)
            result["summarized_keywords"].append({f"output{i+1}": keywords})

        results.append(result)

    return results

if __name__ == "__main__":
    dataset = FullMultiOutputDataset("/root/datasets/data.json", tokenizer=None)
    results = extract_keywords(dataset)

    with open("keywords_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)