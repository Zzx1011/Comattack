import requests
import streamlit as st
import json

API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJ6engiLCJVc2VyTmFtZSI6Inp6eCIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxOTA4NzcwODI0Njc2OTgzNjAzIiwiUGhvbmUiOiIxODk2NDg5MzY0OSIsIkdyb3VwSUQiOiIxOTA4NzcwODI0NjcyNzg5Mjk5IiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiIiwiQ3JlYXRlVGltZSI6IjIwMjUtMDQtMDggMTc6NTg6NTkiLCJUb2tlblR5cGUiOjEsImlzcyI6Im1pbmltYXgifQ.ruNDDJoLEdENj_TnGuqjLXVKO1q4nC_YVYVKg3l8rfoppp-jgidDRX6kp0T1rqn_ihvRJtO80gkNsu8gzKFqHPBiB-WxNzqmrsmm2-Hw3BQLT1NAjjWvBM0UAdu8PmcFhpL7oJEmMUGRb7FmcAx6kCA4MzEm9XZsr3bjejhgy7bvci3l7ihmizBv-E_Sh1qCaZt3OpuWAOgZ7tXiVcqrsn6GicW_k0FUkNsPWE_p3ThfIT4d75ZrnqYF2jpI1mcPj9DhLUVduHcIYOnv8-xuHLPvsXnOTX7j26sEwJSo-XI1P-ilPXF3nM0D8kMBwPzAJPtHYj9xlw5J8upReEJDfQ"
GROUP_ID = "1908770824672789299"
MODEL_NAME = "MiniMax-Text-01"
OUTPUT_JSON_PATH = "/root/Compression_attack/soft_attack_icae/benchmark_outputs.json"

# build prompt
def build_prompt(task_description, input_example, target_list):
    targets = "\n".join([f"- {item}" for item in target_list])
    json_entries = ",\n".join(
        [f'    {{"target": "{item}", "generated_text": "..."}}' for item in target_list]
    )
    
    prompt = f"""
{task_description}

## Example Input:
{input_example}

## Your Task:
Based on the example above, generate corresponding outputs for the following targets:
{targets}

Return the output strictly in the following JSON format:

{{
  "outputs": [
{json_entries}
  ]
}}

Ensure consistency with the example. Keep all outputs in English. Avoid generic descriptions.
""".strip()
    
    return prompt


# API
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
        return result["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

# save to JSON 
def save_output_to_json(text, path):
    try:
        parsed = json.loads(text)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        print(f"✅ Output saved to {path}")
    except json.JSONDecodeError:
        print("❌ Failed to parse output as JSON.")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"⚠️ Raw output saved to {path} instead.")


task_description = """
You are a professional copywriter. Your task is to generate promotional texts in different brand styles based on a given product description. Each output should be approximately 1000 words and match the tone and branding style of the specified target.
"""

input_example = """
Brand: Apple
Product: iPhone 15 Pro
Promotional Text:
Introducing the iPhone 15 Pro – a remarkable fusion of design, performance, and innovation. Crafted with aerospace-grade titanium, it’s not only stronger but lighter than ever. Powered by the A17 Pro chip, it redefines speed and efficiency [...]
"""

target_list = [
    "Samsung: Galaxy S24 Ultra",
    "Google: Pixel 8 Pro",
    "OnePlus: OnePlus 12",
    "Xiaomi: Xiaomi 14 Ultra",
    "Huawei: Mate 60 Pro"
]

if __name__ == "__main__":
    prompt = build_prompt(task_description, input_example, target_list)
    result = call_minimax(prompt)
    if result:
        save_output_to_json(result, OUTPUT_JSON_PATH)