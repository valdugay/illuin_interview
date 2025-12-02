import numpy as np
import json
import os
from mistralai import Mistral
import tqdm


MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

client = Mistral(api_key=MISTRAL_API_KEY)

PROMPT_TEMPLATE = """
You are an expert in algorithmic problem classification.

Your task is to return a JSON list containing ONLY tags from the following set:

["math", "graphs", "strings", "number theory", "trees", "geometry", "games", "probabilities"]

Rules:
- Output MUST be a JSON list of strings.
- Only include tags from the allowed list.
- If no tag applies, return an empty list [].
- Do NOT include any explanation.
- Do NOT create new tags.

Problem Description:
{description}

User Code:
{source_code}

Return ONLY the JSON list.
"""


focus_tags = [
    "math", "graphs", "strings", "number theory",
    "trees", "geometry", "games", "probabilities"
]


# We can adjust our function to be able to monitor cost

def predict_llm_for_df(df, monitor_cost=False):
    vectors = []
    total_cost = 0.0

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):

        # To limit cost, we clip to first 10k characters (in case of very long descriptions/code)
        row["full_description"] = row["full_description"][:10000]
        row["source_code"] = row["source_code"][:10000]
        
        response = client.chat.complete(
            model="codestral-latest",
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(
                description=row["full_description"],
                source_code=row["source_code"]
            )}],
            response_format={"type": "json_object"},
        )

        if monitor_cost:
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens

            cost = (prompt_tokens * 0.3 + completion_tokens * 0.9) / 10**6  # Codestral Pricing
            total_cost += cost


        raw_json = response.choices[0].message.content

        try:
            tags = json.loads(raw_json)
        except:
            # fallback minimal : empty list
            tags = []

        # Force allowed tags only
        tags = [t for t in tags if t in focus_tags]

        vectors.append(tags_to_vector(tags))

    if monitor_cost:
        print(f"Total cost for predictions: ${total_cost:.4f}")

    return tags, np.array(vectors)


def tags_to_vector(tags):
    return [1 if t in tags else 0 for t in focus_tags]