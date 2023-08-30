import os
import anthropic

client = anthropic.Client(open("claude_api_key.txt").read().strip())
response = client.completion(
    prompt=f"{anthropic.HUMAN_PROMPT} How many toes do dogs have?{anthropic.AI_PROMPT}",
    stop_sequences = [anthropic.HUMAN_PROMPT],
    model="claude-v1",
    max_tokens_to_sample=100,
)
print(response)