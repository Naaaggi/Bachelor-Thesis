import openai
from api_secrets import API_KEY_OPENAI
openai.api_key = API_KEY_OPENAI


def ask_gpt(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )
    return response['choices'][0]['text']


if __name__ == "__main__":
    while True:
        prompt = input("You: ")
        ask_gpt(prompt)
