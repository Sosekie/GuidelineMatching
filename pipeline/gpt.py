from openai import OpenAI


with open('prompt_system.txt', 'r', encoding='utf-8') as file:
    system_content = file.read()

with open('prompt_user.txt', 'r', encoding='utf-8') as file:
    user_content = file.read()

with open('prompt_assistant.txt', 'r', encoding='utf-8') as file:
    assistant_content = file.read()

with open('question.txt', 'r', encoding='utf-8') as file:
    question_content = file.read()

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_content},
        # {"role": "user", "content": user_content},
        # {"role": "assistant", "content": assistant_content},
        {
            "role": "user",
            "content": question_content
        }
    ]
)

print(completion.choices[0].message.content)

print(completion.choices[0].message.content[-1])