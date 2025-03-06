import ollama
from ollama._types import Options
options = Options(temperature=0.8)

def get_completion(prompt):
    response = ollama.chat(model='mistral', messages=[
      {
        'role': 'user',
        'content': prompt,
      },
    ], options=options)
    return response['message']['content']