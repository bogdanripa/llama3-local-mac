from mlx_lm import load, generate
import time
import threading
import sys
from rich.markdown import Markdown
from rich import print as rprint

model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")

def get_response(messages):
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt = tokenizer.decode(input_ids)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=500)
    return response

def loading_animation():
    animation = ["   ", ".  ", ".. ", "..."]
    idx = 0
    while not done:
        print("\rLlama: " + animation[idx % len(animation)], end="")
        idx += 1
        time.sleep(0.2)

# Start with a system message
# messages = [{"role": "system", "content": "You are a funny cloud software architect. If needed, summarize the answer to fit in 450 tokens. Answer simple questions with simple answers."}]
messages = [{"role": "system", "content": "You are a code reviewer looking for bugs and way to improve performance. If needed, summarize the answer to fit in 450 tokens. Answer simple questions with simple answers."}]
# messages = [{"role": "system", "content": "Ești un comedian super-funny cu ură pe politicienii de stânga. Dacă este necesar, rezumați răspunnde în în 450 de tokens. Răspunde la întrebări simple cu răspunsuri simple."}]

while True:
    user_content = input("You:")
    messages.append({"role": "user", "content": user_content})

    done = False
    thread = threading.Thread(target=loading_animation)
    thread.start()

    response = get_response(messages)
    done = True  # Stop the loading animation

    print("\rLlama: ", end="")
    rprint(Markdown(response))
    
    messages.append({"role": "assistant", "content": response})
    thread.join()  # Wait for the loading indicator to finish
