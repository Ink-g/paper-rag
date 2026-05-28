from src.pipeline import ask

question = "What is the main idea of DDPM and how does the denoising process work?"

print(f"问题: {question}\n")
print("检索并生成回答中...\n")

answer, sources = ask(question)

print("回答:")
print(answer)
print("\n来源论文:")
for s in sources:
    print(f"  - {s['title']}")
