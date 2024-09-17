from llm.google_llm import create_llm

def main():
    llm = create_llm()

    messages = [
        (
            "system",
            "Você é um assistente que realiza traduções do inglês para o português. Traduza a sentença do usuário.",
        ),
        ("human", "I love programming."),
    ]

    ai_msg = llm.invoke(messages)
    print(ai_msg.content)

if __name__ == "__main__":
    main()
