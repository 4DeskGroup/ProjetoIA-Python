from llm.google_llm import create_llm

def main():
    llm = create_llm()

    print("Bem-vindo! Pergunte o que quiser (ou digite 'exit' para sair).")

    while True:
        user_input = input("Você: ")

        if user_input.lower() == "exit":
            print("Encerrando o chatbot. Até logo!")
            break

        messages = [
            (
                "system",
                "Você é um assistente que responde perguntas de qualquer tipo.",
            ),
            ("human", user_input),
        ]

        ai_msg = llm.invoke(messages)

        print(f"Assistente: {ai_msg.content}")

if __name__ == "__main__":
    main()
