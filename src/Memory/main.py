from helpers import (
    load_environment, get_embedder_model, get_llm_model, initialize_faiss_index,
    add_human_message, add_ai_message, get_first_human_message, get_first_ai_message,
    build_prompt_from_history, search_similar_question, add_vector_to_index
)

# Carregar ambiente e configurar modelos
load_environment()
embedder = get_embedder_model()
llm = get_llm_model()
index = initialize_faiss_index()

# Loop contínuo para capturar e processar as dúvidas do usuário
while True:
    user_input = input("Digite sua dúvida (ou 'sair' para encerrar): ")

    if user_input.lower() == "sair":
        print("Encerrando o chat.")
        break

    if "primeira pergunta" in user_input.lower():
        first_question = get_first_human_message()
        if first_question:
            print(f"Sua primeira pergunta foi: {first_question}")
        else:
            print("Você ainda não fez nenhuma pergunta.")
        continue

    if "primeira resposta" in user_input.lower():
        first_answer = get_first_ai_message()
        if first_answer:
            print(f"A minha primeira resposta foi: {first_answer}")
        else:
            print("Eu ainda não forneci nenhuma resposta.")
        continue

    # Vetoriza a entrada do usuário
    user_input_vector = embedder.encode([user_input])

    # Busca no FAISS por perguntas anteriores semelhantes
    search_similar_question(index, user_input_vector)

    # Adiciona a nova pergunta ao histórico de mensagens
    add_human_message(user_input)

    # Constrói o histórico para ser enviado ao LLM
    formatted_history = build_prompt_from_history() + f"Usuário: {user_input}"

    # Envia a pergunta com o histórico ao LLM
    response = llm.invoke(formatted_history)

    # Exibe a resposta gerada pelo modelo
    print(response)

    # Armazena a resposta no histórico de mensagens
    add_ai_message(response)

    # Vetoriza e armazena a entrada do usuário no FAISS
    add_vector_to_index(index, user_input_vector)
