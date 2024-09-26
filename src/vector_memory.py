import faiss
from langchain.schema import HumanMessage, AIMessage
from sentence_transformers import SentenceTransformer

# Variável global para o histórico de mensagens
message_history = []

# Modelo de embedding para vetorização de perguntas/respostas
def get_embedder_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Inicializa o índice FAISS
def initialize_faiss_index(dimension=384):
    return faiss.IndexFlatL2(dimension)  # Índice FAISS para busca por similaridade de vetores

# Funções para manipulação do histórico
def add_human_message(content):
    """Adiciona uma mensagem do usuário no histórico"""
    message_history.append(HumanMessage(content=content))

def add_ai_message(content):
    """Adiciona uma mensagem da IA no histórico"""
    message_history.append(AIMessage(content=content))

def get_nth_human_message(n):
    """Retorna a N-ésima pergunta feita pelo usuário"""
    human_messages = [msg.content for msg in message_history if isinstance(msg, HumanMessage)]
    if n <= len(human_messages):
        return human_messages[n - 1] 
    return None

def get_nth_ai_message(n):
    """Retorna a N-ésima resposta fornecida pela IA"""
    ai_messages = [msg.content for msg in message_history if isinstance(msg, AIMessage)]
    if n <= len(ai_messages):
        return ai_messages[n - 1]
    return None

def build_prompt_from_history():
    """Constrói o histórico de mensagens para o modelo LLM"""
    history = ""
    for message in message_history:
        if isinstance(message, HumanMessage):
            history += f"Usuário: {message.content}\n"
        elif isinstance(message, AIMessage):
            history += f"IA: {message.content}\n"
    return history

# Funções para manipulação do FAISS
def search_similar_question(index, user_input_vector):
    """Procura perguntas semelhantes no FAISS"""
    if len(message_history) > 0:
        D, I = index.search(user_input_vector, 1)  # Busca pelo vetor mais semelhante
        if D[0][0] < 0.5:  # Se encontrar uma correspondência com distância menor que 0.5 (limite arbitrário)
            similar_question = message_history[I[0][0]].content
            similar_answer = message_history[I[0][0] + 1].content  # Supondo que a resposta segue a pergunta
            print(f"Memória: Lembrei-me de uma pergunta anterior semelhante: {similar_question}")
            print(f"Resposta anterior: {similar_answer}")

def add_vector_to_index(index, user_input_vector):
    """Adiciona o vetor ao índice FAISS"""
    index.add(user_input_vector)
