from datasets import load_dataset
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import json

def dataset_to_vector(dataset_name):
    load_dotenv()

    print(f"Carregando o dataset: {dataset_name}")
    # Carregar o dataset
    dataset = load_dataset(dataset_name, split='train')

    print(f"Total de itens no dataset: {len(dataset)}")

    # Limitar o dataset para teste (ajustado para 2 itens)
    limited_dataset = dataset.select(range(500))

    # Supondo que o dataset tem várias colunas que você quer combinar, como 'review_text' e 'product_name'
    texts = []
    for idx, item in enumerate(dataset):
        review_text = item.get('review_text', '')
        product_name = item.get('product_name', '')
        overall_rating = item.get('overall_rating', '')

        # Adiciona prefixos para separar os contextos
        combined_text = f"Review: {review_text} | Produto: {product_name} | Nota: {overall_rating}"
        texts.append(combined_text)
        print(f"Exemplo {idx}: {combined_text}")

    print("Dividindo os textos em documentos menores...")
    # Dividir o texto em documentos menores
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = []
    for text in texts:
        split_documents.extend(text_splitter.split_text(text))

    print(f"Total de documentos após split: {len(split_documents)}")

    print("Obtendo embeddings...")
    # Utilizando GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Gerar embeddings para os documentos
    document_embeddings = embeddings.embed_documents(split_documents[:2])  # Limitando para teste
    print(f"Embeddings gerados: {len(document_embeddings)}")

    # Salvar os embeddings localmente para debug
    with open('faiss_embeddings.json', 'w') as f:
        json.dump(document_embeddings, f)
        print("Embeddings salvos localmente em faiss_embeddings.json")

    print("Criando a base de vetores com FAISS...")
    try:
        # Criar a base de vetores com FAISS
        vector = FAISS.from_texts(split_documents[:2], embedding=embeddings)
        retriever = vector.as_retriever()
        print("Retriever FAISS criado com sucesso.")
    except Exception as e:
        print(f"Erro ao criar a base de vetores com FAISS: {e}")
        return

    return retriever