from datasets import load_dataset
import numpy as np
import json
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def dataset_to_vector(dataset_name, use_saved_embeddings=False):
    if use_saved_embeddings and os.path.exists('faiss_embeddings.json'):
        print("Carregando embeddings salvos do arquivo JSON...")
        with open('faiss_embeddings.json', 'r') as f:
            data = json.load(f)
            document_embeddings = np.array(data['embeddings'])
            split_documents = data['documents']
        print(f"Embeddings carregados: {document_embeddings.shape[0]}")
        
        # Cria o vetor FAISS a partir dos embeddings carregados
        try:
            # Inicializa o FAISS sem argumentos
            faiss_index = FAISS()
            faiss_index.add_embeddings(document_embeddings, split_documents)
            retriever = faiss_index.as_retriever()
            print("Retriever FAISS criado a partir dos embeddings carregados.")
            return retriever
        except Exception as e:
            print(f"Erro ao criar a base de vetores com FAISS: {e}")
            return None
    
    else:
        print(f"Carregando o dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split='train')

        print(f"Total de itens no dataset: {len(dataset)}")

        limited_dataset = dataset.select(range(200))

        texts = []
        for idx, item in enumerate(limited_dataset):
            review_text = item.get('review_text', '')
            product_name = item.get('product_name', '')
            overall_rating = item.get('overall_rating', '')
            submission_date = item.get('submission_date', '')
            reviewer_id = item.get('reviewer_id', '')
            product_id = item.get('product_id', '')
            product_brand = item.get('product_brand', '')
            site_category_lv1 = item.get('site_category_lv1', '')
            site_category_lv2 = item.get('site_category_lv2', '')
            review_title = item.get('review_title', '')
            recommend_to_a_friend = item.get('recommend_to_a_friend', '')
            reviewer_birth_year = item.get('reviewer_birth_year', '')
            reviewer_gender = item.get('reviewer_gender', '')
            reviewer_state = item.get('reviewer_state', '')

            combined_text = f"review_text: {review_text} | product_name: {product_name} | overall_rating: {overall_rating} | submission_date: {submission_date} | reviewer_id: {reviewer_id} | product_id: {product_id} | product_brand: {product_brand} | site_category_lv1: {site_category_lv1} | site_category_lv2: {site_category_lv2} | review_title: {review_title} | recommend_to_a_friend: {recommend_to_a_friend} | reviewer_birth_year: {reviewer_birth_year} | reviewer_gender: {reviewer_gender} | reviewer_state: {reviewer_state}"
            texts.append(combined_text)
            print(f"Exemplo {idx}: {combined_text}")

        print("Dividindo os textos em sentenças...")
        split_documents = []
        for text in texts:
            split_documents.extend(split_text_into_sentences(text))

        print(f"Total de documentos após split: {len(split_documents)}")

        print("Obtendo embeddings...")
        model_name = "bert-base-uncased"  # Use o modelo que preferir

        # Use HuggingFaceEmbeddings com o argumento correto
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        document_embeddings = embeddings.embed_documents(split_documents)
        document_embeddings = np.array(document_embeddings)  # Convertendo para um array NumPy

        print(f"Embeddings gerados: {len(document_embeddings)}")

        with open('faiss_embeddings.json', 'w') as f:
            json.dump({
                'embeddings': document_embeddings.tolist(),
                'documents': split_documents
            }, f)
            print("Embeddings salvos localmente em faiss_embeddings.json")

        print("Criando a base de vetores com FAISS...")
        try:
            # Inicializa o FAISS sem argumentos
            faiss_index = FAISS()
            faiss_index.add_embeddings(document_embeddings, split_documents)
            retriever = faiss_index.as_retriever()
            print("Retriever FAISS criado com sucesso.")
        except Exception as e:
            print(f"Erro ao criar a base de vetores com FAISS: {e}")
            return None

        return retriever

def split_text_into_sentences(text):
    import re
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]
