from datasets import load_dataset
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import json
import os

def dataset_to_vector(dataset_name, use_saved_embeddings=False):
    load_dotenv()

    if use_saved_embeddings and os.path.exists('faiss_embeddings.json'):
        print("Carregando embeddings salvos do arquivo JSON...")
        with open('faiss_embeddings.json', 'r') as f:
            document_embeddings = json.load(f)
        print(f"Embeddings carregados: {len(document_embeddings)}")
        
        # Criar um vetor FAISS a partir dos embeddings salvos
        try:
            # Nota: Neste ponto, você precisa de documentos correspondentes ao número de embeddings
            # Então é importante também carregar ou gerar os documentos "split_documents"
            dataset = load_dataset(dataset_name, split='train')
            texts = []
            for item in dataset.select(range(500)):  # Limitar a 500 para consistência com embeddings
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

            # Split the text into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_documents = []
            for text in texts:
                split_documents.extend(text_splitter.split_text(text))
            
            # Cria a base de vetores FAISS usando os embeddings carregados
            vector = FAISS(embedding_fn=lambda x: document_embeddings, documents=split_documents)
            retriever = vector.as_retriever()
            print("Retriever FAISS criado a partir dos embeddings carregados.")
            return retriever
        
        except Exception as e:
            print(f"Erro ao criar a base de vetores com FAISS: {e}")
            return None
    
    else:
        print(f"Carregando o dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split='train')

        print(f"Total de itens no dataset: {len(dataset)}")

        limited_dataset = dataset.select(range(1000))

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

        print("Dividindo os textos em documentos menores...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        split_documents = []
        for text in texts:
            split_documents.extend(text_splitter.split_text(text))

        print(f"Total de documentos após split: {len(split_documents)}")

        print("Obtendo embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        document_embeddings = embeddings.embed_documents(split_documents)

        print(f"Embeddings gerados: {len(document_embeddings)}")

        with open('faiss_embeddings.json', 'w') as f:
            json.dump(document_embeddings, f)
            print("Embeddings salvos localmente em faiss_embeddings.json")

        print("Criando a base de vetores com FAISS...")
        try:
            vector = FAISS.from_texts(split_documents, embedding=embeddings)
            retriever = vector.as_retriever()
            print("Retriever FAISS criado com sucesso.")
        except Exception as e:
            print(f"Erro ao criar a base de vetores com FAISS: {e}")
            return None

        return retriever

# Uso da função
retriever = dataset_to_vector('ruanchaves/b2w-reviews01', use_saved_embeddings=True)
