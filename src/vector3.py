BATCH_SIZE = 1000  # Defina o tamanho dos lotes

def process_batch(batch_texts, embeddings):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = []
    
    for text in batch_texts:
        split_documents.extend(text_splitter.split_text(text))

    document_embeddings = embeddings.embed_documents(split_documents)
    return split_documents, document_embeddings

def dataset_to_vector(dataset_name, use_saved_embeddings=False):
    load_dotenv()
    
    if use_saved_embeddings and os.path.exists('faiss_embeddings.json'):
        # Carregar embeddings previamente salvos
        with open('faiss_embeddings.json', 'r') as f:
            document_embeddings = json.load(f)
        # (carregar textos ou outro processamento se necessário)
        return retriever

    # Processar o dataset
    dataset = load_dataset(dataset_name, split='train')
    texts = []
    for idx, item in enumerate(dataset):
        review_text = item.get('review_text', '')
        # (adicionar outras colunas relevantes)
        combined_text = f"review_text: {review_text}"  # Exemplo simplificado
        texts.append(combined_text)

    # Processamento em Lotes
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    all_split_documents = []
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        split_documents, document_embeddings = process_batch(batch_texts, embeddings)
        all_split_documents.extend(split_documents)
        all_embeddings.extend(document_embeddings)
        print(f"Processado lote {i // BATCH_SIZE + 1} de {len(texts) // BATCH_SIZE + 1}")
    
    # Salvar embeddings gerados
    with open('faiss_embeddings.json', 'w') as f:
        json.dump(all_embeddings, f)

    # Criar FAISS com os dados divididos
    vector = FAISS(embedding_fn=lambda x: all_embeddings, documents=all_split_documents)
    retriever = vector.as_retriever()

    return retriever
