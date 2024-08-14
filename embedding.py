from langchain_upstage import UpstageEmbeddings

class UpstageEmbeddingService:
    def __init__(self, api_key: str, model: str = "solar-embedding-1-large"):
        self.embeddings = UpstageEmbeddings(api_key=api_key, model=model)

    def embed_documents(self, documents: list) -> list:
        return self.embeddings.embed_documents(documents)

    def embed_query(self, query: str) -> list:
        return self.embeddings.embed_query(query)