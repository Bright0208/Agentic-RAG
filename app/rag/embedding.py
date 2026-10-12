from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-zh-v1.5"


class EmbeddingModel:

    def __init__(self, device="cpu"):
        self.model = SentenceTransformer(
            MODEL_NAME,
            device=device
        )

    def embed_documents(self, texts: list[str]):
        """将文档转成向量"""
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    def embed_query(self, query: str):
        """将查询转成向量"""
        instruction = "为这个句子生成表示以用于检索相关文章："

        vector = self.model.encode(
            instruction + query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return vector