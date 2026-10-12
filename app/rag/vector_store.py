
from pathlib import Path

from pymilvus import MilvusClient


COLLECTION_NAME = "paper_chunks"
VECTOR_DIM = 512

DB_PATH = Path("data/milvus/knowledge.db")


class MilvusVectorStore:
    def __init__(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        self.client = MilvusClient(uri=str(DB_PATH.resolve()))
        self.collection_name = COLLECTION_NAME

    def create_collection(self, reset=False):
        """创建Collection。reset=True仅用于重建本地测试数据。"""

        exists = self.client.has_collection(
            collection_name=self.collection_name
        )

        if exists and reset:
            self.client.drop_collection(
                collection_name=self.collection_name
            )
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=VECTOR_DIM,
                primary_field_name="id",
                vector_field_name="vector",
                id_type="int",
                metric_type="COSINE",
                auto_id=False,
                enable_dynamic_field=True
            )
            print("Collection创建成功")
        else:
            print("Collection已存在，跳过创建")

    def insert_chunks(self, chunks, vectors):
        """批量插入Chunk及其Embedding向量。"""

        if len(chunks) != len(vectors):
            raise ValueError("Chunk数量和向量数量不一致")

        if len(vectors.shape) != 2 or vectors.shape[1] != VECTOR_DIM:
            raise ValueError("Embedding维度不正确")

        data = []

        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            data.append({
                "id": index,
                "vector": vector.tolist(),
                "text": chunk.page_content,
                "source": str(chunk.metadata.get("source", "")),
                "page_number": int(
                    chunk.metadata.get("page_number", 0)
                ),
                "start_index": int(
                    chunk.metadata.get("start_index", -1)
                )
            })

        for start in range(0, len(data), 16):
            batch = data[start:start + 16]
            self.client.insert(
                collection_name=self.collection_name,
                data=batch
            )

        print(f"写入完成：{len(data)}条Chunk")

    def search(self, query_vector, top_k=5):
        """根据查询向量搜索最相似的Chunk。"""

        if len(query_vector) != VECTOR_DIM:
            raise ValueError("查询向量维度不正确")

        # 检查Collection是否存在
        if not self.client.has_collection(
                collection_name=self.collection_name
        ):
            raise RuntimeError(
                f"Collection {self.collection_name} 不存在，请先执行build_index.py"
            )

        # 加载Collection，使其可以进行向量检索
        self.client.load_collection(
            collection_name=self.collection_name
        )

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector.tolist()],
            limit=top_k,
            search_params={"metric_type": "COSINE"},
            output_fields=[
                "text",
                "source",
                "page_number",
                "start_index"
            ]
        )

        return results[0]
