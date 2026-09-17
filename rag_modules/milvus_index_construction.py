from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient,DataType,CollectionSchema,FieldSchema
from typing import List
from langchain_core.documents import Document


class MilvusIndexConstructionModule:
    """ 负责文本向量化和 Milvus 索引构建 """
    def __init__(
            self,
            host: str = "localhost",
            port: int = 19530,
            collection_name: str = "cooking_knowledge",
            dimension: int = 512,
            model_name:str = "BAAI/bge-small-zh-v1.5"
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.model_name = model_name

        self.client = None
        self.embeddings = None
        self.collection_created = False

        self._setup_client()
        self._setup_embeddings()

    def _setup_client(self):
        self.client = MilvusClient(
            uri=f"http://{self.host}:{self.port}"
        )

    def _setup_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name = self.model_name,
            model_kwargs={"device":"cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )    

    def _create_collection_schema(self)->CollectionSchema:
        fields = [
            FieldSchema(name="id",dtype=DataType.VARCHAR,max_length = 150,is_primary=True,),
            FieldSchema(name="vector",dtype=DataType.FLOAT_VECTOR,dim=self.dimension,),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=15000),
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="recipe_name", dtype=DataType.VARCHAR, max_length=300),
            FieldSchema(name="node_type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="cuisine_type", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="difficulty", dtype=DataType.INT64),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=150),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100),
        ]
        return CollectionSchema(
            fields=fields,
            description="中式烹饪知识图谱向量集合",
        )

    def create_collection(self)->bool:
        if self.client.has_collection(self.collection_name):
            self.collection_created = True
            return True

        schema = self._create_collection_schema()

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            metric_type="COSINE",
            consistency_level="Strong",
        )

        self.collection_created = True
        return True

    # 据说创建向量索引不仅能增加速度，还能改善召回
    def create_index(self)->bool:
        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={
                "M":16,
                "efConstruction":200
            },
        )

        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params,
        )

        return True

    # 将文档块编码为向量，并写入Milvus
    def build_vector_index(self,chunks:List[Document])-> bool:

        self.create_collection()

        texts = [chunk.page_content for chunk in chunks]
        vectors = self.embeddings.embed_documents(texts)

        entities = []
        for chunk,vector in zip(chunks,vectors):
            metadata = chunk.metadata

            entity = {
                "id": metadata["chunk_id"],
                "vector": vector,
                "text": chunk.page_content,
                "node_id": metadata["node_id"],
                "recipe_name": metadata["recipe_name"],
                "node_type": metadata["node_type"],
                "category": metadata["category"],
                "cuisine_type": metadata["cuisine_type"],
                "difficulty": int(metadata["difficulty"]),
                "doc_type": metadata["doc_type"],
                "chunk_id": metadata["chunk_id"],
                "parent_id": metadata["parent_id"],
            }

            entities.append(entity)

        batch_size = 100
        for i in range(0,len(entities),batch_size):
            batch = entities[i:i + batch_size]

            self.client.insert(
                collection_name=self.collection_name,
                data=batch,
            )

            print(f"已写入 {min(i + batch_size, len(entities))}/{len(entities)}")

        self.create_index()
        self.client.load_collection(
            collection_name=self.collection_name
        )

        return True

    def similarity_search(self,query:str,k:int=5)->list:
        query_vector = self.embeddings.embed_query(query)

        # 随查询结果返回来的字段
        output_fields = [
            "text", "node_id", "recipe_name", "node_type",
            "category", "cuisine_type", "difficulty",
            "doc_type", "chunk_id", "parent_id",
        ]

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="vector",
            limit=k,
            output_fields=output_fields,
            search_params={
                "metric_type": "COSINE",
                "params": {"ef": 64},
            },
        )

        formatted_results = []
        # 这里results[0]指的是第一个问题的结果，因为search可以同时输入多个问题
        for hit in results[0]:
            entity = hit["entity"]
            formatted_results.append({
                "id": hit["id"],
                "score": hit["distance"],
                "text": entity["text"],
                # 这里是一个字典推导式，我没看出来
                "metadata": {
                    field: entity[field]
                    for field in output_fields
                    if field != "text"
                },
            })

        return formatted_results