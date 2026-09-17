from dataclasses import dataclass
from rag_modules.graph_data_preparation import GraphNode

@dataclass
class GraphRAGConfig:
    """ 图谱 RAG 系统配置 """

    # 搜索配置
    top_k = 5

    # neo4j配置
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "all-in-rag"
    neo4j_database: str = "neo4j"

    # 切块配置 单位是字符，不是token
    chunk_size: int = 500
    chunk_overlap: int = 50 

    # 配置embedding模型
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    milvus_dimension: int = 512

    # Milvus相关
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "cooking_knowledge"

    # 是否启用父文档回填
    enable_parent_doc_retrieval :bool = False
    # 对RFF的前几条效果进行回填
    parent_doc_top_n: int = 3
    # 每篇回填正文的字符上限(采用直接截断)
    parent_doc_max_chars: int = 4000

    # 大模型
    llm_model:str = "deepseek-flash"
    temperature:float = 0.1
    max_tokens :int = 2048



DEFAULT_CONFIG = GraphRAGConfig()