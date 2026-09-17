from config import DEFAULT_CONFIG
from rag_modules.graph_data_preparation import GraphDataPreparationModule
from rag_modules.milvus_index_construction import MilvusIndexConstructionModule
from rag_modules.hybrid_retrieval import HybridRetrievalModule
from rag_modules.graph_indexing import GraphIndexingModule
from dotenv import load_dotenv
from rag_modules.generation_integration import GenerationIntegrationModule
from rag_modules.graph_rag_retrieval import GraphRAGRetrieval,GraphQuery,QueryType
from rag_modules.intelligent_query_router import IntelligentQueryRouter
from time import perf_counter


class AdvancedGraphRAGSystem:
    def __init__(self,config=None):
        self.config = config or DEFAULT_CONFIG

    def initialize_system(self):
        config = self.config

        self.data_module = GraphDataPreparationModule(
                uri=config.neo4j_uri,
                user=config.neo4j_user,
                password=config.neo4j_password,
                database=config.neo4j_database,
        )

        self.index_module = MilvusIndexConstructionModule(
                host=config.milvus_host,
                port=config.milvus_port,
                collection_name=config.milvus_collection_name,
                dimension=config.milvus_dimension,
                model_name=config.embedding_model,
        )

        self.generation_module = GenerationIntegrationModule(
                model_name=config.llm_model,
                temperature=config.temperature,
                max_tokens=config.max_tokens
        )

        self.traditional_retrieval = HybridRetrievalModule(
            config=config,
            milvus_module=self.index_module,
            data_module=self.data_module,
            llm_client=self.generation_module.client,
        )

        self.graph_rag_retrieval = GraphRAGRetrieval(
                config=config,
                llm_client=self.generation_module.client,
        )

        self.query_router = IntelligentQueryRouter(
                    traditional_retrieval=self.traditional_retrieval,
                    graph_rag_retrieval=self.graph_rag_retrieval,
                    llm_client=self.generation_module.client,
                    config=config,
        )

    def build_knowledge_base(self):
        self.data_module.load_graph_data()
        self.data_module.build_recipe_documents()

        chunks = self.data_module.chunk_documents(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        collection_name = self.config.milvus_collection_name

        if self.index_module.client.has_collection(
            collection_name=collection_name,
        ):
            self.index_module.client.load_collection(
                collection_name=collection_name,
            )
            print("已加载现有向量集合")
        else:
            print("首次运行，开始构建向量索引")
            self.index_module.build_vector_index(chunks)

        self.traditional_retrieval.initialize(chunks)
        self.graph_rag_retrieval.initialize()

        print(
            f"知识库准备完成："
            f"{len(self.data_module.documents)} 篇文档，"
            f"{len(chunks)} 个块"
        )

    def ask_question_with_routing(self,question:str):
        start = perf_counter()

        documents,analysis=self.query_router.route_query(
            question,
            top_k=self.config.top_k,
        )

        retrieval_end = perf_counter()

        context_chars = sum(
            len(doc.page_content) for doc in documents
        )

        print(
            f"路由与检索：{retrieval_end - start:.2f} 秒"
            f"返回文档：{len(documents)};"
            f"正文字符数：{context_chars}"
        )

        for index,doc in enumerate(documents,1):
            metadata = doc.metadata
            print(
                f"证据 {index}:",
                metadata.get("recipe_name", "未命名"),
                "| 节点:", metadata.get("node_id", "未提供"),
                "| 类型:", metadata.get("search_type", "未提供"),
                "| 字符数:", len(doc.page_content),
            )


        if not documents:
            return "没有检索到相关信息。",analysis

        answer = self.generation_module.generate_adaptive_answer(
            question,
            documents,
        )

        generation_end = perf_counter()

        print(
            f"答案生成：{generation_end - retrieval_end:.2f}"
            f"问题总耗时：{generation_end - start:.2f} 秒"
        )

        return answer,analysis

    def close(self):
        self.data_module.close()
        self.traditional_retrieval.close()
        self.graph_rag_retrieval.close()
        self.index_module.client.close()
        self.generation_module.client.close()      


    def run_interactive(self):
        print("\n菜谱问答已就绪，输入 quit 退出。")

        while True:
            question = input("\n你的问题：").strip()

            if not question:
                continue
            if question.lower() == "quit":
                break

            answer,analysis = self.ask_question_with_routing(question)
            print("检索策略：",analysis.recommended_strategy.value)
            print("推荐理由：",analysis.reasoning)
            print("\n回答\n",answer)



if __name__ == "__main__":
    load_dotenv()

    config = DEFAULT_CONFIG
    config.enable_parent_doc_retrieval = True
    config.top_k = 3

    system = AdvancedGraphRAGSystem(config)
    system.initialize_system()

    try:
        system.build_knowledge_base()
        system.run_interactive()
    finally:
        system.close()
    