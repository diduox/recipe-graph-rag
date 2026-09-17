from time import perf_counter

from dotenv import load_dotenv

from main import AdvancedGraphRAGSystem
from config import DEFAULT_CONFIG
from rag_modules.intelligent_query_router import SearchStrategy

if __name__ == "__main__":
    load_dotenv()

    config = DEFAULT_CONFIG
    config.enable_parent_doc_retrieval = True
    config.top_k = 3

    system = AdvancedGraphRAGSystem(config)
    system.initialize_system()

    try:
        system.build_knowledge_base()

        # python会自动把相邻的字符串拼接起来
        question = ("青蟹和椰浆可以通过哪道菜谱关联起来？"
                    "这道菜中椰浆的用量是多少？")

        strategies = [
            SearchStrategy.HYBRID_TRADITIONAL,
            SearchStrategy.GRAPH_RAG,
            SearchStrategy.COMBINED,
        ]

        for strategy in strategies:
            start = perf_counter()

            documents = system.query_router.search_with_strategy(
                query=question,
                strategy=strategy,
                top_k=config.top_k,
            )

            elapsed = perf_counter() - start

            print(f"\n{'=' * 50}")
            print(f"策略：{strategy.value}")
            print(f"检索耗时：{elapsed:.2f} 秒")
            print(f"文档数：{len(documents)}")

            for index,doc in enumerate(documents,1):
                print(f"\n[证据{index}]")
                print(doc.page_content)

            if strategy == SearchStrategy.COMBINED:
                answer = (
                    system.generation_module.generate_adaptive_answer(
                        question=question,
                        documents=documents,
                    )
                )

                print("\n组合检索生成的回答：")
                print(answer)



    finally:
        system.close()