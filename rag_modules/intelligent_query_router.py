from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from typing import List,Tuple
from langchain_core.documents import Document


class SearchStrategy(Enum):
    HYBRID_TRADITIONAL = "hybrid_traditional"
    GRAPH_RAG = "graph_rag"
    COMBINED = "combined"

@dataclass
class QueryAnalysis:
    query_complexity:float          # 由LLM给出问题的复杂度评分
    relationship_intensity:float    # 对实体之间关系的依赖程度
    reasoning_required:bool
    entity_count:int
    recommended_strategy:SearchStrategy # 该问题推荐的检索策略
    confidence:float
    reasoning:str

class IntelligentQueryRouter:
    def __init__(
        self,
        traditional_retrieval,
        graph_rag_retrieval,
        llm_client,
        config,
    ):
        self.traditional_retrieval = traditional_retrieval
        self.graph_rag_retrieval = graph_rag_retrieval
        self.llm_client = llm_client
        self.config = config

        self.route_stats = {
            "traditional_count":0,
            "graph_rag_count":0,
            "combined_count":0,
            "total_queries":0,
        }

    def analyze_query(self,query:str)->QueryAnalysis:
        prompt = f"""
            作为 RAG 系统的查询分析专家，请分析以下查询：

            查询：{query}

            分析维度：
            1. query_complexity：查询复杂度，0 到 1。
            简单信息查找较低，多条件、比较或关系分析较高。
            2. relationship_intensity：对实体间关系的依赖程度，0 到 1。
            3. reasoning_required：是否需要推理，返回布尔值。
            4. entity_count：明确实体的数量。
            5. recommended_strategy：请根据当前已实现的能力选择策略：
            hybrid_traditional：
            对菜谱文本进行混合检索，支持回填完整菜谱。
            优先用于指定菜谱的食材、用量、制作步骤和步骤先后顺序问题。

            - graph_rag：
            支持实体间的多跳连接路径，以及围绕明确中心实体的局部子图。
            适合“哪些菜谱同时连接两种食材”等路径探索问题。
            没有专门的动作时序推理、因果推理或最短路径算法。
            不要仅因问题出现“关系”“之前”“之后”就选择此策略。

            - combined：
            当回答确实同时需要图连接证据和完整菜谱文本时使用。

            独立图检索的 multi_hop 分支已支持 max_difficulty：
            路径上的菜谱必须具有难度值，且不超过指定上限。
            此能力尚未覆盖子图分支或混合检索，也不支持其他属性约束。
            6. confidence：对推荐策略的信心，0 到 1。
            7. reasoning：简短说明推荐理由。

            只返回 JSON，不要添加 Markdown 代码块：
            {{
                "query_complexity": 0.6,
                "relationship_intensity": 0.8,
                "reasoning_required": true,
                "entity_count": 2,
                "recommended_strategy": "graph_rag",
                "confidence": 0.85,
                "reasoning": "查询关注实体之间的连接关系"
            }}
        """

        response = self.llm_client.chat.completions.create(
            model=self.config.llm_model,
            messages = [{"role":"user","content":prompt}],
            temperature=0.1,
            max_tokens=self.config.max_tokens,
            extra_body={"thinking": {"type": "disabled"}},
        )

        result = json.loads(
            response.choices[0].message.content.strip()
        )

        return QueryAnalysis(
            query_complexity=result.get("query_complexity", 0.5),
            relationship_intensity=result.get("relationship_intensity", 0.5),
            reasoning_required=result.get("reasoning_required", False),
            entity_count=result.get("entity_count", 1),
            recommended_strategy=SearchStrategy(
                result.get("recommended_strategy", "hybrid_traditional")
            ),
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning", "默认分析"),
        )

    def _combined_search(
        self,
        query:str,
        top_k:int,
    )->List[Document]:
        traditional_k = max(1,top_k //2)
        graph_k = top_k - traditional_k

        traditional_docs = self.traditional_retrieval.hybrid_search(
            query,traditional_k
        )
        graph_docs = self.graph_rag_retrieval.graph_rag_research(
            query,graph_k
        )

        combined_docs =[]
        seen_contents = set()

        max_len = max(len(traditional_docs),len(graph_docs))

        # 什么奇怪循环，我第一次还没看懂，要是我就把 i 写在里面
        for i in range(max_len):
            for source,docs in[
                ("graph_rag",graph_docs),
                ("traditional",traditional_docs),
            ]:
                if i >= len(docs):
                    continue

                doc = docs[i]
                content_hash = hashlib.md5(
                    doc.page_content[:100].encode("utf-8")
                ).hexdigest()

                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc.metadata["search_source"] = source
                    combined_docs.append(doc)

        return combined_docs[:top_k]

    def route_query(
        self,
        query:str,
        top_k:int=5, 
    )->Tuple[List[Document],QueryAnalysis]:
        analysis = self.analyze_query(query)
        strategy = analysis.recommended_strategy

        self._update_route_stats(strategy)

        documents = self.search_with_strategy(
            query=query,
            strategy=strategy,
            top_k=top_k,
        )

        documents = self._post_process_results(
            documents,analysis
        )
        return documents,analysis

    def _update_route_stats(self,strategy:SearchStrategy):
        self.route_stats["total_queries"] += 1
        if strategy == SearchStrategy.HYBRID_TRADITIONAL:
            self.route_stats["traditional_count"] += 1
        elif strategy == SearchStrategy.GRAPH_RAG:
            self.route_stats["graph_rag_count"] += 1
        else:
            self.route_stats["combined_count"] += 1

    # 把Analysis产生的信息放入Doc
    def _post_process_results(
        self,
        documents:List[Document],
        analysis:QueryAnalysis,
    )->List[Document]:
        for doc in documents:
            doc.metadata.update({
                "route_strategy": analysis.recommended_strategy.value,
                "query_complexity": analysis.query_complexity,
                "route_confidence": analysis.confidence,
            })

        return documents

    # 将选择查询策略进一步封装
    def search_with_strategy(
        self,
        query:str,
        strategy:SearchStrategy,
        top_k:int=5
    )->List[Document]:
        if strategy == SearchStrategy.HYBRID_TRADITIONAL:
            return self.traditional_retrieval.hybrid_search(
                query,top_k
            )

        if strategy == SearchStrategy.GRAPH_RAG:
            return self.graph_rag_retrieval.graph_rag_research(
                query,top_k
            )

        return self._combined_search(query,top_k)