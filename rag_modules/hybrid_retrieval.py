import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from typing import List,Tuple,Any,Dict
from neo4j import GraphDatabase
from .graph_indexing import GraphIndexingModule
from dataclasses import dataclass
import json

# 过滤掉“的、是、在”等无实际语义的噪声词，让算法只专注于核心关键词，从而大幅提升搜索的准确度与效率。
_CHINESE_STOPWORDS = set("""
的 了 和 是 在 我 有 就 不 也 都 还 这 那 一 个 与 及 等 上 下 中 为 以 于 从 把 被 让 使 又 而 但 或
什么 怎么 如何 哪些 哪个 哪里 谁 多少 几 你 他 她 它 我们 他们 她们 它们
请问 请 想 要 需要 能 可以 应该 会 啊 呢 吧 嘛 吗 哦 呀 哈
之 其 此 该 即 各 每 些 种 类 时 后 前 里 外 内 间 已经 正在 一些 一下
""".split())

# 结果类（怎么和当初学javaweb一样，还有一个结果类）
@dataclass
class RetrievalResult:
    content:str
    node_id:str
    node_type:str
    relevance_score:float
    retrieval_level:str     # 这个是新加入的，以后自会描述
    metadata:Dict[str,Any]


class HybridRetrievalModule:
    def __init__(self,config,milvus_module,data_module,llm_client):
        self.config = config
        self.milvus_module = milvus_module
        self.data_module = data_module
        self.llm_client = llm_client
        self.driver = None
        self.bm25 = None
        self.bm25_corpus_docs = []
        self.graph_indexing = GraphIndexingModule()
        # 图索引有没有被构建
        self.graph_indexed = False

        
    @staticmethod
    def _tokenize_chinese(text:str)->List[str]:
        return [
            token
            for token in jieba.lcut(text)
            if token.strip() and token not in _CHINESE_STOPWORDS
        ]

    def initialize(self,chunks:List[Document]):
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(
                self.config.neo4j_user,
                self.config.neo4j_password,
                ),
        )

        # list()是复制列表，避免改变原来的列表
        self.bm25_corpus_docs = list(chunks)

        tokenized_corpus = [
            self._tokenize_chinese(doc.page_content)
            for doc in chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self._build_graph_index()

        self._parent_doc_map = self._build_parent_doc_map()

    def bm25_search(self,query:str,top_k:int=5)->List[Document]:
        tokenized_query = self._tokenize_chinese(query)
        # 这里的bm25是BM250kapi对象，已经包含了所有答案的信息
        scores = self.bm25.get_scores(tokenized_query)

        # 类似这个样子[4, 1, 3, ......]
        top_indices = sorted(
            range(len(scores)),
            key=lambda i:scores[i],
            reverse=True,
        )[:top_k]

        docs = []
        for idx in top_indices:
            score = float(scores[idx])
            # 过滤分数为0或者负数的文档
            if score <= 0:
                continue

            src = self.bm25_corpus_docs[idx]
            doc = Document(
                page_content=src.page_content,
                metadata={
                    **src.metadata,
                    "search_method": "bm25",
                    "search_type": "bm25",
                    "bm25_score": score,
                }
            )
            docs.append(doc)

        return docs

    # 经过图增强的向量搜索
    def vector_search_enhanced(self,query:str,top_k:int = 5,) -> List[Document]:

        results = self.milvus_module.similarity_search(
            query,
            # 这里目前还没有意义，因为我们还没做额外筛选和重排
            k=top_k*2,
        )
        # 将milvus查找的结果也变成doc
        docs = []
        for result in results:
            content = result["text"]
            metadata = result["metadata"]

            neighbors = self._get_node_neighbors(
                node_id = metadata["node_id"]
            )

            if neighbors:
               content += f"\n相关信息: {', '.join(neighbors)}"

            doc = Document(
                page_content=content,
                metadata={
                    **metadata,
                    "score":result["score"],
                    "search_type":"vector_enhanced",
                },
            )
            docs.append(doc)
            
        return docs[:top_k]

    """
    ranked_lists 差不多长这样
    ranked_lists = [
    ("vector", vector_docs),
    ("bm25", bm25_docs),
    ]
    """
    # 这个函数好恶心啊
    @staticmethod
    def _rrf_merge(ranked_lists:list,top_k:int,k:int=60,)-> List[Document]:
        # 每道菜在每一路的最好名次，用来算分
        """
        差不多长这样
        best_rank_per_source = {
            "咖喱炒蟹": {
                "vector": 1,  # 向量检索中最好排第 1
                "bm25": 3,   # BM25 中最好排第 3
                }
            }
        """

        best_rank_per_source = {}
        # 每道菜在每一路命中了多少块，用于观察
        chunk_hits_per_source ={}
        best_doc_info ={}

        """
        (0, ("vector", ["块A", "块B"])) 
        (1, ("bm25", ["块C", "块A"]))
        迭代的每个元素差不多长这样
        """ 
        for priority,(source,docs) in enumerate(ranked_lists):
            for rank,doc in enumerate(docs,start=1):
                doc_id = str(doc.metadata["node_id"])

                # 因为输入已经排好序了，所以第一次出现就是最好名次
                if doc_id not in best_rank_per_source:
                    best_rank_per_source[doc_id] = {}
                    chunk_hits_per_source[doc_id] = {}

                # source_ranks指向内部字典 为其赋值
                source_ranks = best_rank_per_source[doc_id]
                # source指的是来自BM25还是vector 这里就相当于为每一个菜谱记录它们的最好BM名次和Vector名次
                if source not in source_ranks:
                    source_ranks[source] = rank

                hits = chunk_hits_per_source[doc_id]
                hits[source] = hits.get(source,0) + 1
                # 这个(,)比较是先比较第一个，如果相同再比较第二个，就是说如果排名相同，我们认为第一种方式更重要
                # 这个只用来展示，不用来计算排名
                if (doc_id not in best_doc_info or (rank,priority) < best_doc_info[doc_id][:2]):
                    best_doc_info[doc_id] = (rank,priority,doc)

        rrf_scores = {
            doc_id:sum(
                1 / (k + rank)
                for rank in source_ranks.values()
            )
            for doc_id,source_ranks in best_rank_per_source.items()
        }
        # 直接遍历字典默认遍历键 .values()遍历值 .items()遍历键和值
        # 所以sorted_ids是个列表
        sorted_ids = sorted(
            rrf_scores,
            key=lambda doc_id: rrf_scores[doc_id],
            reverse=True,
        )

        merged = []
        for doc_id in sorted_ids[:top_k]:
            # 2是doc的具体内容 (rank,priority,doc)
            source_doc = best_doc_info[doc_id][2]

            merged.append(Document(
                page_content=source_doc.page_content,
                metadata={
                    **source_doc.metadata,
                    "rrf_score": rrf_scores[doc_id],
                    "rrf_sources": list(best_rank_per_source[doc_id]),
                    "rrf_ranks": dict(best_rank_per_source[doc_id]),
                    "rrf_chunk_hits": dict(chunk_hits_per_source[doc_id]),
                    "final_score": rrf_scores[doc_id],
                },
            ))

        return merged

    def hybrid_search(
            self,
            query:str,
            top_k:int=5
    ) -> List[Document]:
        # 每路先多取候选，再融合选前几名。因为同一道菜可能占据多个块，如果每路只取三个块，按菜谱合并后可能只剩一道菜。
        candidate_k = max(top_k * 2 ,10)

        dual_docs = self.dual_level_retrieval(
            query, top_k=candidate_k
        )
        vector_docs = self.vector_search_enhanced(
            query,
            top_k=candidate_k,
        )
        bm25_docs = self.bm25_search(
            query,
            top_k=candidate_k
        )

        for doc in dual_docs:
            doc.metadata.setdefault("search_method", "dual_level")
        for doc in vector_docs:
            doc.metadata["search_method"] = "vector"
        # BM25的在之前已经添加过了

        final_docs = self._rrf_merge(
            ranked_lists=[
                ("dual_level", dual_docs),
                ("vector", vector_docs),
                ("bm25", bm25_docs),
            ],
            top_k=top_k,
        )

        if self.config.enable_parent_doc_retrieval:
            final_docs = self._attach_parent_documents(final_docs)

        return final_docs

    # 这个方法直接返回邻居节点的名称，不反悔node_id
    # 因为我们直接把其当作一个信息拼接到正文中
    def _get_node_neighbors(
        self,
        node_id:str,
        max_neighbors:int=3,
    )->List[str]:
        # 没有箭头表示匹配时不限制关系方向
        query = """
            MATCH (n {nodeId: $node_id})-[r]-(neighbor)
            RETURN neighbor.name AS name
            LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(
                query,
                {"node_id":node_id,"limit":max_neighbors},
            )
            return [
                record["name"]
                for record in result
                if record["name"]
            ]

    def close(self):
        if self.driver:
            self.driver.close()

    """
    经过加载、文档构建和分块后, data_module的内容
    data_module
    ├── driver          Neo4j 驱动
    ├── recipes         菜谱节点列表
    ├── ingredients     食材节点列表
    ├── cooking_steps   步骤节点列表
    ├── documents       完整菜谱文档列表
    └── chunks          分块后的文档列表
    """
    # 相当于我们把self.data_module.documents的列表变成方便查询的字典
    def _build_parent_doc_map(self)->dict:
        return {
            str(doc.metadata["node_id"]):doc
            for doc in self.data_module.documents
        }

    # 父文档回填
    def _attach_parent_documents(
        self,
        docs:List[Document],
    ) -> List[Document]:
        out = []

        for i,doc in enumerate(docs):
            # 如果超过了回填名额，就只填入原始块
            if i >= self.config.parent_doc_top_n:
                out.append(doc)
                continue

            node_id = str(doc.metadata["node_id"])
            parent = self._parent_doc_map.get(node_id)
            # 如果找不到父文档，就填入原始块
            if parent is None:
                out.append(doc)
                continue

            content = parent.page_content
            if len(content) > self.config.parent_doc_max_chars:
                content = (
                    content[:self.config.parent_doc_max_chars] + "…（父文档已截断）"
                )

            out.append(Document(
                page_content=content,
                metadata=dict(doc.metadata)
            ))

        return out    

    def _extract_relationships_from_graph(self)->List[Tuple[str,str,str]]:
        query = """
            MATCH (source)-[r]->(target)
            WHERE source.nodeId >= '200000000'
                OR target.nodeId >= '200000000'
            RETURN source.nodeId AS source_id,
                type(r) AS relation_type,
                target.nodeId AS target_id
            LIMIT 1000
        """

        relationships = []

        with self.driver.session() as session:
            for record in session.run(query):
                relationships.append((
                    record["source_id"],
                    record["relation_type"],
                    record["target_id"],
                ))

        return relationships

    def _build_graph_index(self):
        if self.graph_indexed:
            return

        self.graph_indexing.create_entity_key_values(
            recipes=self.data_module.recipes,
            ingredients=self.data_module.ingredients,
            cooking_steps=self.data_module.cooking_steps,
        )

        relationships = self._extract_relationships_from_graph()

        self.graph_indexing.create_relation_key_values(relationships)

        self.graph_indexed = True

    # 目前只进行查找实体
    def entity_level_retrieval(
        self,
        entity_keywords:List[str],
        top_k:int=5,
    )->List[RetrievalResult]:
        results = []
        # 根据关键字找实体
        for keyword in entity_keywords:
            entities = self.graph_indexing.get_entities_by_key(keyword)
            # 根据实体找邻居
            for entity in entities:
                neighbors =self._get_node_neighbors(
                    entity.metadata["node_id"],
                    max_neighbors=2
                )
                content = entity.value_content
                if neighbors:
                    content += f"\n相关信息: {', '.join(neighbors)}"
                # 两个type内容相同，只是存放位置不同
                results.append(RetrievalResult(
                    content=content,
                    node_id=entity.metadata["node_id"],
                    node_type=entity.entity_type,
                    # 目前还没有实现针对实体级检索的相似度算法
                    relevance_score=0.9,
                    retrieval_level="entity",
                    metadata={
                        "entity_name":entity.entity_name,
                        "entity_type": entity.entity_type,
                        "index_keys": entity.index_keys,
                        "matched_keyword": keyword,
                    }
                ))

        # 如果没有查到足够的结果，我们再进行一次兜底查询
        if len(results) < top_k:
            extra_results = self._neo4j_entity_level_search(
                entity_keywords,
                limit=top_k - len(results),
            )
            results.extend(extra_results)

        results.sort(
            key=lambda result: result.relevance_score,
            reverse=True,
        )

        return results[:top_k]

    
    # 在精确匹配不足时，调用Neo4j全文索引补充菜谱(这个是我们导入数据时，其脚本自己创建的索引)
    def _neo4j_entity_level_search(
        self,
        keywords:List[str],
        limit:int,
    ) -> List[RetrievalResult]:
        query = """
            UNWIND $keywords AS keyword
            CALL db.index.fulltext.queryNodes(
                'recipe_fulltext_index',
                '"' + keyword + '"'
            )
            YIELD node, score
            WHERE node:Recipe
            AND node.nodeId <> '200000000'
            RETURN node.nodeId AS node_id,
                node.name AS name,
                node.description AS description,
                labels(node) AS labels,
                score
            ORDER BY score DESC
            LIMIT $limit
        """

        results = []
        with self.driver.session() as session:
            records = session.run(
                query,
                {"keywords": keywords, "limit": limit},
            )

            for record in records:
                content_parts = []
                if record["name"]:
                    content_parts.append(f"菜品: {record['name']}")
                if record["description"]:
                    content_parts.append(f"描述: {record['description']}")
                results.append(RetrievalResult(
                    content="\n".join(content_parts),
                    node_id=record["node_id"],
                    node_type="Recipe",
                    relevance_score=float(record["score"]) * 0.7,   # 对兜底结果进行降权
                                                                    # 但是全文分数乘 0.7 并不能保证它低于精确匹配的 0.9，因为全文分数没有被归一化。
                    retrieval_level="entity",
                    metadata={
                        "name": record["name"],
                        "labels": record["labels"],
                        "source": "neo4j_fallback",
                    },
                ))
        return results

    def topic_level_retrieval(
        self,
        topic_keywords:List[str],
        top_k:int=5,
    ) ->List[RetrievalResult]:
        results =[]

        for keyword in topic_keywords:
            relations = self.graph_indexing.get_relation_by_key(keyword)

            for relation in relations:
                # 该关系的源实体
                source = self.graph_indexing.entity_kv_store.get(relation.source_entity)
                # 该关系的尾实体
                target = self.graph_indexing.entity_kv_store.get(relation.target_entity)

                if source is None or target is None:
                    continue

                content_parts = [
                    f"主题: {keyword}",
                    relation.value_content,
                    f"相关菜品: {source.entity_name}",
                    f"相关信息: {target.entity_name}",
                ]

                if source.entity_type == "Recipe":
                    first_line = source.value_content.split("\n")[0]
                    content_parts.append(f"菜品详情: {first_line}")

                results.append(RetrievalResult(
                    content="\n".join(content_parts),
                    node_id=relation.source_entity,
                    node_type=source.entity_type,
                    relevance_score=0.95,
                    retrieval_level="topic",
                    metadata={
                        "relation_id": relation.relation_id,
                        "relation_type": relation.relation_type,
                        "source_name": source.entity_name,
                        "target_name": target.entity_name,
                        "matched_keyword": keyword,
                        "index_keys": relation.index_keys,
                    },
                ))

        # 主题检索的实体匹配分支 无论前面检索到的够不够 都会执行。
        for keyword in topic_keywords:
            entities = self.graph_indexing.get_entities_by_key(keyword)

            for entity in entities:
                if entity.entity_type == "Recipe":
                    content_parts = [
                        f"主题分类: {keyword}",
                        entity.value_content,
                    ]

                    results.append(RetrievalResult(
                        content="\n".join(content_parts),
                        node_id=entity.metadata["node_id"],
                        node_type=entity.entity_type,
                        relevance_score=0.85,
                        retrieval_level="topic",
                        metadata={
                            "entity_name": entity.entity_name,
                            "entity_type": entity.entity_type,
                            "matched_keyword": keyword,
                            "source": "category_match",
                        },
                    ))

        if len(results) < top_k:
            extra_results = self._neo4j_topic_level_search(
                topic_keywords,
                limit=top_k - len(results),
            )
            results.extend(extra_results)


        results.sort(
            key=lambda result: result.relevance_score,
            reverse=True,
        )

        return results[:top_k]

    def _neo4j_topic_level_search(
        self,
        keywords: List[str],
        limit: int,
    ) -> List[RetrievalResult]:
        query = """
        UNWIND $keywords AS keyword
        MATCH (r:Recipe)
        WHERE r.nodeId <> '200000000'
        AND (
            r.category CONTAINS keyword
            OR r.cuisineType CONTAINS keyword
            OR r.tags CONTAINS keyword
        )
        WITH r, keyword
        OPTIONAL MATCH (r)-[:REQUIRES]->(i:Ingredient)
        WITH r, keyword, collect(i.name)[0..3] AS ingredients
        RETURN r.nodeId AS node_id,
               r.name AS name,
               r.category AS category,
               r.cuisineType AS cuisine_type,
               r.difficulty AS difficulty,
               ingredients,
               keyword AS matched_keyword
        ORDER BY r.difficulty ASC, r.name
        LIMIT $limit
        """

        results = []
        with self.driver.session() as session:
            records = session.run(
                query,
                {"keywords": keywords, "limit": limit},
            )

            for record in records:
                content_parts = [f"菜品: {record['name']}"]

                if record["category"]:
                    content_parts.append(f"分类: {record['category']}")
                if record["cuisine_type"]:
                    content_parts.append(f"菜系: {record['cuisine_type']}")
                if record["difficulty"]:
                    content_parts.append(f"难度: {record['difficulty']}")
                if record["ingredients"]:
                    ingredients = ", ".join(record["ingredients"])
                    content_parts.append(f"主要食材: {ingredients}")

                results.append(RetrievalResult(
                    content="\n".join(content_parts),
                    node_id=record["node_id"],
                    node_type="Recipe",
                    relevance_score=0.75,
                    retrieval_level="topic",
                    metadata={
                        "name": record["name"],
                        "category": record["category"],
                        "cuisine_type": record["cuisine_type"],
                        "difficulty": record["difficulty"],
                        "matched_keyword": record["matched_keyword"],
                        "source": "neo4j_fallback",
                    },
                ))

        return results

    # 既做提取，也允许推测
    # 但是其不知道我们有什么索引词，所以有可能不能匹配成功
    # 突然想到一个事：1.有点像分词器，如果我们的索引词就是LLM生成，那使用相同的LLM查找，准确率会更高吧。
    def extract_query_keywords(
        self,
        query:str,
    )->Tuple[List[str],List[str]]:
        prompt = f"""
            作为烹饪知识助手，请分析查询并提取两个层次的关键词。

            查询：{query}

            提取规则：
            1. 实体级关键词：具体食材、菜品、工具、品牌等。
            对于抽象查询，可以推测相关的具体食材或菜品。
            2. 主题级关键词：烹饪主题、饮食风格、营养特点等。
            排除“推荐、介绍、制作、怎么做”等动作词。

            示例：
            查询：推荐几个减肥菜
            {{
                "entity_keywords": ["鸡胸肉", "西兰花", "水煮蛋", "胡萝卜", "黄瓜"],
                "topic_keywords": ["减肥", "低热量", "高蛋白", "低脂"]
            }}

            查询：川菜有什么特色
            {{
                "entity_keywords": ["麻婆豆腐", "宫保鸡丁", "水煮鱼", "辣椒", "花椒"],
                "topic_keywords": ["川菜", "麻辣", "香辣", "下饭菜"]
            }}

            只返回 JSON 对象，不要添加 Markdown 代码块或其他文字：
            {{
                "entity_keywords": ["实体关键词"],
                "topic_keywords": ["主题关键词"]
            }}
        """

        response = self.llm_client.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.1,
            max_tokens=2048,# 这里不要设置的太小，否则会导致json还没有闭合就被截断
            extra_body={"thinking": {"type": "disabled"}},
        )

        result = json.loads(response.choices[0].message.content.strip())

        return (
            result.get("entity_keywords",[]),
            result.get("topic_keywords",[]),
        )

    def dual_level_retrieval(
        self,
        query:str,
        top_k:int=5,
    )->List[Document]:
        entity_keywords, topic_keywords = self.extract_query_keywords(query)

        entity_results = self.entity_level_retrieval(
            entity_keywords, top_k
        )
        topic_results = self.topic_level_retrieval(
            topic_keywords, top_k
        )

        all_results = entity_results + topic_results
        all_results.sort(
            key=lambda result: result.relevance_score,
            reverse=True,
        )

        seen_nodes = set()
        unique_results = []

        for result in all_results:
            if result.node_id not in seen_nodes:
                seen_nodes.add(result.node_id)
                unique_results.append(result)

        documents = []
        for result in unique_results[:top_k]:
            recipe_name = (
                result.metadata.get("name")
                or result.metadata.get("entity_name","未知菜品")
            )

            documents.append(Document(
                page_content=result.content,
                metadata={
                    "node_id":result.node_id,
                    "node_type":result.node_type,
                    "retrieval_level":result.retrieval_level,
                    "relevance_score":result.relevance_score,
                    "recipe_name":recipe_name,
                    "search_type":"dual_level",
                    **result.metadata
                },
            ))

        return documents
