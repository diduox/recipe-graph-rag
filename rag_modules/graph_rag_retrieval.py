from dataclasses import dataclass
from enum import Enum
from typing import Any,Dict,List
from neo4j import GraphDatabase
import json
from langchain_core.documents import Document


class QueryType(Enum):
    ENTITY_RELATION = "entity_relation" # 实体之间有什么关系
    MULTI_HOP = "multi_hop"             # 沿多条关系寻找关联
    SUBGRAPH  = "subgraph"              # 获取某实体关系周围的相关节点和关系
    PATH_FINDING = "path_finding"       # 寻找两个实体之间的连接路径
    CLUSTERING = "clustering"           # 寻找相关或相似实体


@dataclass
class GraphQuery:
    query_type:QueryType
    source_entities:List[str]
    target_entities:List[str] = None
    relation_types:List[str] = None
    max_depth:int=2
    max_nodes:int=50
    constraints:Dict[str,Any] = None

# 多跳查询返回的结果
@dataclass
class GraphPath:
    nodes:List[Dict[str,Any]]
    relationships:List[Dict[str,Any]]
    path_length:int         # 多跳查询通常包含多个关系，这里记录关系数
    relevance_score:float
    path_type:str

# 子图查询返回的结果
@dataclass
class KnowledgeSubgraph:
    central_nodes:List[Dict[str,Any]]
    connected_nodes:List[Dict[str,Any]]
    relationships:List[Dict[str,Any]]
    graph_metrics:Dict[str,float]
    reasoning_chains:List[List[str]]

class GraphRAGRetrieval:
    def __init__(self,config,llm_client):
        self.config = config
        self.llm_client = llm_client
        self.driver = None

        self.entity_cache = {}
        self.relation_cache = {}
        self.subgraph_cache = {}

    def initialize(self):
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(
                self.config.neo4j_user,
                self.config.neo4j_password,
            ),
        )
        self._build_graph_index()

    def _build_graph_index(self):
        entity_query = """
        MATCH (n)
        WHERE n.nodeId IS NOT NULL
        WITH n, COUNT { (n)--() } AS degree
        RETURN labels(n) AS node_labels,
               n.nodeId AS node_id,
               n.name AS name,
               n.category AS category,
               degree
        ORDER BY degree DESC
        LIMIT 1000
        """

        relation_query = """
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(r) AS frequency
        ORDER BY frequency DESC
        """

        with self.driver.session() as session:
            for record in session.run(entity_query):
                self.entity_cache[record["node_id"]]={
                    "labels":record["node_labels"],  # Neo4j节点标签列表
                    "name":record["name"],
                    "category":record["category"],
                    "degree":record["degree"],
                }

            for record in session.run(relation_query):
                self.relation_cache[record["rel_type"]] = record["frequency"]

    def close(self):
        if self.driver:
            self.driver.close()

    def understand_graph_query(self,query:str)->GraphQuery:
        prompt = f"""
            分析烹饪问题，将它映射到已有图结构。

            节点类型：
            - Recipe：菜谱，含 name、category、cuisineType 等属性
            - Ingredient：食材，含 name、category 等属性
            - Category：分类
            - CookingStep：制作步骤

            主要关系：
            - (Recipe)-[:REQUIRES]->(Ingredient)
            - (Recipe)-[:BELONGS_TO_CATEGORY]->(Category)
            - (Recipe)-[:CONTAINS_STEP]->(CookingStep)

            查询类型：
            - entity_relation：实体间的关系
            - multi_hop：通过中间节点寻找关联
            - subgraph：获取实体周围的知识网络
            - path_finding：寻找实体之间的路径
            - clustering：寻找相关或相似实体

            规则：
            source_entities 使用可能存在于图中的实体名称。
            target_entities 仅在需要限制路径终点时填写，否则为 []。
            不要把时间限制、健康要求等抽象约束当成实体。
            relation_types 填写希望考虑的关系类型。
            max_depth 为 1 到 3 的整数。
            难度约束规则：
            - 用户明确要求难度上限时，使用：
            "constraints": {{"max_difficulty": 3}}
            - max_difficulty 必须是数字，表示难度小于或等于该值。
            - 没有明确难度上限时，不要猜测，返回 "constraints": {{}}。
            - 当前程序仅支持 max_difficulty，不要声称其他限制已经执行。

            问题：{query}

            只返回 JSON，不要使用 Markdown 代码块。示例：
            {{
                "query_type": "multi_hop",
                "source_entities": ["青蟹"],
                "target_entities": ["椰浆"],
                "relation_types": ["REQUIRES"],
                "max_depth": 2,
                "constraints": {{}}
            }}
    
        """

        response = self.llm_client.chat.completions.create(
            model = self.config.llm_model,
            messages = [{"role":"user","content":prompt}],
            temperature=0.1,
            max_tokens=self.config.max_tokens,
            extra_body={"thinking": {"type": "disabled"}},
        )

        result = json.loads(
            response.choices[0].message.content.strip()
        )


        return GraphQuery(
            query_type=QueryType(result.get("query_type","subgraph")),
            source_entities=result.get("source_entities",[]),
            target_entities=result.get("target_entities",[]),
            relation_types = result.get("relation_types",[]),
            max_depth=result.get("max_depth=",2),
            max_nodes=50,
            constraints=result.get("constraints") or {},
        )

    def _parse_neo4j_path(self,record)->GraphPath:
        path_nodes = []
        # 遍历迭代cypher取出来的节点列表
        for node in record["path_nodes"]:
            path_nodes.append({
                "id":node.get("nodeId",""),
                "name":node.get("name",""),
                "labels":list(node.labels),
                "properties":dict(node),
            })
        relationships = []
        for rel in record["rels"]:
            relationships.append({
                "type":rel.type,
                "properties":dict(rel),
            })

        return GraphPath(
            nodes = path_nodes,
            relationships= relationships,
            path_length=record["path_len"],
            relevance_score=record["relevance"],
            path_type="multi_hop",
        )

    def multi_hop_traversal(
        self,
        graph_query:GraphQuery,
    )-> List[GraphPath]:
        paths = []

        if graph_query.query_type == QueryType.MULTI_HOP:
            target_keywords = graph_query.target_entities or []
            max_depth = graph_query.max_depth
            constraints = graph_query.constraints or {}
            max_difficulty = constraints.get("max_difficulty")


            target_filter = ""
            if target_keywords:
                target_filter ="""
                AND ANY(kw IN $target_keywords WHERE
                    (target.name IS NOT NULL AND (
                        toString(target.name) CONTAINS kw
                        OR kw CONTAINS toString(target.name)
                    ))
                    OR
                    (target.category IS NOT NULL AND (
                        toString(target.category) CONTAINS kw
                        OR kw CONTAINS toString(target.category)
                    ))
                )
                """

            query = f"""
            UNWIND $source_entities AS source_name
            MATCH (source)
            WHERE source.name CONTAINS source_name
               OR source.nodeId = source_name

            MATCH path = (source)-[*1..{max_depth}]-(target)
            WHERE NOT source = target
            AND (
                size($relation_types) = 0
                OR ALL(rel IN relationships(path)
                        WHERE type(rel) IN $relation_types)
            )
            {target_filter}
            AND (
                $max_difficulty IS NULL
                OR (
                    ANY(n IN nodes(path) WHERE n:Recipe)
                    AND ALL(
                        n IN nodes(path) WHERE
                        NOT (n:Recipe)
                        OR (
                            n.difficulty IS NOT NULL
                            AND toFloat(n.difficulty) <= $max_difficulty
                        )
                    )
                )
            )
            WITH path,
                 length(path) AS path_len,
                 relationships(path) AS rels,
                 nodes(path) AS path_nodes

            WITH path_len, rels, path_nodes,
                 (1.0 / path_len)
                 + (
                     REDUCE(
                         s = 0.0, n IN path_nodes |
                         s + COUNT {{ (n)--() }}
                     ) / 10.0 / size(path_nodes)
                 )
                 + (
                     CASE
                         WHEN ANY(r IN rels WHERE
                             type(r) IN $relation_types
                         )
                         THEN 0.3
                         ELSE 0.0
                     END
                 ) AS relevance

            ORDER BY relevance DESC
            LIMIT 20
            RETURN path_nodes, rels, path_len, relevance
            """

            with self.driver.session() as session:
                records = session.run(
                            query,
                        {
                            "source_entities": graph_query.source_entities,
                            "target_keywords": target_keywords,
                            "relation_types": graph_query.relation_types or [],
                            "max_difficulty": max_difficulty,
                        },
                    )

                for record in records:
                    paths.append(self._parse_neo4j_path(record))

        return paths    

    def _build_knowledge_subgraph(self,record)->KnowledgeSubgraph:

        source = record["source"]
        central_nodes = [{
            **dict(source),
            "_labels":list(source.labels),
            "_element_id": source.element_id,
        }]

        connected_nodes = [
            {
                **dict(node),
                "_labels":list(node.labels),
                "_element_id": node.element_id,
            }
            for node in record["nodes"]
        ]

        relationships = [
            {
                "source_id":rel.start_node.element_id,
                "target_id":rel.end_node.element_id,
                "type":rel.type,
                "properties":dict(rel)
            }
            for rel in record["rels"]
        ]

        return KnowledgeSubgraph(
            central_nodes=central_nodes,
            connected_nodes=connected_nodes,
            relationships=relationships,
            graph_metrics=dict(record["metrics"]),
            reasoning_chains=[],
        )

    def extract_knowledge_subgraph(
        self,
        graph_query:GraphQuery,
    )->KnowledgeSubgraph:
        query = f""" 
        UNWIND $source_entities AS entity_name
        MATCH (source)
        WHERE source.name CONTAINS entity_name
           OR source.nodeId = entity_name

        MATCH (source)-[path_rels*1..{graph_query.max_depth}]-(neighbor)
        WHERE size($relation_types) = 0
            OR ALL(rel IN path_rels
                WHERE type(rel) IN $relation_types)
        WITH source,
             collect(DISTINCT neighbor) AS neighbors,
             collect(path_rels) AS relation_lists
        WHERE size(neighbors) <= $max_nodes

        UNWIND relation_lists AS relation_list
        UNWIND relation_list AS rel
        WITH source, neighbors, collect(DISTINCT rel) AS rels

        RETURN source,
               neighbors AS nodes,
               rels,
               {{
                   node_count: size(neighbors),
                   relationship_count: size(rels)
               }} AS metrics
        """

        with self.driver.session() as session:
            record = session.run(
                query,
                {
                    "source_entities": graph_query.source_entities,
                    "max_nodes": graph_query.max_nodes,   
                    "relation_types": graph_query.relation_types or [],
                },
            ).single()  # 意为只返回cypher查询结果的第一条数据

            if record is not None:
                return self._build_knowledge_subgraph(record)

        return KnowledgeSubgraph(
            central_nodes=[],
            connected_nodes=[],
            relationships=[],
            graph_metrics={},
            reasoning_chains=[],
        )

    def _build_subgraph_description(
        self,
        subgraph:KnowledgeSubgraph,
    )->str:
        central_names = [
           node.get("name","未知")
           for node in subgraph.central_nodes
       ]

        parts = [
           f"中心实体：{','.join(central_names)}",
           f"关联节点数：{len(subgraph.connected_nodes)}",
           f"关系数：{len(subgraph.relationships)}",
           "\n关联节点（未按关系类型分类或步骤顺序排列）："
       ]

        for node in subgraph.connected_nodes:
            labels = ",".join(node["_labels"])
            parts.append(f"- 名称: {node.get('name', '未知')}；类型: {labels}")

            if node.get("description"):
                parts.append(f"  描述: {node['description']}")

        node_names = {
            node["_element_id"]: node.get("name", "未知")
            for node in subgraph.central_nodes + subgraph.connected_nodes
        }

        nodes_by_id = {
            node["_element_id"]:node
            for node in subgraph.central_nodes + subgraph.connected_nodes
        }

        parts.append("\n关系（箭头表示数据库中的方向）:")

        ingredient_relations = [
            rel for rel in subgraph.relationships
            if rel["type"] == "REQUIRES"
        ]

        step_relations = [
            rel for rel in subgraph.relationships
            if rel["type"] == "CONTAINS_STEP"
        ]

        step_relations.sort(
            key=lambda rel:(
                rel["properties"].get("stepOrder")
                if rel["properties"].get("stepOrder") is not None
                else float("inf")
            )
        )

        other_relations = [
            rel for rel in subgraph.relationships
            if rel["type"] not in {"REQUIRES","CONTAINS_STEP"}
        ]

        for rel in ingredient_relations + step_relations + other_relations:
            source_name = node_names.get(
                rel["source_id"],str(rel["source_id"])
            )

            target_name = node_names.get(
                rel["target_id"],str(rel["target_id"])
            )

            parts.append(
                f"- {source_name} —[{rel['type']}]→ {target_name}"
            )

            if rel["properties"]:
                parts.append(f"关系属性：{rel['properties']}")

            if rel["type"] == "CONTAINS_STEP":
                step_node = nodes_by_id[rel["target_id"]]
                description = step_node.get("description")
                if description:
                    parts.append(f"步骤描述：{description}")


        return "\n".join(parts)



    def _subgraph_to_document(
        self,
        subgraph:KnowledgeSubgraph,
        reasoning_chains:List[str],
        query:str,
    )->List[Document]:
        return [
            Document(
            page_content=self._build_subgraph_description(subgraph),
            metadata={
                "search_type": "knowledge_subgraph",
                "node_count": len(subgraph.connected_nodes),
                    "relationship_count": len(subgraph.relationships),
                    "reasoning_chains": reasoning_chains,
                    "recipe_name": (
                        subgraph.central_nodes[0].get("name", "知识子图")
                        if subgraph.central_nodes else "知识子图"
                    ),
            },
        )
        ]


    def _build_path_description(self,path:GraphPath)->str:
        parts = []

        for i,node in enumerate(path.nodes):
            parts.append(node.get("name",f"节点{i}"))

            if i < len(path.relationships):
                relation_type = path.relationships[i].get("type","相关")
                parts.append(f" —[{relation_type}]— ")

        description ="".join(parts)
        recipe_details = []
        for node in path.nodes:
            if "Recipe" in node["labels"]:
                difficulty = node["properties"].get("difficulty")
                if difficulty is not None:
                    recipe_details.append(
                        f"菜谱属性：{node['name']},难度：{difficulty}星"
                    )

        if recipe_details:
            description += "\n" + "\n".join(recipe_details)

        return description

    def _paths_to_document(
        self,
        paths:List[GraphPath],
        query:str,
    )->List[Document]:
        documents = []

        for path in paths:
            documents.append(Document(
                page_content=self._build_path_description(path),
                metadata={
                    "search_type": "graph_path",
                    "path_length": path.path_length,
                    "relevance_score": path.relevance_score,
                    "path_type": path.path_type,
                    "node_count": len(path.nodes),
                    "relationship_count": len(path.relationships),
                                        "recipe_name": (
                        path.nodes[0].get("name", "图结构结果")
                        if path.nodes else "图结构结果"
                    ),
                },
            ))
        return documents

    # 图检索的统一入口
    def graph_rag_research(
        self,
        query:str,
        top_k:int=5,
    )->list[Document]:
        # 用LLM把自然语言询问变成图询问
        graph_query = self.understand_graph_query(query)
        print("图查询约束:", graph_query.constraints)
        print("图查询类型:", graph_query.query_type.value)

        results = []

        if graph_query.query_type == QueryType.MULTI_HOP:
            paths = self.multi_hop_traversal(graph_query)
            results = self._paths_to_document(paths,query)
        elif graph_query.query_type in [
            QueryType.SUBGRAPH,
            QueryType.CLUSTERING,
        ]:
            subgraph = self.extract_knowledge_subgraph(graph_query)
            results = self._subgraph_to_document(
                subgraph,
                reasoning_chains=[],
                query=query,
            )

        results = self._rank_by_graph_relevance(results,query)

        return results[:top_k]

    def _rank_by_graph_relevance(
        self,
        documents:List[Document],
        query:str,
    )->List[Document]:
        return sorted(
            documents,
            key=lambda doc: doc.metadata.get("relevance_score", 0.0),
            reverse=True
        )