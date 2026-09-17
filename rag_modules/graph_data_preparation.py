from dataclasses import dataclass
from typing import Any, Dict, List
from neo4j import GraphDatabase
from langchain_core.documents import Document


"""
    两个类都是Python中的数据表示，和Neo4j无关
"""

@dataclass
class GraphNode:
    """ 
    从 Neo4j 读取的图节点
    
    示意节点：
    GraphNode(
    node_id="200000001",
    labels=["Ingredient"],
    name="土豆",
    properties={"nodeId": "200000001", "name": "土豆"},
    )
       
    """
    node_id:str
    labels:List[str]
    name:str
    properties:Dict[str,Any]


@dataclass
class GraphRelation:
    """图关系数据结构。"""

    start_node_id: str
    end_node_id: str
    relation_type: str
    properties: Dict[str, Any]

class GraphDataPreparationModule:
    """ 从 Neo4j 读取图数据并转换为文档 """
    # 初始化配置
    def __init__(
            self,
            uri:str,
            user:str,
            password:str,
            database:str = "neo4j"
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self._connect()
    # 底层驱动创建
    def _connect(self):
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user,self.password),
            database=self.database,
        )
    # 资源清理
    def close(self):
        self.driver.close()

    # 从Neo4j读取菜谱，并转换为GraphNode对象。
    def load_graph_data(self) -> Dict[str,Any]:
        # cypher查询语句
        recipes_query = """
        MATCH (r:Recipe)
        WHERE r.nodeId >= '200000000'
        AND r.nodeId <> '200000000'
        OPTIONAL MATCH (r)-[:BELONGS_TO_CATEGORY]->(c:Category)
        WITH r, collect(c.name) AS categories
        RETURN r.nodeId AS nodeId,
               labels(r) AS labels,
               r.name AS name,
               properties(r) AS originalProperties,
               CASE WHEN size(categories) > 0
                    THEN categories[0]
                    ELSE COALESCE(r.category, '未知')
               END AS mainCategory,
               CASE WHEN size(categories) > 0
                    THEN categories
                    ELSE [COALESCE(r.category, '未知')]
               END AS allCategories
        ORDER BY r.nodeId
        """
        ingredients_query = """
        MATCH (i:Ingredient)
        WHERE i.nodeId >= '200000000'
        RETURN  i.nodeId AS nodeId, 
                labels(i) AS labels,
                i.name AS name,
                properties(i) AS properties
                ORDER BY i.nodeId                  
        """
        steps_query = """
        MATCH (s:CookingStep)
        WHERE s.nodeId >= '200000000'
        RETURN s.nodeId AS nodeId, labels(s) AS labels,
            s.name AS name, properties(s) AS properties
        ORDER BY s.nodeId
        """

        self.recipes = []
        self.ingredients = []
        self.cooking_steps = []

        with self.driver.session() as session:

            for record in session.run(recipes_query):
                properties = dict(record["originalProperties"])
                properties["category"] = record["mainCategory"]
                properties["all_categories"] = record["allCategories"]
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=properties,
                )
                self.recipes.append(node)

            for record in session.run(ingredients_query):
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=record["properties"],
                )
                self.ingredients.append(node)

            for record in session.run(steps_query):
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=record["properties"],
                )
                self.cooking_steps.append(node)
            
            return {"recipes": len(self.recipes),
                    "ingredients": len(self.ingredients),
                    "cooking_steps": len(self.cooking_steps),
                    }

    def build_recipe_documents(self) -> List[Document]:
        ingredients_query = """
            MATCH (r:Recipe {nodeId: $recipe_id})-[req:REQUIRES]->(i:Ingredient)
            RETURN i.name AS name,
                req.amount AS amount,
                req.unit AS unit,
                i.description AS description
            ORDER BY i.name
        """
        steps_query = """
            MATCH (r:Recipe {nodeId: $recipe_id})-[c:CONTAINS_STEP]->(s:CookingStep)
            RETURN s.name AS name,
                s.description AS description,
                s.methods AS methods,
                s.tools AS tools,
                s.timeEstimate AS timeEstimate
            ORDER BY COALESCE(c.stepOrder, s.stepNumber, 999)
        """
    
        documents = []

        with self.driver.session() as session:
            # 遍历所有菜谱
            for recipe in self.recipes:

                # 当前这个菜谱的所有食材的info
                result = session.run(
                    ingredients_query,
                    {"recipe_id": recipe.node_id},
                )
                ingredients_info = []
                for record in result:
                   
                    ingredient_text = record["name"]
                    amount = record["amount"]
                    unit = record["unit"]

                    if amount and unit:
                        ingredient_text += f"({amount}{unit})"

                    if record["description"]:
                        ingredient_text += f" - {record['description']}"

                    ingredients_info.append(ingredient_text)

                # 当前这个食谱做法全流程的info
                steps_result = session.run(
                    steps_query,
                    {"recipe_id": recipe.node_id},
                )
                steps_info = []
                for record in steps_result:
                    step_text = f"步骤: {record['name']}"

                    if record["description"]:
                        step_text += f"\n描述: {record['description']}"
                    if record["methods"]:
                        step_text += f"\n方法: {record['methods']}"
                    if record["tools"]:
                        step_text += f"\n工具: {record['tools']}"
                    if record["timeEstimate"]:
                        step_text += f"\n时间: {record['timeEstimate']}"

                    steps_info.append(step_text)

                content_parts = [f"# {recipe.name}"]

                if recipe.properties.get("description"):
                    content_parts.append(
                        f"\n## 菜品描述\n{recipe.properties['description']}"
                    )

                if recipe.properties.get("cuisineType"):
                    content_parts.append(
                        f"\n菜系: {recipe.properties['cuisineType']}"
                    )

                if recipe.properties.get("difficulty"):
                    content_parts.append(
                        f"难度: {recipe.properties['difficulty']}星"
                    )

                time_info = []
                if recipe.properties.get("prepTime"):
                    time_info.append(
                        f"准备时间: {recipe.properties['prepTime']}"
                    )

                if recipe.properties.get("cookTime"):
                    time_info.append(
                        f"烹饪时间: {recipe.properties['cookTime']}"
                    )

                if time_info:
                    content_parts.append(
                        f"\n时间信息: {', '.join(time_info)}"
                    )

                if recipe.properties.get("servings"):
                    content_parts.append(
                        f"份量: {recipe.properties['servings']}"
                    )    

                if ingredients_info:
                    content_parts.append("\n## 所需食材")
                    for index, ingredient in enumerate(ingredients_info, 1):
                        content_parts.append(f"{index}. {ingredient}")

                if steps_info:
                    content_parts.append("\n## 制作步骤")
                    for index, step in enumerate(steps_info, 1):
                        content_parts.append(
                            f"\n### 第{index}步\n{step}"
                        )

                if recipe.properties.get("tags"):
                    content_parts.append(
                        f"\n## 标签\n{recipe.properties['tags']}"
                    )

                full_content = "\n".join(content_parts)
                
                doc = Document(
                    page_content=full_content,
                    metadata={
                        "node_id": recipe.node_id,
                        "recipe_name": recipe.name,
                        "node_type": "Recipe",
                        "category": recipe.properties["category"],
                        "ingredients_count": len(ingredients_info),
                        "steps_count": len(steps_info),
                        "doc_type": "recipe",
                        "cuisine_type": recipe.properties.get("cuisineType", "未知"),
                        "difficulty": recipe.properties.get("difficulty", 0),
                        "prep_time": recipe.properties.get("prepTime", ""),
                        "cook_time": recipe.properties.get("cookTime", ""),
                        "servings": recipe.properties.get("servings", ""),
                        "content_length": len(full_content),
                    },
                )
                documents.append(doc)
        self.documents = documents
        return documents

    def chunk_documents(
            self,
            chunk_size: int = 500,
            chunk_overlap: int = 50,
    ) -> List[Document]:
        chunks = []
        chunk_id = 0
        for doc in self.documents:
            content = doc.page_content
            # 对于短文档
            if len(content) <= chunk_size:
                chunk = Document(
                    page_content=content,
                    metadata={
                        **doc.metadata,
                        "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                        "parent_id": doc.metadata["node_id"],
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "chunk_size": len(content),
                        "doc_type": "chunk",
                    },
                )
                chunks.append(chunk)
                chunk_id += 1
            else:
                # 对于长文档：有二级标题就按照章节切，没有就按照字符长度切
                sections = content.split("\n## ")
                if len(sections) <= 1:
                    total_chunks = (
                        (len(content) - 1) // (chunk_size - chunk_overlap) + 1
                    )
                    for i in range(total_chunks):
                        start = i * (chunk_size - chunk_overlap)
                        end = min(start + chunk_size, len(content))
                        chunk_content = content[start:end]

                        chunk = Document(
                            page_content=chunk_content,
                            metadata={
                                **doc.metadata,
                                "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                                "parent_id": doc.metadata["node_id"],
                                "chunk_index": i,
                                "total_chunks": total_chunks,
                                "chunk_size": len(chunk_content),
                                "doc_type": "chunk",
                            },
                        )
                        chunks.append(chunk)
                        chunk_id += 1
                else:
                    total_chunks = len(sections)

                    for i, section in enumerate(sections):
                        if i == 0:
                            chunk_content = section
                        else:
                            chunk_content = f"## {section}"

                        chunk = Document(
                            page_content=chunk_content,
                            metadata={
                            **doc.metadata,
                            "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                            "parent_id": doc.metadata["node_id"],
                            "chunk_index": i,
                            "total_chunks": total_chunks,
                            "chunk_size": len(chunk_content),
                            "doc_type": "chunk",
                            "section_title": (
                            section.split("\n")[0] if i > 0 else "主标题"
                            ),
                            },
                            )
                        chunks.append(chunk)
                        chunk_id += 1

        self.chunks = chunks
        return chunks