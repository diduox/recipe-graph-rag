"""
图键值索引模块。

在内存中建立“索引词 → 实体或关系编号 → 描述”的映射，
方便通过关键词查找图数据。源数据更新后需主动同步。

当前仅定义实体记录结构，索引构建和查询逻辑待实现。
"""
from dataclasses import dataclass
from typing import Any,Dict,List,Tuple
# defaultdict 访问不存在的键时，自动创建一个默认值
from collections import defaultdict

"""
示例
EntityKeyValue(
    entity_name="咖喱炒蟹",
    index_keys=["咖喱炒蟹"],
    value_content="菜品名称: 咖喱炒蟹\n分类: 水产\n菜系: 泰国菜",
    entity_type="Recipe",
    metadata={"node_id": "201000001"},
)
"""
@dataclass
class EntityKeyValue:
    """ 用于实体检索的键值记录。 """

    entity_name:str
    # 这个实体的索引词
    index_keys:List[str]
    value_content:str
    entity_type:str
    metadata:Dict[str,Any]

@dataclass
class RelationKeyvalue:
    """ 用于关系检索的键值记录 """

    relation_id:str
    index_keys:List[str]
    value_content:str
    relation_type:str
    source_entity:str
    target_entity:str
    metadata:Dict[str,Any]

class GraphIndexingModule:
    def __init__(self):
        # 实体编号到实体 key-value
        """ 
        {
            "201000001": EntityKeyValue(...),
            "201000023": EntityKeyValue(...),
        }
        """
        self.entity_kv_store = {}
        # 名称到实体编号
        """
        {
            "咖喱炒蟹": ["201000001"],
            "土豆": ["编号A", "编号B"],
        }
        """
        self.key_to_entities = defaultdict(list)

        # 通过ID找关系实体
        self.relation_kv_store = {}
        # 通过关键词找关系实体的ID
        self.key_to_relations = defaultdict(list)


    def create_entity_key_values(
        self,
        recipes:List[Any],# 每个recipe就是一个GraphNode节点
        ingredients: List[Any],
        cooking_steps: List[Any],
    )->Dict[str,EntityKeyValue]:
        for recipe in recipes:
            entity_id = recipe.node_id
            entity_name = recipe.name
            props = recipe.properties

            content_parts = [f"菜品名称: {entity_name}"]
            if props.get("description"):
                content_parts.append(f"描述: {props['description']}")
            if props.get("category"):
                content_parts.append(f"分类: {props['category']}")
            if props.get("cuisineType"):
                content_parts.append(f"菜系: {props['cuisineType']}")
            if props.get("difficulty"):
                content_parts.append(f"难度: {props['difficulty']}")
            if props.get("cookingTime"):
                content_parts.append(f"制作时间: {props['cookingTime']}")

            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=[entity_name],
                value_content="\n".join(content_parts),
                entity_type="Recipe",
                metadata={
                    "node_id": entity_id,
                    "properties": props,
                },
            )

            self.entity_kv_store[entity_id] = entity_kv
            # 因为同一个名称可能对应不同实体 所以这里用列表
            self.key_to_entities[entity_name].append(entity_id)

        for ingredient in ingredients:
            entity_id = ingredient.node_id
            entity_name = ingredient.name
            props = ingredient.properties

            content_parts = [f"食材名称: {entity_name}"]
            if props.get("category"):
                content_parts.append(f"类别: {props['category']}")
            if props.get("nutrition"):
                content_parts.append(f"营养信息: {props['nutrition']}")
            if props.get("storage"):
                content_parts.append(f"储存方式: {props['storage']}")

            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=[entity_name],
                value_content="\n".join(content_parts),
                entity_type="Ingredient",
                metadata={
                    "node_id": entity_id,
                    "properties": props,
                },
            )

            self.entity_kv_store[entity_id] = entity_kv
            self.key_to_entities[entity_name].append(entity_id)
        # 每一步烹饪步骤在neo4j中都是单独的一个节点
        for step in cooking_steps:
            entity_id = step.node_id
            entity_name = f"步骤_{entity_id}"
            props = step.properties

            content_parts = [f"烹饪步骤: {entity_name}"]
            if props.get("description"):
                content_parts.append(f"步骤描述: {props['description']}")
            if props.get("order"):
                content_parts.append(f"步骤顺序: {props['order']}")
            if props.get("technique"):
                content_parts.append(f"技巧: {props['technique']}")
            if props.get("time"):
                content_parts.append(f"时间: {props['time']}")

            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=[entity_name],
                value_content="\n".join(content_parts),
                entity_type="CookingStep",
                metadata={
                    "node_id": entity_id,
                    "properties": props,
                },
            )
            
            self.entity_kv_store[entity_id] = entity_kv
            self.key_to_entities[entity_name].append(entity_id)

        return self.entity_kv_store

    # 这里的key是索引词,不是具体编号
    # 索引词 -> 编号列表 -> 实体记录列表
    def get_entities_by_key(self,key:str)->List[EntityKeyValue]:
        entity_ids = self.key_to_entities.get(key,[]) # 如果找不到，返回空列表

        return [
            self.entity_kv_store[entity_id]
            for entity_id in entity_ids
            if entity_id in self.entity_kv_store
        ]

    def _generate_relation_index_keys(
        self,
        source_entity:EntityKeyValue,
        target_entity:EntityKeyValue,
        relation_type:str
    ) -> List[str]:
        keys = [relation_type]

        # 如果使用append加入的就是整个列表，使用extend加入的是元素
        if relation_type == "REQUIRES":
            keys.extend([
                "食材搭配",
                "烹饪原料",
                f"{source_entity.entity_name}_食材",
                target_entity.entity_name,
            ])
        elif relation_type == "HAS_STEP":
            keys.extend([
                "制作步骤",
                "烹饪过程",
                f"{source_entity.entity_name}_步骤",
                "制作方法",
            ])
        elif relation_type == "BELONGS_TO_CATEGORY":
            keys.extend([
                "菜品分类",
                "美食类别",
                target_entity.entity_name,
            ])
        # set去重之后，再转换成列表
        return list(set(keys))

    def create_relation_key_values(
        self,
        relationships:List[Tuple[str,str,str]],
    )->Dict[str,RelationKeyvalue]:
        for i,(source_id,relation_type,target_id) in enumerate(relationships):

            relation_id = f"rel_{i}_{source_id}_{target_id}"

            source_entity = self.entity_kv_store.get(source_id)
            target_entity = self.entity_kv_store.get(target_id)

            if source_entity is None or target_entity is None:
                continue

            content_parts = [
                f"关系类型: {relation_type}",
                f"源实体: {source_entity.entity_name} ({source_entity.entity_type})",
                f"目标实体: {target_entity.entity_name} ({target_entity.entity_type})",
            ]

            index_keys = self._generate_relation_index_keys(
                source_entity,
                target_entity,
                relation_type,
            )

            relation_kv = RelationKeyvalue(
                relation_id=relation_id,
                index_keys=index_keys,
                value_content="\n".join(content_parts),
                relation_type=relation_type,
                source_entity=source_id,
                target_entity=target_id,
                metadata={
                    "source_name": source_entity.entity_name,
                    "target_name": target_entity.entity_name,
                    "created_from_graph": True,
                },
            )

            self.relation_kv_store[relation_id] = relation_kv

            for key in index_keys:
                self.key_to_relations[key].append(relation_id)

        return self.relation_kv_store

    def get_relation_by_key(self,key:str)->List[RelationKeyvalue]:
        relation_ids = self.key_to_relations.get(key,[])

        return [
            self.relation_kv_store[relation_id]
            for relation_id in relation_ids
            if relation_id in self.relation_kv_store
        ]