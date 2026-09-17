# 菜谱 Graph RAG 问答系统

基于已有菜谱知识图谱构建的问答项目，使用 Neo4j 保存实体和关系，使用 Milvus 保存文本块向量，通过 DeepSeek API 生成回答。

支持混合检索、图检索和组合检索，并可自动选择策略。回答中的证据编号对应本次检索返回的文档。

## 运行环境

- Python 3.11（开发环境为 3.11.4）
- Docker Desktop，使用 Linux 容器
- 可用的 DeepSeek API Key

以下命令均在仓库根目录的 PowerShell 中执行。

## 首次启动

### 1. 安装 Python 依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. 配置 API Key

首次配置时，将 `.env.example` 复制为 `.env`：

```powershell
Copy-Item .env.example .env
```

在 `.env` 中填写：

```ini
DEEPSEEK_API_KEY=你的API密钥
```

已有 `.env` 时直接编辑，不必重新复制。不要将真实密钥提交到 Git。

数据库连接、模型名称等配置位于 `config.py`。

### 3. 启动数据库

```powershell
docker compose up -d neo4j
docker compose -p recipe-milvus -f docker-compose.milvus.yml up -d
```

Neo4j 浏览器地址：http://localhost:7474/browser/

本地演示配置的用户名为 `neo4j`，密码为 `all-in-rag`。

### 4. 导入图谱

确认 `cypher` 目录包含：

```text
cypher/
├── neo4j_import.cypher
├── nodes.csv
└── relationships.csv
```

首次准备图谱时执行：

```powershell
docker compose run --rm neo4j-init
```

日常启动不需要重复导入。

### 5. 启动问答

数据库服务就绪后执行：

```powershell
.\.venv\Scripts\python.exe main.py
```

首次运行会下载嵌入模型。如果 Milvus 集合不存在，程序会计算文本向量、写入数据并建立索引；集合存在时直接加载。

当前逻辑通过集合是否存在判断是否需要构建，不负责自动同步数据变更或恢复中断的构建。

启动后输入问题，输入 `quit` 退出。

## 日常启动

```powershell
docker compose up -d neo4j
docker compose -p recipe-milvus -f docker-compose.milvus.yml up -d
.\.venv\Scripts\python.exe main.py
```

## 示例问题

- 咖喱炒蟹需要哪些食材，各用多少？
- 咖喱炒蟹中，椰浆和蛋清是在关火前还是关火后加入？
- 青蟹和椰浆可以通过哪些菜谱关联起来？
- 请通过菜谱与食材的连接，查找同时使用青蟹和椰浆、难度不超过3星的菜。

## 检索策略对照

```powershell
.\.venv\Scripts\python.exe compare_retrieval.py
```

该脚本针对代码中指定的问题，对照不同策略返回的证据与耗时。它用于观察具体案例，不代表完整的准确率评测。
## 模块分工

| 文件 | 职责 |
|---|---|
| `main.py` | 初始化系统、准备知识库、接收问题并组织问答流程 |
| `config.py` | 保存数据库连接、嵌入模型和检索参数等配置 |
| `rag_modules/graph_data_preparation.py` | 读取图谱，将菜谱组织为文档并切块 |
| `rag_modules/milvus_index_construction.py` | 文本向量化、向量写入、索引创建和向量检索 |
| `rag_modules/graph_indexing.py` | 构建实体与关系的内存关键词索引 |
| `rag_modules/hybrid_retrieval.py` | 融合向量、BM25、实体与主题检索结果，并回填父文档 |
| `rag_modules/graph_rag_retrieval.py` | 规划图查询，检索多跳路径或局部子图，并转换为文本证据 |
| `rag_modules/intelligent_query_router.py` | 分析问题、选择检索策略、执行组合检索 |
| `rag_modules/generation_integration.py` | 调用 DeepSeek，根据检索证据生成带引用编号的回答 |
| `compare_retrieval.py` | 对照同一问题在不同策略下的检索结果 |

## 数据与问答流程

本项目使用已有的结构化菜谱图谱，没有实现从原始文章中自动抽取实体和关系的流程。

知识库准备流程：

```text
CSV 数据 → Neo4j 图谱
             ↓
        菜谱文档与文本块
             ↓
        向量化 → Milvus
```

Neo4j 保存实体、属性与关系，Milvus 保存用于语义检索的文本块向量。两者不要求每个节点一一对应。

问答流程：

```text
用户问题
   ↓
查询路由
   ├── 混合检索
   ├── 图检索
   └── 组合检索
   ↓
组织证据上下文
   ↓
DeepSeek 生成回答并标注证据编号
```

## 三种检索策略

### 混合检索：`hybrid_traditional`

融合向量检索、BM25，以及实体与主题层级检索，通过 RRF 合并排名，并可回填完整菜谱正文。

适用于食材、用量、制作步骤等问题。

该策略内部也使用图谱相关信息，不能视为纯向量检索基线。

### 图检索：`graph_rag`

通过实体间的多跳路径或围绕中心实体的局部子图获取证据。

例如，查找“青蟹”和“椰浆”经由哪道菜谱连接。多跳检索支持按关系类型过滤，并支持菜谱难度上限 `max_difficulty`。

### 组合检索：`combined`

分别执行混合检索和图检索，将两路结果交替合并并去重。

适用于同时需要关系路径与菜谱正文的问题。当前组合方式不是统一评分后的重排序，也不保证效果一定优于单一路径。

## 当前能力边界

- 当前图查询主要实现多跳路径与局部子图检索，没有实现专门的最短路径、因果推理或动作时序推理算法。
- 难度上限过滤只在独立图检索的多跳分支中实现，尚未覆盖所有检索策略和查询类型。
- 向量集合存在时直接复用，没有实现数据变更后的自动增量同步。
- 当前实体缓存和关系索引存在数量上限，不能将其理解为完整图谱的全部内存索引。
- 交互界面支持连续输入，但没有将历史问答作为对话记忆传递给模型。
- 证据编号用于追溯本次回答的来源，不代表引用内容经过自动正确性验证。
- 空结果表示本次检索没有找到证据，不等于已证明整个知识库中不存在答案。
- 已进行若干案例验证，尚未完成固定评测集上的准确率、召回率及策略消融评测。

## 已观察到的案例

- 从完整菜谱中读取食材用量。
- 根据步骤原文判断椰浆和蛋清的加入顺序。
- 对照食材表和步骤，指出姜用量不一致。
- 查找青蟹与椰浆之间的菜谱连接路径。
- 在多跳查询中按难度上限过滤菜谱。
- 将图路径与菜谱正文共同用于生成带证据引用的回答。

这些案例用于说明已经跑通的功能，不作为整体性能保证。