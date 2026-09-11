# recipe-graph-rag

从零复现 Datawhale all-in-rag C9，构建基于 Neo4j、Milvus 与混合检索的菜谱问答系统，记录实现、评测与优化过程。

## 当前进度
第一步：与 C9 对齐的文件结构、独立环境、配置加载与脱敏、配置验证测试。
尚未实现数据库连接、检索和大模型问答。

使用方法和学习说明见 [第一步：工程基础](docs/01-foundation.md)。

## 文件结构
本仓库根目录对应原项目的 `code/C9/`：

```text
recipe-graph-rag/
├── main.py
├── config.py
├── requirements.txt
├── .env.example
├── rag_modules/
│   ├── __init__.py
│   ├── graph_data_preparation.py
│   ├── milvus_index_construction.py
│   ├── graph_indexing.py
│   ├── hybrid_retrieval.py
│   ├── graph_rag_retrieval.py
│   ├── intelligent_query_router.py
│   └── generation_integration.py
├── tests/
├── docs/
└── data/
```

`rag_modules` 当前为模块占位，尚未实现检索功能；后续按原项目模块名逐步实现。
`tests/`、`docs/`、`data/` 是独立复现仓库的辅助目录。
原 C9 的 `agent(代码系ai生成)/` 是数据构建辅助工具，进入该阶段再添加。

## 来源
参考项目：[Datawhale all-in-rag](https://github.com/datawhalechina/all-in-rag)，重点为 code/C9 和 docs/chapter9。
本仓库当前基础工程为重新编写，尚未复制上游代码或数据。
后续引入上游材料时记录来源、修改范围，并遵循其许可要求；上游 README 声明 CC BY-NC-SA 4.0。
