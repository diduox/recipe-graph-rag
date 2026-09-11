# 第一步：工程基础与配置

## 目的
本阶段只验证本地工程和配置格式，不连接 Neo4j、Milvus 或大模型。
Python 3.11 是当前本机环境；后续数据库和模型依赖逐步安装、验证。

## 文件职责
- `requirements.txt`：按 C9 的方式管理依赖；当前只安装配置读取依赖。
- `rag_modules/`：与 C9 同名的业务模块，目前为占位文件，后续逐步实现。
- `.env.example`：可提交的配置模板，密码和模型名称留空等待实际配置。
- `.env`：本地实际配置，Git 忽略。
- `config.py`：默认值 < 指定 .env 文件 < 系统环境变量，检查端口、URL 和 top_k。
- `main.py`：统一命令入口，输出脱敏配置。
- `tests/`：验证配置优先级、错误输入和敏感字段脱敏。
- `data/raw/`、`data/processed/`：原始与处理后数据，暂不提交数据内容。

## PowerShell 操作
在仓库根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe main.py check-config
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

已有 .env 时不要重复覆盖。直接调用虚拟环境解释器，无需更改 PowerShell 执行策略。
默认从当前目录读 .env；在其他目录运行时使用 `--env-file` 指定文件。
格式校验通过不代表数据库可达、凭证有效或模型存在。下一步验证 Neo4j。

## 理解检查
1. 为什么公开仓库提交 .env.example，却不提交 .env？
2. 为什么部署时允许环境变量覆盖配置文件？
3. 为什么配置格式检查与服务连通性检查要分别做？

## 后续里程碑
Neo4j 数据建模与导入 → 向量检索基线 → BM25/RRF → 图检索与路由 → 对比评测。

## 与 C9 的对应关系
本仓库根目录对应上游 code/C9，直接运行 main.py，不再使用 src 包布局。
文件结构对齐不代表功能已完整复现，当前仅实现配置检查。
