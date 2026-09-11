"""Load configuration without connecting to external services."""
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values


@dataclass(frozen=True)
class Settings:
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = field(default="", repr=False)
    neo4j_database: str = "neo4j"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_base_url: str = "https://api.moonshot.cn/v1"
    llm_model: str = ""
    llm_api_key: str = field(default="", repr=False)
    top_k: int = 5

    def __post_init__(self):
        if not 1 <= self.milvus_port <= 65535:
            raise ValueError("MILVUS_PORT must be between 1 and 65535")
        if self.top_k <= 0:
            raise ValueError("TOP_K must be positive")
        for key, value, schemes in (
            ("NEO4J_URI", self.neo4j_uri, {"bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"}),
            ("LLM_BASE_URL", self.llm_base_url, {"http", "https"}),
        ):
            parsed = urlparse(value)
            if parsed.scheme not in schemes or not parsed.hostname:
                raise ValueError(f"{key} has an invalid URL")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError(f"{key} must not contain credentials, query parameters or fragments")

    def safe_summary(self):
        result = asdict(self)
        for key in ("neo4j_password", "llm_api_key"):
            result[key] = "<configured>" if result[key] else "<not configured>"
        return result


def load_settings(env_file: Path | str = ".env") -> Settings:
    # Explicit path; no parent-directory search and no process environment mutation.
    values = {**dotenv_values(env_file, interpolate=False), **os.environ}
    kwargs = {}
    for name in Settings.__dataclass_fields__:
        key = name.upper()
        if key not in values:
            continue
        value = values[key]
        if value is None:
            raise ValueError(f"{key} must have a value")
        if name in {"milvus_port", "top_k"}:
            try:
                value = int(value)
            except ValueError:
                raise ValueError(f"{key} must be an integer") from None
        kwargs[name] = value
    return Settings(**kwargs)
