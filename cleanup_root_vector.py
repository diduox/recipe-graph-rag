from pymilvus import MilvusClient

from config import DEFAULT_CONFIG

if __name__ == "__main__":
    config = DEFAULT_CONFIG

    client = MilvusClient(
        uri = f"http://{config.milvus_host}:{config.milvus_port}"
    )

    try:
        client.load_collection(
            collection_name=config.milvus_collection_name,
        )

        result = client.delete(
            collection_name=config.milvus_collection_name,
            filter='node_id == "200000000"'
        )

        print("根节点向量清理结束：",result)

    finally:
        client.close()