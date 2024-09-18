from typing import List
from urllib.parse import urlparse

from elasticsearch import Elasticsearch, helpers
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore

from backend.constants import DATA_POINT_FQN_METADATA_KEY, DATA_POINT_HASH_METADATA_KEY
from backend.logger import logger
from backend.modules.vector_db.base import BaseVectorDB
from backend.types import DataPointVector, ElasticsearchClientConfig, VectorDBConfig

BATCH_SIZE = 1000

class ElasticsearchVectorDB(BaseVectorDB):
    def __init__(self, config: VectorDBConfig):
        logger.debug(f"Connecting to Elasticsearch using config: {config.model_dump()}")
        self.url = config.url

        self.elasticsearch_kwargs = ElasticsearchClientConfig.model_validate(config.config or {})
        if self.url and (self.url.startswith("http://") or self.url.startswith("https://")):
            if self.elasticsearch_kwargs.port is None:
                port = urlparse(self.url).port
                if port is None:
                    self.elasticsearch_kwargs.port = 9200
                else:
                    self.elasticsearch_kwargs.port = port
            else:
                port = urlparse(self.url).port
                if port is None:
                    self.url = f"{self.url.strip('/')}:{self.elasticsearch_kwargs.port}"

        self.es_client = Elasticsearch(
            self.url,
        )

    def create_collection(self, collection_name: str, embeddings: Embeddings):
        logger.debug(f"[Elasticsearch] Creating new index {collection_name}")

        # Calculate embedding size
        logger.debug(f"[Elasticsearch] Embedding a dummy doc to get vector dimensions")
        partial_embeddings = embeddings.embed_documents(["Initial document"])
        vector_size = len(partial_embeddings[0])
        logger.debug(f"Vector size: {vector_size}")

        index_body = {
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "dense_vector",
                        "dims": vector_size
                    },
                    f"metadata.{DATA_POINT_FQN_METADATA_KEY}": {
                        "type": "keyword"
                    }
                }
            }
        }

        self.es_client.indices.create(index=collection_name, body=index_body)
        logger.debug(f"[Elasticsearch] Created new index {collection_name}")

    def _get_records_to_be_upserted(
        self, collection_name: str, data_point_fqns: List[str], incremental: bool
    ):
        if not incremental:
            return []
        # For incremental deletion, we delete the documents with the same document_id
        logger.debug(
            f"[Elasticsearch] Incremental Ingestion: Fetching documents for {len(data_point_fqns)} data point fqns for index {collection_name}"
        )
        query = {
            "query": {
                "terms": {
                    f"metadata.{DATA_POINT_FQN_METADATA_KEY}": data_point_fqns
                }
            }
        }
        response = self.es_client.search(index=collection_name, body=query, size=BATCH_SIZE)
        record_ids_to_be_upserted = [hit['_id'] for hit in response['hits']['hits']]
        logger.debug(
            f"[Elasticsearch] Incremental Ingestion: index={collection_name} Addition={len(data_point_fqns)}, Updates={len(record_ids_to_be_upserted)}"
        )
        return record_ids_to_be_upserted

    def upsert_documents(
        self,
        collection_name: str,
        documents,
        embeddings: Embeddings,
        incremental: bool = True,
    ):
        if len(documents) == 0:
            logger.warning("No documents to index")
            return
        # get record IDs to be upserted
        logger.debug(
            f"[Elasticsearch] Adding {len(documents)} documents to index {collection_name}"
        )
        data_point_fqns = [doc.metadata.get(DATA_POINT_FQN_METADATA_KEY) for doc in documents]
        record_ids_to_be_upserted: List[str] = self._get_records_to_be_upserted(
            collection_name=collection_name,
            data_point_fqns=data_point_fqns,
            incremental=incremental,
        )

        # Add Documents
        ElasticsearchStore(
            es_url=self.url,
            es_user=self.elasticsearch_kwargs.username,
            es_password=self.elasticsearch_kwargs.password,
            index_name=collection_name,
            embedding=embeddings,
        ).add_documents(documents=documents)
        logger.debug(
            f"[Elasticsearch] Added {len(documents)} documents to index {collection_name}"
        )

        # Delete Documents
        if len(record_ids_to_be_upserted):
            logger.debug(
                f"[Elasticsearch] Deleting {len(record_ids_to_be_upserted)} outdated documents from index {collection_name}"
            )
            delete_actions = [
                {
                    "_op_type": "delete",
                    "_index": collection_name,
                    "_id": record_id
                }
                for record_id in record_ids_to_be_upserted
            ]
            helpers.bulk(self.es_client, delete_actions)
            logger.debug(
                f"[Elasticsearch] Deleted {len(record_ids_to_be_upserted)} outdated documents from index {collection_name}"
            )

    def get_collections(self) -> List[str]:
        logger.debug(f"[Elasticsearch] Fetching indices")
        indices = self.es_client.indices.get_alias("*")
        logger.debug(f"[Elasticsearch] Fetched {len(indices)} indices")
        return list(indices.keys())

    def delete_collection(self, collection_name: str):
        logger.debug(f"[Elasticsearch] Deleting {collection_name} index")
        self.es_client.indices.delete(index=collection_name)
        logger.debug(f"[Elasticsearch] Deleted {collection_name} index")

    def get_vector_store(self, collection_name: str, embeddings: Embeddings):
        logger.debug(f"[Elasticsearch] Getting vector store for index {collection_name}")
        return ElasticsearchStore(
            es_url=self.url,
            es_user=self.elasticsearch_kwargs.username,
            es_password=self.elasticsearch_kwargs.password,
            index_name=collection_name,
            embedding=embeddings,
        )

    def get_vector_client(self):
        logger.debug(f"[Elasticsearch] Getting Elasticsearch client")
        return self.es_client

    def list_data_point_vectors(
        self, collection_name: str, data_source_fqn: str, batch_size: int = BATCH_SIZE
    ) -> List[DataPointVector]:
        logger.debug(
            f"[Elasticsearch] Listing all data point vectors for index {collection_name}"
        )
        query = {
            "query": {
                "match": {
                    f"metadata.{DATA_POINT_FQN_METADATA_KEY}": data_source_fqn
                }
            }
        }
        response = self.es_client.search(index=collection_name, body=query, size=batch_size)
        data_point_vectors = [
            DataPointVector(
                data_point_vector_id=hit['_id'],
                vector=hit['_source']['vector'],
                metadata=hit['_source']['metadata']
            )
            for hit in response['hits']['hits']
        ]
        logger.debug(
            f"[Elasticsearch] Listing {len(data_point_vectors)} data point vectors for index {collection_name}"
        )
        return data_point_vectors

    def delete_data_point_vectors(
        self,
        collection_name: str,
        data_point_vectors: List[DataPointVector],
        batch_size: int = BATCH_SIZE,
    ):
        """
        Delete data point vectors from the index
        """
        logger.debug(f"[Elasticsearch] Deleting {len(data_point_vectors)} data point vectors")
        vectors_to_be_deleted_count = len(data_point_vectors)
        deleted_vectors_count = 0
        for i in range(0, vectors_to_be_deleted_count, batch_size):
            data_point_vectors_to_be_processed = data_point_vectors[i : i + batch_size]
            delete_actions = [
                {
                    "_op_type": "delete",
                    "_index": collection_name,
                    "_id": vector.data_point_vector_id
                }
                for vector in data_point_vectors_to_be_processed
            ]
            helpers.bulk(self.es_client, delete_actions)
            deleted_vectors_count += len(data_point_vectors_to_be_processed)
            logger.debug(
                f"[Elasticsearch] Deleted [{deleted_vectors_count}/{vectors_to_be_deleted_count}] data point vectors"
            )
        logger.debug(
            f"[Elasticsearch] Deleted {vectors_to_be_deleted_count} data point vectors"
        )

    def list_documents_in_collection(
        self, collection_name: str, base_document_id: str = None
    ) -> List[str]:
        """
        List all documents in an index
        """
        logger.debug(
            f"[Elasticsearch] Listing all documents with base document id {base_document_id} for index {collection_name}"
        )
        query = {
            "query": {
                "match": {
                    f"metadata.{DATA_POINT_FQN_METADATA_KEY}": base_document_id
                }
            }
        } if base_document_id else {"query": {"match_all": {}}}
        response = self.es_client.search(index=collection_name, body=query, size=BATCH_SIZE)
        document_ids = [hit['_id'] for hit in response['hits']['hits']]
        logger.debug(
            f"[Elasticsearch] Found {len(document_ids)} documents with base document id {base_document_id} for index {collection_name}"
        )
        return document_ids

    def delete_documents(self, collection_name: str, document_ids: List[str]):
        """
        Delete documents from the index
        """
        logger.debug(
            f"[Elasticsearch] Deleting {len(document_ids)} documents from index {collection_name}"
        )
        delete_actions = [
            {
                "_op_type": "delete",
                "_index": collection_name,
                "_id": doc_id
            }
            for doc_id in document_ids
        ]
        helpers.bulk(self.es_client, delete_actions)
        logger.debug(
            f"[Elasticsearch] Deleted {len(document_ids)} documents from index {collection_name}"
        )

    def list_document_vector_points(
        self, collection_name: str
    ) -> List[DataPointVector]:
        """
        List all documents in an index
        """
        logger.debug(
            f"[Elasticsearch] Listing all document vector points for index {collection_name}"
        )
        query = {"query": {"match_all": {}}}
        response = self.es_client.search(index=collection_name, body=query, size=BATCH_SIZE)
        document_vector_points = [
            DataPointVector(
                data_point_vector_id=hit['_id'],
                vector=hit['_source']['vector'],
                metadata=hit['_source']['metadata']
            )
            for hit in response['hits']['hits']
        ]
        logger.debug(
            f"[Elasticsearch] Listing {len(document_vector_points)} document vector points for index {collection_name}"
        )
        return document_vector_points
