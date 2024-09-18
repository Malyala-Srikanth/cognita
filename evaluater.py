import os
import time
import httpx
import openai
from tqdm import tqdm

from typing import List
from elasticsearch import Elasticsearch
from llama_index.core import Document
from llama_index.core.evaluation import DatasetGenerator

from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

openai.api_key = os.environ.get("OPENAI_API_KEY")

class DeepEvaluator:
    def __init__(self, es_host: str, collection_name: str, num_eval_questions: int = 25):
        self.es_host = es_host
        self.collection_name = collection_name
        self.num_eval_questions = num_eval_questions
        self.es_vector_db = Elasticsearch(hosts=[es_host])

    def generate_eval_questions(self):
        print("Fetching documents from ES")
        self.documents = self._fetch_documents_from_es()
        self.documents = [Document(text=doc) for doc in self.documents]
        self.eval_documents = self.documents[0:20]

        print("Generating questions from nodes")
        self.data_generator = DatasetGenerator.from_documents(self.eval_documents)
        self.eval_questions = self.data_generator.generate_questions_from_nodes()
        self.k_eval_questions = self.eval_questions[:self.num_eval_questions]

    def _fetch_documents_from_es(self) -> List[dict]:
        query = {
            "query": {
                "match_all": {}
            }
        }
        documents = []
        response = self.es_vector_db.search(index=self.collection_name, body=query)
        for doc in response['hits']['hits']:
            documents.append(doc["_source"]['text'])
        return documents

    def initialize_evaluaters(self):
        print("Initializing evaluators")
        self.faithfulness_evaluator = FaithfulnessMetric(
            threshold=0.7,
            model="gpt-4o",
            include_reason=True
        )
        self.answerrelevancy_evaluator = AnswerRelevancyMetric(
            threshold=0.7,
            model="gpt-4o",
            include_reason=True
        )

    def get_rag_output(self, query):
        timeout = httpx.Timeout(10.0)
        # Define the URL for the endpoint
        url = "http://localhost:8000/retrievers/basic-rag/answer"
        # Define the request payload
        payload = {
            "collection_name": "simpplrassignment",
            "internet_search_enabled": False,
            "model_configuration": {
                "name": "openai/gpt-4o",
                "parameters": {
                    "temperature": 0.1
                }
            },
            "prompt_template": "You are an AI assistant specialising in information retrieval and analysis. Answer the following question based only on the given context:\nContext: {context} \nQuestion: {question}",
            "query": query,
            "retriever_config": {
                "search_type": "similarity",
                "search_kwargs": {
                    "k": 5
                }
            },
            "retriever_name": "vectorstore",
            "stream": False
        }
        # Define the headers
        headers = {
            "Content-Type": "application/json"
        }
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        return response.json()

    def evaluate(self):
        self.generate_eval_questions()
        self.initialize_evaluaters()

        test_cases = []
        for question in tqdm(self.k_eval_questions):
            time.sleep(5)
            rag_output = self.get_rag_output(question)
            test_case = LLMTestCase(
                input=question,
                actual_output=rag_output['answer'],
                retrieval_context=[doc['page_content'] for doc in rag_output['docs']]
            )
            test_cases.append(test_case)
        evaluate(test_cases, [self.faithfulness_evaluator, self.answerrelevancy_evaluator])


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set in the environment")
    evaluator = DeepEvaluator("http://localhost:9200", "simpplrassignment", 10)
    evaluator.evaluate()
