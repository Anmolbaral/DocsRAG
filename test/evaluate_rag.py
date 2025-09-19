import os
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from test_dataset import dataset

load_dotenv()

llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0)
embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')

def evaluate_RAG_system():

	result = evaluate(
		dataset= dataset,
		metrics = [
			faithfulness,
			answer_relevancy,
			context_precision,
			context_recall
		],
		llm = llm,
		embeddings = embeddings
	)

	print(result)

if __name__ == "__main__":
	evaluate_RAG_system()
