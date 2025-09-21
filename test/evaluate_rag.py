import os
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from test_dataset import dataset
import warnings

load_dotenv()

warnings.filterwarnings("ignore")

llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0)
embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')

def evaluate_RAG_system():
	print(dataset)
	for i, data in enumerate(dataset):
		print(f"Data: {data}")
		print(f"\n--- Testing Question {i+1} ---")
		print(f"Question: {data['question']}")
		print(f"Answer length: {len(data['answer'])}")
		print(f"Contexts count: {len(data['contexts'])}")
		try:
			result = evaluate(
				dataset=data,
				metrics=[
					faithfulness,
					answer_relevancy,
					context_precision,
					context_recall
				],
				llm=llm,
				embeddings=embeddings,
				raise_exceptions=False
			)
			print(result)
		
		except Exception as e:
			print(f"Error during evaluation: {e}")
			# Try with just faithfulness metric
			try:
				simple_result = evaluate(
					dataset=dataset,
					metrics=[faithfulness],
					llm=llm,
					embeddings=embeddings,
					raise_exceptions=False
				)
				print("Simplified evaluation result:")
				print(simple_result)
			except Exception as simple_e:
				print(f"Simplified evaluation also failed: {simple_e}")

if __name__ == "__main__":
	evaluate_RAG_system()