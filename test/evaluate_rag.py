import os
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from test_dataset import dataset
import warnings

load_dotenv()

# Suppress warnings that might clutter the output
warnings.filterwarnings("ignore")

llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0)
embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')

def evaluate_RAG_system():
	try:
		print("Starting RAG evaluation...")
		print(f"Dataset size: {len(dataset)}")
		
		# Validate dataset structure
		required_keys = ['question', 'contexts', 'answer', 'ground_truth']
		for i, item in enumerate(dataset):
			for key in required_keys:
				if key not in item:
					print(f"Warning: Missing key '{key}' in item {i}")
				elif key == 'contexts' and not isinstance(item[key], list):
					print(f"Warning: 'contexts' should be a list in item {i}")
				elif key == 'contexts' and len(item[key]) == 0:
					print(f"Warning: Empty contexts list in item {i}")

		result = evaluate(
			dataset=dataset,
			metrics=[
				faithfulness,
				answer_relevancy,
				context_precision,
				context_recall
			],
			llm=llm,
			embeddings=embeddings,
			raise_exceptions=False  # Don't raise exceptions, continue evaluation
		)

		print("Evaluation completed successfully!")
		print(result)
		
	except Exception as e:
		print(f"Error during evaluation: {e}")
		print("This might be due to dataset structure issues or API problems.")
		# Try with a simpler evaluation
		try:
			print("Attempting simplified evaluation with just faithfulness...")
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
