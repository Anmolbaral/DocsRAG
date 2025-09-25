import os
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from test_dataset import dataset
from vector_embedding.system import DocumentRAGSystem
import warnings

load_dotenv()

warnings.filterwarnings("ignore")

llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0)
embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')

def evaluate_RAG_system():
	# Initialize RAG system
	system = DocumentRAGSystem()
	system.initialize()
	
	# Generate answers using your RAG system
	evaluation_data = []
	for data in dataset:
		# Get answer from your RAG system
		generated_answer = system.ragPipeline.ask(data['question'])
		
		# Create evaluation record
		eval_record = {
			'question': data['question'],
			'contexts': [str(context) for context in data['contexts']],  
			'answer': generated_answer,  
			'ground_truth': str(data['ground_truth'])  
		}
		evaluation_data.append(eval_record)
	
	# Convert to Dataset
	eval_dataset = Dataset.from_list(evaluation_data)

	try:
		# First, try with stable metrics only (excluding answer_relevancy which causes IndexError)
		result = evaluate(
			dataset=eval_dataset,
			metrics=[
				faithfulness,
				context_precision,
				context_recall,
				answer_relevancy
			],
			llm=llm,
			embeddings=embeddings,
			raise_exceptions=False,
			batch_size=1,  # Process one at a time to avoid parallel processing issues
			show_progress=True
		)
		print("\n=== RAGAS EVALUATION RESULTS ===")
		print(result)
		
	except Exception as e:
		print(f"Error during evaluation: {e}")
		# Try with just faithfulness metric
		try:
			simple_result = evaluate(
				dataset=eval_dataset,
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