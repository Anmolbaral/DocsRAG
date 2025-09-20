from evaluate_rag import evaluate_RAG_system
from vector_embedding.system import DocumentRAGSystem

def test_current_system():
	system = DocumentRAGSystem()
	system.initialize()

	print("Evaluating current system...")
	questions = [
		"What is Anmol's work experience?",
		"What did he do at Apple?",
		"How long was he at Apple?",
	]

	for i, question in enumerate(questions, 1):
		answer = system.ragPipeline.ask(question)
		print(f"Question {i}: {question}")
		print(f"Answer {i}: {answer}")
		print("-"*100)

	evaluate_RAG_system()
	print("Evaluation complete")


if __name__ == "__main__":
	test_current_system()
	