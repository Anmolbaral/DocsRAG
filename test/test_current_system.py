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
		"What technologies and tools did Anmol work with?",
		"What achievements or impact did Anmol have in his roles?"
	]

	for i, question in enumerate(questions, 1):
		answer = system.ragPipeline.ask(question)
		assert answer is not None and answer.strip() != ""

	evaluate_RAG_system()
	print("Evaluation complete")


if __name__ == "__main__":
	test_current_system()
	