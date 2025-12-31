from dotenv import load_dotenv

load_dotenv()

# Lazy loading of DocumentRAGSystem
def __getattr__(name):
	if name == "DocumentRAGSystem":
		from .system import DocumentRAGSystem
		return DocumentRAGSystem
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["DocumentRAGSystem"]
