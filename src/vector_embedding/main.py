from .system import DocumentRAGSystem
import sys

if __name__ == "__main__":
	system = DocumentRAGSystem()
	try:		
		system.initialize()
		if sys.stdin.isatty():
			system.user_interaction()
		else:
			print("System initialized. Skipping interactive mode (no TTY detected).")
	except Exception as e:
		print(f"System Initialization Error: {e}")