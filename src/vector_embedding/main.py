from .system import DocumentRAGSystem

if __name__ == "__main__":
	system = DocumentRAGSystem()
	try:		
		system.initialize()
		print("System initialized successfully.")
	except Exception as e:
		print(f"System Initialization Error: {e}")