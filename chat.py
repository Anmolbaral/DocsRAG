#!/usr/bin/env python3
"""
Interactive terminal chat interface for Document RAG System
Run with: python chat.py
"""

import sys
from src.vector_embedding.system import DocumentRAGSystem

def main():
    print("=" * 60)
    print("🚀 Document RAG System - Interactive Chat")
    print("=" * 60)
    print("\nInitializing system...")
    
    try:
        system = DocumentRAGSystem()
        system.initialize()
        
        print("✅ System ready! You can now ask questions.")
        print("\nCommands:")
        print("  - Type your question and press Enter")
        print("  - Type 'quit' or 'exit' to exit")
        print("  - Type 'clear' to clear the screen")
        print("-" * 60)
        
        while True:
            try:
                # Get user input
                query = input("\n💬 You: ").strip()
                
                # Handle commands
                if not query:
                    continue
                elif query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                elif query.lower() == 'clear':
                    import os
                    os.system('clear' if os.name != 'nt' else 'cls')
                    continue
                
                # Process query
                print("\n🤔 Thinking...")
                response = system.query(query)
                
                # Display response
                print("\n🤖 Assistant:")
                print("-" * 60)
                print(response)
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                sys.exit(0)
            except Exception as e:
                print(f"\n Error: {e}")
                print("Please try again or type 'quit' to exit.")
                
    except Exception as e:
        print(f"\n Failed to initialize system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

