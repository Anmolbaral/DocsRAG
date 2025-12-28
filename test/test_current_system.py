from vector_embedding.system import DocumentRAGSystem
import numpy as np


class FakeEmbedder:
    def __init__(self, dim=1536):
        self.dim = dim

    def _vector_for(self, text):
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.random(self.dim, dtype=np.float32)
        return vec

    def get_embedding(self, text):
        return self._vector_for(text)

    def get_embeddings(self, texts):
        return [self._vector_for(t) for t in texts]


def fake_chat_client(messages):
    # messages contain a system prompt, a document context, prior turns, and the user query
    doc_line = next(
        (
            m
            for m in messages
            if m.get("role") == "user"
            and str(m.get("content", "")).startswith("Document Context:")
        ),
        None,
    )
    query_line = next(
        (
            m
            for m in messages
            if m.get("role") == "user"
            and not str(m.get("content", "")).startswith("Document Context:")
        ),
        None,
    )
    context_preview = ""
    if doc_line:
        content = str(doc_line["content"])
        context_preview = content[:120]
    query_text = query_line["content"] if query_line else ""
    return f"Based on the provided context, here is a concise answer to your question: {query_text}. Context used: {context_preview}"


def test_current_system_offline(tmp_path):
    # Use temp cache dir to avoid mutating real cache during tests
    cache_dir = tmp_path.as_posix()

    # Initialize system with fakes to ensure offline, deterministic behavior
    system = DocumentRAGSystem(
        embedder=FakeEmbedder(),
        chat_client=fake_chat_client,
        cacheDir=cache_dir,
        dataDir="data",
    )
    system.initialize()

    questions = [
        "What is Anmol's work experience?",
        "What did he do at Apple?",
        "How long was he at Apple?",
    ]

    print("\n--- RAG Offline Smoke Test (metrics) ---")
    print(f"Cache dir: {cache_dir}")
    print(f"Num questions: {len(questions)}\n")

    for question in questions:
        answer = system.ragPipeline.ask(question)
        is_str = isinstance(answer, str)
        ans_len = len(answer) if is_str else 0
        # contains_phrase = ("Based on the provided context" in answer) if is_str else False

        # CLI metrics output per question
        print(f"Question: {question}")
        print(f"  - is_str: {is_str}")
        print(f"  - answer_length: {ans_len}")
        # print(f"  - contains_phrase: {contains_phrase}")
        print()

        # Assertions keep the test meaningful in CI
        assert is_str
        assert ans_len > 20
        # assert contains_phrase
