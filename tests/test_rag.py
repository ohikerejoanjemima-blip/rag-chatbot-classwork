import unittest
from pathlib import Path
from types import SimpleNamespace
from conversation import ConversationalPDFRAG
from documents import extract_pages, MAX_PDF_BYTES


class WordTokenizer:
    def encode(self, text, **kwargs):
        return text.split()

    def decode(self, tokens, **kwargs):
        return " ".join(tokens)


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def invoke(self, question):
        self.queries.append(question)
        return [SimpleNamespace(page_content="Carried-over leave expires on March 31.",
                                metadata={"page": 1, "source": "handbook.pdf"})]


class FakeGenerator:
    def __init__(self):
        self.calls = []
        self.fail = False

    def __call__(self, messages, **kwargs):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("Inference failed")
        if "Rewrite" in messages[0]["content"]:
            return "When do carried-over annual leave days expire?"
        return "March 31. [S1]"


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.retriever = FakeRetriever()
        self.generator = FakeGenerator()
        self.bot = ConversationalPDFRAG(self.retriever, self.generator, WordTokenizer())

    def test_follow_up_uses_history_and_fresh_retrieval(self):
        self.bot.ask("Can I carry over unused annual leave?")
        result = self.bot.ask("When do those expire?")
        self.assertEqual(len(self.retriever.queries), 2)
        self.assertEqual(self.retriever.queries[-1], "When do carried-over annual leave days expire?")
        self.assertIn("Can I carry over", self.generator.calls[1][-1]["content"])
        self.assertTrue(any(m["role"] == "assistant" for m in self.generator.calls[-1]))
        self.assertEqual(result["sources"][0]["page"], "2")
        self.assertEqual(result["invalid_citations"], [])

    def test_session_isolation(self):
        other = ConversationalPDFRAG(self.retriever, self.generator, WordTokenizer())
        self.bot.ask("My question")
        self.assertEqual(other.history, [])

    def test_history_window_and_reset(self):
        for _ in range(12):
            self.bot.ask("Tell me about leave")
        self.assertEqual(len(self.bot.history), 12)
        self.bot.reset()
        self.assertEqual(self.bot.history, [])

    def test_failed_generation_does_not_save_turn(self):
        self.bot.ask("First question")
        before = list(self.bot.history)
        self.generator.fail = True
        with self.assertRaises(RuntimeError):
            self.bot.ask("Follow-up")
        self.assertEqual(self.bot.history, before)

    def test_invalid_citation_is_reported(self):
        self.bot.generate_fn = lambda *a, **kw: "Unknown source [S9]"
        self.assertEqual(self.bot.ask("Question")["invalid_citations"], ["S9"])

    def test_empty_and_long_questions_are_rejected(self):
        for text in (" ", "word " * 257):
            with self.assertRaises(ValueError):
                self.bot.ask(text)
        self.assertEqual(self.bot.history, [])


class PDFTests(unittest.TestCase):
    def test_sample_page_labels_and_text(self):
        sample = Path(__file__).resolve().parents[1] / "sample_data" / "detailed_rag_test_handbook.pdf"
        pages, count = extract_pages(sample.read_bytes(), "../../handbook.pdf")
        self.assertEqual(count, 8)
        self.assertEqual([p["page"] for p in pages], list(range(8)))
        self.assertEqual(pages[0]["source"], "handbook.pdf")
        self.assertIn("300,000", pages[4]["text"])

    def test_invalid_empty_and_oversized_inputs(self):
        for data in (b"", b"not a pdf", b"a" * (MAX_PDF_BYTES + 1)):
            with self.assertRaises(ValueError):
                extract_pages(data, "test.pdf")


if __name__ == "__main__":
    unittest.main()
