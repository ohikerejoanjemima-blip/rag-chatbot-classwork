"""Reusable model resources; inference is serialized across browser sessions."""
import threading


class ChatModel:
    def __init__(self, model_id):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.lock = threading.RLock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def generate(self, messages, max_new_tokens=256):
        with self.lock, self.torch.inference_mode():
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                text, return_tensors="pt", add_special_tokens=False
            ).to(self.device)
            length = inputs["input_ids"].shape[-1]
            if length > 4096:
                raise ValueError("Please shorten the question or clear conversation memory.")
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            answer = self.tokenizer.decode(output[0, length:], skip_special_tokens=True).strip()
            if not answer:
                raise ValueError("The model returned an empty answer. Please try again.")
            return answer


class EmbeddingResource:
    def __init__(self):
        from langchain_huggingface import HuggingFaceEmbeddings
        self.lock = threading.RLock()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
