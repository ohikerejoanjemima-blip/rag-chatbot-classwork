# Conversational PDF RAG in Streamlit

A Streamlit wrapper for the conversational RAG notebook: PDF upload, Qwen chat generation, Chroma retrieval, session-local conversation memory, source excerpts, clear-memory control, and conversation export. The detailed fictional handbook is bundled for testing.

## Run locally

Use Python 3.11 or 3.12 in a **new virtual environment**, rather than the Colab environment modified during earlier experiments. The first run downloads the Qwen model and sentence embedding model. There is no API key requirement. A GPU is recommended; CPU generation is slower. Allow several GB of disk and RAM for the default 1.5B model.

```bash
cd streamlit_rag
python -m venv .venv
```

Activate it:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Then run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the URL printed by Streamlit (normally http://localhost:8501). Select the sample handbook or upload your own PDF, click **Load document**, and start chatting.

For a smaller model with lower memory requirements, set `RAG_MODEL_ID` before launching:

```bash
# macOS/Linux
export RAG_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
```

```powershell
# Windows PowerShell
$env:RAG_MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
```

Restart the Streamlit server when changing the model. Avoid repeatedly loading different model sizes in one server process, since cached models stay in memory.

## Test memory

Ask these consecutively:

1. What is my annual learning budget? (NGN 300,000 after probation)
2. If I spend NGN 120,000, how much remains? (NGN 180,000)
3. Can I carry that balance into next year? (No)

Expand **Sources and search question** to inspect retrieval. The next question is rewritten with recent conversation before fresh retrieval. Answers can still be incorrect; citation-ID checks do not verify semantic support.

## What is saved

Conversation memory and the PDF vector collection remain in the current Streamlit browser session. Six recent turns are retained for the model, subject to token trimming. The UI keeps the full displayed transcript. Clearing conversation clears both. Selecting a different PDF deletes the old collection and resets memory before you load the new document. Refreshing/disconnecting the browser can reset Streamlit session state; server restarts reset all session data.

Model weights are cached on disk by Hugging Face. The app caches model resources globally, with locks around shared inference and embeddings; chat histories and document indices are separate per session. Conversations can be exported as JSON. Export does not enable automatic restore, and the app does not persist a PDF index across server restarts.

## Files

- `app.py`: Streamlit interface and session lifecycle
- `conversation.py`: history, follow-up rewriting, retrieval, and grounded answer construction
- `documents.py`: text extraction, splitting, and separate Chroma collections
- `resources.py`: chat and embedding model loading
- `sample_data/`: eight-page fictional test PDF
- `tests/`: offline tests for memory and PDF parsing

## Troubleshooting

- **No selectable text:** run OCR on your scanned PDF before uploading.
- **Model download fails:** check internet access and free disk space; no secret token is required for these public models.
- **Out of memory / slow CPU:** use the smaller model above and restart the server.
- **`torchvision::nms` import failure:** in this project's virtual environment only, run `python -m pip uninstall -y torchvision`. This app uses text models and does not install torchvision.
- **Bad follow-up:** inspect the rewritten question or clear conversation and ask an explicit standalone question.

This is a local prototype, not an authenticated public service. For public multi-user deployment, add authentication, durable per-user storage, lifecycle cleanup for disconnected sessions, and an appropriate model server. PDF upload limit is 25 MB, 500 pages, and two million extracted characters.

## Offline checks

```bash
python -m unittest discover -s tests -v
```

Core logic and PDF parsing can be checked without downloading models (only `pypdf` is needed for the parsing tests). Full inference requires the installed dependencies and downloaded model weights.

Documentation: [Streamlit chat](https://docs.streamlit.io/develop/api-reference/chat), [Streamlit caching](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource), [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
