# YouTube Chatbot with LangChain + Streamlit

A Streamlit app that lets you ask questions about a YouTube video by building a retrieval pipeline over its transcript.

## Features

- Load a YouTube video using URL or video ID.
- Fetch transcript and split into chunks.
- Create embeddings with Gemini embeddings.
- Store vectors in FAISS and retrieve relevant context.
- Answer questions with Gemini chat model.
- Fallback support: paste transcript manually if YouTube blocks transcript requests.

## Project Structure

- app.py: Streamlit app for interactive Q&A.
- main.py: Script-based version of transcript Q&A flow.
- requirements.txt: Python dependencies for local and cloud deployment.

## Tech Stack

- Streamlit
- LangChain
- langchain-google-genai
- FAISS
- youtube-transcript-api
- python-dotenv

## Prerequisites

- Python 3.10+
- A Google AI API key with access to Gemini models

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies.
4. Add environment variables.

### 1) Create virtual environment

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment variables

Create a .env file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

## Run the App

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (usually http://localhost:8501).

## How to Use

1. Enter a YouTube URL or video ID.
2. Click Load Video.
3. Ask a question in the Question field.
4. Click Ask to get an answer grounded in retrieved transcript chunks.

If transcript loading fails because YouTube blocks requests from your IP:

1. Paste transcript text into the manual transcript box.
2. Click Use Pasted Transcript.
3. Continue asking questions.

## Notes on YouTube Blocking

Some cloud IP addresses are blocked by YouTube transcript endpoints. This can affect hosted deployments (including Streamlit Cloud).

Recommended options:

- Run locally on a residential IP.
- Use the built-in manual transcript fallback.
- Add a transcription fallback pipeline (for example, ASR) if needed.

## Streamlit Cloud Deployment

1. Push code to GitHub.
2. Create a Streamlit Cloud app connected to the repo.
3. Ensure requirements.txt is present at repo root.
4. Add GOOGLE_API_KEY in Streamlit Cloud Secrets.
5. Deploy.

## Troubleshooting

### ModuleNotFoundError: dotenv

Install dependencies and make sure python-dotenv is in requirements.txt.

### Failed to load video transcript

This is often an IP block/rate-limit issue from YouTube. Use manual transcript fallback.

### Gemini authentication/model errors

Verify GOOGLE_API_KEY is valid and has access to:

- models/gemini-embedding-001
- models/gemini-2.5-flash

## Future Improvements

- Automatic fallback to audio transcription when transcript fetch fails.
- Chat history memory per session.
- Source snippet citations in answers.
- Multi-language transcript support.

## License

MIT (or update based on your preference).
