# Verbi Invoice Reviewer – Full Application Breakdown

## 🔍 Overview
`app.py` runs **Verbi**, a voice-enabled assistant that reviews uploaded invoices. It supports both PDFs and image files, processes them using LLMs, and replies via text and speech. 
Users can interact with Verbi using their voice, making the experience feel like a chat with a finance teammate.

---

## 🎯 Design Goals

### 1. Multimodal Interaction
The app accepts:
- **PDFs**: Extracted using `PyPDF2`, chunked for LLM input.
- **Images**: Converted to base64, then transcribed using a vision-capable LLM.
- **Voice input**: Recorded via `record_audio`, transcribed, and processed like a text query.

This flow supports natural, human-like interaction and aligns with how invoices arrive in the real world (as files or screenshots, often discussed over calls).

### 2. Modular Pipeline
Functions and logic are cleanly separated:
- `voice_assistant/audio.py`: Audio I/O
- `voice_assistant/transcription.py`: Speech-to-text
- `voice_assistant/text_to_speech.py`: Text-to-speech
- `app.py`: Orchestrates everything

This modular structure makes it easy to test and extend, e.g., swap models.

---

## 🤖 LLM Usage & Prompt Strategy

### Model Choices
```python
llama_scout = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", ...)
deepseek = ChatGroq(model_name="deepseek-r1-distill-llama-70b", ...)
```
- **Llama 4 Scout**: New Multimodal model from Meta, good at vision + text (used to transcribe invoices from images).
- **DeepSeek**: Strong at structured reasoning (used to assess completeness of invoice data).

### Prompt Engineering
The system prompt:
```python
system_prompt = """You are Verbi, an invoice review assistant..."""
```
- Instructs the LLM to respond briefly, clearly, and only with conclusions — no reasoning or step-by-step.
- Matches the tone of a fast-paced finance assistant, not a chatty bot.
- Internal thoughts fo deepseek-r1 model (`<think>...</think>`) are removed using:
```python
cleaned_response = re.sub(r'<think>.*?</think>', '', response_text)
```

---

## ⚙️ Document Handling Logic

### PDFs
```python
if file_type == "pdf":
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    ...
```
- Full text is extracted and split using `RecursiveCharacterTextSplitter` for better handling in LLMs.
- Each chunk is passed to the LLM one by one.
- Findings are merged to form a single final output.

### Images
```python
image_description, img_base64 = process_image(uploaded_file)
```
- The image is described and encoded.
- Sent to the LLM via an API payload with a user message of type `image_url`.

```python
transcription_payload = {
    "messages": [
        {"type": "text", ...},
        {"type": "image_url", "image_url": {"url": "data:image/..."}}
    ]
}
```

The response is transcribed invoice text, which is then fed into a reasoning chain using DeepSeek.

---

## 🗣 Voice Flow

```python
recorded_file = record_audio(...)
user_input = transcribe_audio(...)
response = st.session_state.conversation_chain.invoke(...)
text_to_speech(...)
```

- Voice is recorded and transcribed to text.
- The text is sent through the same invoice analysis pipeline.
- Verbi's response is then played back using TTS.

This mirrors a real-time phone call and increases accessibility.

---

## 🧠 Memory & Context

Conversation memory is tracked using:
```python
ConversationBufferMemory(memory_key="chat_history", ...)
```
This lets Verbi remember earlier responses and user questions, so it can continue a thread:
> “You mentioned missing payment details earlier — have they been added?”

All memory is stored in `st.session_state.chat_history` and displayed in a dynamic chat log.

---

## 🖼 UI Flow (Streamlit)

- Sidebar handles file upload and preview (PDFs and images).
- Main column includes a “Click to talk” button and shows responses.
- UI elements like spinners and success messages provide clarity on current app state.

```python
st.sidebar.file_uploader(...)
st.button("Click to talk")
st.markdown("### Chat History")
```

---

## 🔐 Security & Config

- Environment variables (`GROQ_API_KEY`) are loaded via `.env`.
- API calls are authenticated and abstracted using helper modules (`api_key_manager.py`).

---

## 🔧 Error Handling

- Every file I/O and model call is wrapped in `try-except` blocks with `logging`.
- User-friendly messages appear in the UI when something fails:
```python
st.error("Failed to analyze the document. Please try again.")
```

