# Verbi Agent: Invoice Review Assistant

![Cover](https://raw.githubusercontent.com/yYorky/Verbi/refs/heads/main/static/chatbot%20image.png)

## 1. Introduction

Verbi Agent is a prototype web application designed to assist in the review of documents such as **invoices**, **claims**, or **resumes**, where human-like validation and reasoning are typically required before acceptance or processing.

This demo focuses on **invoices submitted to a company's finance team** via a web portal. Users can upload invoice files (PDF or image), and Verbi Agent will:
- Extract and analyze document content using multi-modal LLMs
- Check for missing information, incorrect values, or invalid fields
- Provide a natural language summary of the issues (or validation confirmation)
- Optionally allow users to ask follow-up questions or interact via voice

---

## 2. Inspiration

The idea for this project was seeded during a conversation I had with [Max](https://www.linkedin.com/in/maxxumengxiang/)  back in Feb 2025. 
Max had a strong conviction: even in their current form, large language models (LLMs) can already create real value, 
not necessarily by becoming smarter on their own, but by working together. His belief was that if we could orchestrate a group of them, 
each specializing in something, we could solve real business problems. That insight stuck with me.

At the time, I had been working on a Retrieval-Augmented Generation (RAG) pipeline I called Verbi RAG: a conversational RAG using LLMs. 
which was inspired by [**Verbi by Prompt Engineer**](https://github.com/PromtEngineer/Verbi.git), 
designed to use discussed the context document with the user. Energized by the conversation, 
I immediately jumped into experimenting with a call agent, 
wiring up **Twilio** to let Verbi make phone calls to users to clarify missing or incorrect details in documents. 
The goal was simple: make Verbi more than just a chatbot—turn it into something that could initiate action.

But it didn't go as planned. The voice interaction worked okay until it didn't. 
The call would break unexpectedly after the user hung up, and interruptions during the call weren't handled well. 
I shelved the work temporarily, choosing to focus on wrapping up my semester at SMU's MITB program.

A few weeks ago, I picked it up again, only to hit another wall. 
Twilio had implemented new restrictions (following IMDA regulations), 
limiting international numbers from calling Singapore numbers unless they were verified. 
This made testing and iterating nearly impossible. After some reflection, I decided it was time to pivot.

That's how this version of Verbi came to be. Rather than focusing on a future step like a follow-up phone call, 
I redesigned the flow so that the assistant interacts immediately **as soon as the user uploads the document**. 
This shift allowed the project to stay grounded, working with what we already know LLMs can do well today: 
reading, interpreting, and reasoning with unstructured information.

The original brainstorming with Max had also touched on other tools like **UI-tars**, `browser-use`, `browserbase`, 
and even RPA-style workflows. He had suggested chaining LLMs with different modalities and specializations: one to read the document, 
another to plan actions, maybe even one to browse the web or trigger follow-ups. 
He also suggested to look at **AutoGen from Microsoft** as a possible framework for chaining agents together. 
The vision was to automate complex, high-friction workflows—like document validation, KYC, 
claims handling—where humans typically get pulled in too early.

While this project doesn't implement the full chain of agents yet, it's a concrete and usable step in that direction: 
a tool that reviews uploaded invoices, checks for completeness and accuracy, 
and interacts naturally with the user via voice or text—all without needing human review unless necessary.

And that's really the spirit of Verbi: less automation for the sake of it, more augmentation that feels useful and frictionless.

### What Was Implemented:
- Multimodal & Reasoning LLMs (Llama4 for vision, DeepSeek for reasoning)
- File upload and intelligent parsing (PDF & image support)
- Voice assistant to clarify invoice issues
- Step-by-step chunking and conversational document validation
- Modular pipeline for future integration of multi-agent follow-up

### What Was Not Yet Implemented:
- Multi-agent chaining (e.g., browser agents, dynamic calling)
- Automated correction workflow or submission loop
- Web search or external info retrieval via agent
- Calling functionality (e.g. via Twilio)

---

## 3. Demo

![Demo](https://github.com/yYorky/Verbi-Agent/blob/main/static/verbiagent.gif?raw=true)

Try it out by:
1. Uploading an invoice (PDF/image) using the left sidebar.
2. Watching Verbi analyze and explain (verbally) if the invoice is valid.
3. Clicking "Click to talk" to ask questions or clarify the findings via voice.

### Example Scenarios

![Sample Invoice](https://raw.githubusercontent.com/yYorky/Verbi-Agent/refs/heads/main/static/Incorrect%20Invoice%20picture.JPG)

#### Scenario 1: Missing Payment Details
- **Uploaded Invoice**: An image of an invoice where the payment details (PayNow) is missing.
- **Verbi's Response**: 
  - Text & Voice: "The UEN number for payment via Paynow is missing"

#### Scenario 2: Incorrect Total Amount
- **Uploaded Invoice**: A PDF invoice where the total amount is listed as $3,000, but the line items add up to $2,900.
- **Verbi's Response**: 
  - Text & Voice: "Additionally, there's a calculation error - the subtotal at 3,000 but the adjustment of -100 isn't applied correctly."

### How to Use
- Upload PDF/image in the sidebar.
- App extracts and splits content into chunks (if PDF document).
- Uses LLaMA-4 (Vision) to transcribe images.
- Uses DeepSeek to check for issues in invoice fields.
- Responds with final conclusion (missing fields or ready for processing).

---

## 4. How to Use

### Prerequisites
- Python 3.10+
- Streamlit
- GROQ API Key (for LLaMA and DeepSeek)
- Voice Assistant dependencies from VERBI by Prompt Engineer(`whisper`, `TTS`, audio libs)

### Setup
1. Clone the repository:
   ```bash
   git clone git@github.com:yYorky/Verbi-Agent.git
   cd Verbi-Agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file:
   ```
   GROQ_API_KEY=your_groq_api_key
   CARTESIA_API_KEY =your_cartesia_api_key
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

### Upload Flow
- Upload PDF/image in sidebar
- App extracts and splits content into chunks (if pdf document)
- Uses LLaMA-4 (Vision) to transcribe images
- Uses DeepSeek to check for issues in invoice fields
- Responds with final conclusion (missing fields or ready for processing)

### Voice Assistant
- Click "Click to talk"
- Speak your query (e.g., "What's wrong with the invoice?")
- Verbi replies via text and voice using TTS

---

## 5. Folder Structure

```
├── app.py
├── voice_assistant/
│   ├── audio.py
│   ├── transcription.py
│   ├── text_to_speech.py
│   ├── local_tts_generation.py
│   ├── response_generation.py
│   ├── config.py
│   └── api_key_manager.py
├── static/
│   ├── Incorrect Invoice picture.JPG
│   ├── Incorrect Invoice.pdf
│   └── verbiagent.gif
├── README.md
└── requirements.txt
```

