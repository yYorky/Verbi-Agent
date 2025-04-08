import streamlit as st
import logging
import os
import tempfile
import base64
import PyPDF2
import re
from PIL import Image
import io
import requests
from voice_assistant.audio import record_audio, play_audio
from voice_assistant.transcription import transcribe_audio
from voice_assistant.text_to_speech import text_to_speech
from langchain_groq import ChatGroq
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from voice_assistant.config import Config
from voice_assistant.api_key_manager import get_transcription_api_key, get_tts_api_key

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Define API endpoints and headers
api_url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {groq_api_key}",
    "Content-Type": "application/json"
}

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the system prompt
system_prompt = """You are Verbi, an invoice review assistant. Your primary objective is to review uploaded invoices and identify:
1. Missing information required for payment processing
2. Incorrect amounts, calculations, or totals
3. Missing or incorrect payment details
4. Any other issues that would prevent proper invoice processing

When an invoice is uploaded, you should:
1. Greet the user warmly but briefly
2. Introduce yourself as an invoice review assistant
3. Analyze the invoice for completeness and accuracy
4. If issues are found:
   - Clearly state the issues found
   - Ask the user to rectify and resubmit
5. If no issues are found:
   - Inform the user that the invoice will be processed by finance & accounting in 1-2 working days

IMPORTANT: Keep your responses concise and direct, like a phone conversation. Avoid lengthy explanations and get straight to the point. Use short sentences and a conversational tone."""

# Function to clean assistant response by removing thinking process
def clean_assistant_response(response_text):
    """
    Removes the thinking process (<think>...</think>) from the assistant's response.
    Returns only the actual response part.
    """
    # Pattern to match content within <think> tags (including the tags)
    think_pattern = r'<think>.*?</think>'
    
    # Remove the thinking part using regex (non-greedy match)
    cleaned_response = re.sub(think_pattern, '', response_text, flags=re.DOTALL)
    
    # Trim any leading/trailing whitespace
    cleaned_response = cleaned_response.strip()
    
    return cleaned_response

# Initialize Streamlit app
st.set_page_config(page_title="Verbi Invoice Reviewer", layout="wide")

# Add a header with an image
st.markdown("<h1 style='text-align: center;'>Verbi Invoice Reviewer</h1>", unsafe_allow_html=True)

# Add resized image with a round border using HTML and CSS
st.markdown(
    """
    <div style='text-align: center;'>
        <img src="https://raw.githubusercontent.com/yYorky/Verbi/refs/heads/main/static/chatbot%20image.png" 
             style="width: 200px; height: 200px; border-radius: 50%; object-fit: cover; border: 3px solid #4CAF50;">
    </div>
    """,
    unsafe_allow_html=True,
)

# Ensure `chat_history` is part of session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "invoice_analyzed" not in st.session_state:
    st.session_state.invoice_analyzed = False

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

if "invoice_type" not in st.session_state:
    st.session_state.invoice_type = None

# Sidebar for invoice upload
st.sidebar.title("Upload Invoice for Review")
uploaded_file = st.sidebar.file_uploader("Upload an invoice (PDF or image) for review", type=["pdf", "png", "jpg", "jpeg"])

# Display a preview of the uploaded file if it exists
if uploaded_file:
    # Get file extension to determine type
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Display a preview based on file type
    st.sidebar.subheader("Preview")
    
    if file_extension in ['png', 'jpg', 'jpeg']:
        # For image files, display the image directly
        image = Image.open(uploaded_file)
        
        # Create a container with a border and padding
        st.sidebar.markdown(
            """
            <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 10px; margin-top: 10px;">
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Display image with caption
        st.sidebar.image(image, caption=uploaded_file.name, use_column_width=True)
        
        # Reset file pointer to beginning after reading
        uploaded_file.seek(0)
    else:
        # For PDFs, display an info message
        st.sidebar.info(f"PDF file uploaded: {uploaded_file.name}")
        
        # Optional: Add PDF icon or placeholder
        st.sidebar.markdown(
            """
            <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 10px; text-align: center; margin-top: 10px;">
                <i class="material-icons" style="font-size: 48px; color: #FF5733;">description</i>
                <p>PDF Preview Not Available</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n\n"
    return text

# Function to process image file
def process_image(image_file):
    # Open the image
    image = Image.open(image_file)
    
    # Convert image to base64 for LLM processing
    buffered = io.BytesIO()
    image.save(buffered, format=image.format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Create a more detailed description of the image
    image_description = f"""
    [Invoice Analysis]
    Format: {image.format}
    Dimensions: {image.size[0]}x{image.size[1]} pixels
    Mode: {image.mode}
    
    This is an invoice that has been uploaded for review. Please analyze the visual content, 
    identify all text, amounts, dates, and payment details visible in the invoice.
    """
    
    return image_description, img_str

# Function to chunk text into smaller pieces
def chunk_text(text, chunk_size=2000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to analyze document with Llama 4 LLM
def analyze_document(file_path, file_type):
    if not uploaded_file:
        st.error("No file uploaded. Please upload a document.")
        return None
    
    # Initialize Llama 4 Scout LLM for image transcription
    llama_scout = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.1
    )
    
    # Initialize DeepSeek LLM for reasoning
    deepseek = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="deepseek-r1-distill-llama-70b",
        temperature=0.1
    )
    
    # Process based on file type
    if file_type == "pdf":
        # Extract text from PDF
        text = extract_text_from_pdf(file_path)
        
        # Chunk the text into smaller pieces
        chunks = chunk_text(text)
        st.session_state.document_chunks = chunks
        document_content = chunks[0]  # Use first chunk for initial analysis
        
        # Create prompt template for document analysis
        prompt_template = PromptTemplate(
            input_variables=["document_content", "chat_history", "system_prompt", "file_type"],
            template="""
            {system_prompt}
            
            Document Type: {file_type}
            Document Content: {document_content}
            
            Chat History:
            {chat_history}
            
            Please analyze this document and identify any missing or incorrect information.
            Remember to be concise and direct in your response, like a phone conversation.
            """
        )
        
        # Create memory for conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="text",
            input_key="document_content",
            human_prefix="User",
            ai_prefix="Verbi",
        )
        
        # Create conversation chain
        conversation_chain = LLMChain(
            llm=llama_scout,
            prompt=prompt_template,
            memory=memory,
            verbose=True
        )
        
        # Analyze PDF document
        initial_response = conversation_chain.invoke({
            "document_content": document_content,
            "chat_history": "",
            "system_prompt": system_prompt,
            "file_type": file_type
        })
        
        # For PDFs, process remaining chunks
        if len(chunks) > 1:
            all_findings = [initial_response["text"]]
            for i, chunk in enumerate(chunks[1:], 1):
                with st.spinner(f"Analyzing document part {i+1}/{len(chunks)}..."):
                    response = conversation_chain.invoke({
                        "document_content": chunk,
                        "chat_history": "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history]),
                        "system_prompt": system_prompt,
                        "file_type": file_type
                    })
                    all_findings.append(response["text"])
            
            # Combine all findings
            combined_response = "\n\n".join(all_findings)
            return conversation_chain, combined_response
        
        return conversation_chain, initial_response["text"]
    else:  # Image file
        # Process image and get base64 encoding for multimodal input
        image_description, img_base64 = process_image(uploaded_file)
        st.session_state.document_chunks = [image_description]  # Store image description as a chunk
        document_content = image_description
        
        # Create a template for invoice transcription
        transcription_template = PromptTemplate(
            input_variables=["system_prompt"],
            template="""
            {system_prompt}
            
            Please transcribe this invoice in detail, including:
            1. Invoice number and date
            2. Vendor/supplier details
            3. Line items with descriptions, quantities, and amounts
            4. Subtotal, tax, and total amounts
            5. Payment terms and due date
            6. Any special instructions or notes
            
            Format the transcription in a clear, structured manner.
            """
        )
        
        # Create a template for invoice analysis
        analysis_template = PromptTemplate(
            input_variables=["transcription", "system_prompt"],
            template="""
            {system_prompt}
            
            Invoice Transcription:
            {transcription}
            
            Based on the invoice transcription above, provide ONLY a final conclusion in this format:
            
            If issues are found:
            "I've reviewed your invoice and found the following issues that need to be addressed:
            [List specific issues]
            Please correct these issues and resubmit the invoice."
            
            If no issues are found:
            "I've reviewed your invoice and everything looks good. Your invoice will be processed by the finance & accounting department within 1-2 working days."
            
            DO NOT show your reasoning process or analysis steps. Provide ONLY the final conclusion in the format above.
            Keep the response concise and direct, like a phone conversation.
            """
        )
        
        try:
            # First, use Llama 4 Scout for detailed transcription
            transcription_chain = LLMChain(
                llm=llama_scout,
                prompt=transcription_template,
                verbose=True
            )
            
            # Make the transcription API call
            transcription_payload = {
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Please transcribe this invoice in detail, including all text, amounts, dates, and payment details."},
                        {"type": "image_url", "image_url": {"url": f"data:image/{uploaded_file.type.split('/')[-1]};base64,{img_base64}"}}
                    ]}
                ],
                "temperature": 0.1,
                "max_tokens": 2048
            }
            
            transcription_response = requests.post(api_url, headers=headers, json=transcription_payload)
            transcription_json = transcription_response.json()
            
            if transcription_response.status_code == 200:
                # Extract transcription text
                transcription_text = transcription_json["choices"][0]["message"]["content"]
                
                # Now use DeepSeek for analysis
                analysis_chain = LLMChain(
                    llm=deepseek,
                    prompt=analysis_template,
                    verbose=True
                )
                
                # Analyze the transcription
                analysis_response = analysis_chain.invoke({
                    "transcription": transcription_text,
                    "system_prompt": system_prompt
                })
                
                # Store in memory for conversation continuation
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="text",
                    input_key="document_content",
                    human_prefix="User",
                    ai_prefix="Verbi",
                )
                
                memory.save_context({"document_content": document_content}, {"text": analysis_response["text"]})
                
                return analysis_chain, analysis_response["text"]
            else:
                error_msg = f"Error from GROQ API: {transcription_json.get('error', {}).get('message', 'Unknown error')}"
                logging.error(error_msg)
                return None, error_msg
                
        except Exception as e:
            error_msg = f"Error processing invoice: {str(e)}"
            st.error(error_msg)
            logging.error(error_msg)
            return None, error_msg

# Function to handle voice assistant interaction
def handle_voice_assistant():
    if not st.session_state.get("conversation_chain"):
        st.error("Please upload an invoice first.")
        return

    st.info("Recording... Speak now.")
    recorded_file = record_audio(Config.INPUT_AUDIO)

    if not recorded_file:
        st.warning("No audio recorded. Try again.")
        return

    transcription_api_key = get_transcription_api_key()
    user_input = transcribe_audio(
        Config.TRANSCRIPTION_MODEL, transcription_api_key, recorded_file, Config.LOCAL_MODEL_PATH
    )

    if not user_input:
        st.warning("Unable to transcribe audio. Try again.")
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.info(f"You said: {user_input}")

    # Get response from conversation chain
    response = st.session_state.conversation_chain.invoke({
        "transcription": st.session_state.document_chunks[0],
        "system_prompt": system_prompt
    })
    
    response_text = response["text"]
    # Clean the response to remove thinking process
    cleaned_response = clean_assistant_response(response_text)
    st.session_state.chat_history.append({"role": "assistant", "content": cleaned_response})
    st.success(f"Verbi: {cleaned_response}")

    # Convert response to speech
    output_file = "output.mp3"
    tts_api_key = get_tts_api_key()
    text_to_speech(Config.TTS_MODEL, tts_api_key, cleaned_response, output_file, Config.LOCAL_MODEL_PATH)

# Process uploaded file
if uploaded_file and not st.session_state.invoice_analyzed:
    # Determine file type
    file_extension = uploaded_file.name.split('.')[-1].lower()
    file_type = "pdf" if file_extension == "pdf" else "image"
    st.session_state.invoice_type = file_type
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    # Analyze invoice
    with st.spinner("Analyzing invoice..."):
        conversation_chain, initial_response = analyze_document(temp_file_path, file_type)
        st.session_state.conversation_chain = conversation_chain
        st.session_state.invoice_analyzed = True
        
        # Clean the response to remove thinking process
        cleaned_response = clean_assistant_response(initial_response)
        
        # Add initial response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": cleaned_response})
        
        # Display initial response
        st.success(f"Verbi: {cleaned_response}")
        
        # Convert initial response to speech
        output_file = "output.mp3"
        tts_api_key = get_tts_api_key()
        text_to_speech(Config.TTS_MODEL, tts_api_key, cleaned_response, output_file, Config.LOCAL_MODEL_PATH)

# Main UI
st.markdown("<div style='position: fixed; top: 10px; width: 100%; text-align: center;'>", unsafe_allow_html=True)

# Create three columns
col1, col2, col3 = st.columns(3)

# Place the button in the center column
with col1:
    pass  # Empty columns for spacing
with col3:
    pass  # Empty columns for spacing
with col2:
    if st.button("Click to talk"):
        handle_voice_assistant()        
        
        # Display chat history dynamically
        st.markdown("### Chat History")
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            elif message["role"] == "assistant":
                st.markdown(f"**Verbi:** {message['content']}")