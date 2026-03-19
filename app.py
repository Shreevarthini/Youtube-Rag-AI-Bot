import gradio as gr
import re 
import os
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains import LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import PromptTemplate
 
api_key = os.environ.get("GOOGLE_API_KEY")


def get_video_id(url):    
    pattern = r'(?:v=|\/)([a-zA-Z0-9_-]{11})(?:[&?]|$)'
    match = re.search(pattern, url)
    if match:
            return match.group(1)
        
    if len(url) == 11:
        return url
        
    return None

def get_transcript(url):

    video_id = get_video_id(url)
    ytt_api = YouTubeTranscriptApi()
    transcripts = ytt_api.list(video_id)
   
    transcript = ""
    for t in transcripts:

        if t.language_code == 'en':
            if t.is_generated:

                if len(transcript) == 0:
                    transcript = t.fetch()
            else:

                transcript = t.fetch()
                break
   
    return transcript if transcript else None
 
 
def process(transcript):
    txt = ""
    for i in transcript:
        try:

            text = i.text
            start = i.start
            txt += f"Text: {text} Start: {start}\n"
        except AttributeError:

            pass
    return txt
 
def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
 

    chunks = text_splitter.split_text(processed_transcript)
    return chunks
 

def initialize_gemini():

    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

def setup_gemini_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


 
def create_faiss_index(chunks, embedding_model):
    """
    Create a FAISS index from text chunks using the specified embedding model.
   
    :param chunks: List of text chunks
    :param embedding_model: The embedding model to use
    :return: FAISS index
    """

    return FAISS.from_texts(chunks, embedding_model)
 
 
 
def perform_similarity_search(faiss_index, query, k=3):
    """
    Search for specific queries within the embedded transcript using the FAISS index.
   
    :param faiss_index: The FAISS index containing embedded text chunks
    :param query: The text input for the similarity search
    :param k: The number of similar results to return (default is 3)
    :return: List of similar results
    """

    results = faiss_index.similarity_search(query, k=k)
    return results
 
 
def create_summary_prompt():
    """
    Create a PromptTemplate for summarizing a YouTube video transcript.
   
    :return: PromptTemplate object
    """

    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.
 
    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.
 
    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Please summarize the following YouTube video transcript:
 
    {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
   

    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=template
    )
   
    return prompt
 
 
def create_summary_chain(llm, prompt, verbose=True):
    """
    Create an LLMChain for generating summaries.
   
    :param llm: Language model instance
    :param prompt: PromptTemplate instance
    :param verbose: Boolean to enable verbose output (default: True)
    :return: LLMChain instance
    """
    return LLMChain(llm=llm, prompt=prompt, verbose=verbose)
 
 
def retrieve(query, faiss_index, k=7):
    """
    Retrieve relevant context from the FAISS index based on the user's query.
 
    Parameters:
        query (str): The user's query string.
        faiss_index (FAISS): The FAISS index containing the embedded documents.
        k (int, optional): The number of most relevant documents to retrieve (default is 3).
 
    Returns:
        list: A list of the k most relevant documents (or document chunks).
    """
    relevant_context = faiss_index.similarity_search(query, k=k)
    return relevant_context
 
def create_qa_prompt_template():
    """
    Create a PromptTemplate for question answering based on video content.
    Returns:
        PromptTemplate: A PromptTemplate object configured for Q&A tasks.
    """
   

    qa_template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert assistant providing detailed and accurate answers based on the following video content. Your responses should be:
    1. Precise and free from repetition
    2. Consistent with the information provided in the video
    3. Well-organized and easy to understand
    4. Focused on addressing the user's question directly
    If you encounter conflicting information in the video content, use your best judgment to provide the most likely correct answer based on context.
    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|>
 
    <|start_header_id|>user<|end_header_id|>
    Relevant Video Context: {context}
    Based on the above context, please answer the following question:
    {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )
    return prompt_template
 
 
def create_qa_chain(llm, prompt_template, verbose=True):
    """
    Create an LLMChain for question answering.
 
    Args:
        llm: Language model instance
            The language model to use in the chain (e.g., WatsonxGranite).
        prompt_template: PromptTemplate
            The prompt template to use for structuring inputs to the language model.
        verbose: bool, optional (default=True)
            Whether to enable verbose output for the chain.
 
    Returns:
        LLMChain: An instantiated LLMChain ready for question answering.
    """
   
    return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)
 
 
def generate_answer(question, faiss_index, qa_chain_not_needed, k=7):
    """
    Retrieve relevant context and generate an answer based on user input.
 
    Args:
        question: str
            The user's question.
        faiss_index: FAISS
            The FAISS index containing the embedded documents.
        qa_chain: LLMChain
            The question-answering chain (LLMChain) to use for generating answers.
        k: int, optional (default=3)
            The number of relevant documents to retrieve.
 
    Returns:
        str: The generated answer to the user's question.
    """
 

    relevant_context = retrieve(question, faiss_index, k=k)
    context_text = "\n".join([doc.page_content for doc in relevant_context])
    
    llm = initialize_gemini()
    prompt = create_qa_prompt_template()
    chain = prompt | llm

    answer = chain.invoke({"context": context_text, "question": question})
 
    return answer.content
 
transcript_storage = ""
 
def summarize_video(video_url):
    """
    Title: Summarize Video
 
    Description:
    This function generates a summary of the video using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.
 
    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
 
    Returns:
        str: The generated summary of the video or a message indicating that no transcript is available.
    """
    raw_data = get_transcript(video_url)
    if not raw_data: return "Could not find an English transcript for this video."
    
    full_text = process(raw_data)
    llm = initialize_gemini()
    
    prompt = PromptTemplate.from_template("""
    <|begin_of_text|>system
    You are an AI assistant summarizing YouTube transcripts. 
    Provide a concise paragraph capturing the main points. Ignore timestamps.
    
    user
    Summarize this transcript: {transcript}
    assistant
    """)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"transcript": full_text})
 
 
def answer_question(video_url, user_question):
    raw_data = get_transcript(video_url)
    if not raw_data or not user_question: return "Ensure URL and Question are provided."
    
    full_text = process(raw_data)
    
    chunks = chunk_transcript(full_text, 500)
    embeddings = setup_gemini_embeddings()
    vector_db = FAISS.from_texts(chunks, embeddings)
    
    docs = vector_db.similarity_search(user_question, k=5)
    context = "\n".join([d.page_content for d in docs])
    
    llm = initialize_gemini()
    prompt = PromptTemplate.from_template("""
    Answer the question using ONLY the video context provided.
    Context: {context}
    Question: {question}
    """)
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": user_question})
 
with gr.Blocks() as interface:
 
    gr.Markdown(
        "<h2 style='text-align: center;'>YouTube Video Summarizer and Q&A</h2>"
    )
 
    video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter the YouTube Video URL")
   
    summary_output = gr.Textbox(label="Video Summary", lines=5)
    question_input = gr.Textbox(label="Ask a Question About the Video", placeholder="Ask your question")
    answer_output = gr.Textbox(label="Answer to Your Question", lines=5)

    summarize_btn = gr.Button("Summarize Video")
    question_btn = gr.Button("Ask a Question")
 

    transcript_status = gr.Textbox(label="Transcript Status", interactive=False)

    summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
    question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)

interface.launch(server_name="0.0.0.0", server_port=7860)
