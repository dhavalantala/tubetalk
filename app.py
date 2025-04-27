# app.py

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from youtube_transcript_api.formatters import TextFormatter

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

import re
from typing import Optional, List
import gradio as gr
import asyncio

from dotenv import load_dotenv

load_dotenv()

# --- Utility Functions ---

def extract_video_id(url: str) -> Optional[str]:
    """Extracts the YouTube video ID from a URL."""
    match_v = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    if match_v:
        return match_v.group(1)
    match_be = re.search(r"youtu\\.be\\/([a-zA-Z0-9_-]+)", url)
    if match_be:
        return match_be.group(1)
    return None

def get_transcript(video_id: str, languages: Optional[List[str]] = None) -> Optional[str]:
    """Fetches the transcript of a YouTube video."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        transcript_text = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript_text
    except TranscriptsDisabled:
        print("No captions available for this video.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def extract_transcript_from_url(youtube_url: str, languages: Optional[List[str]] = None) -> Optional[str]:
    """Extracts the transcript from a YouTube video URL."""
    video_id = extract_video_id(youtube_url)
    if not video_id:
        print(f"Error: Could not extract video ID from URL: {youtube_url}")
        return None
    return get_transcript(video_id, languages)

def format_docs(retrieved_docs) -> str:
    """Formats retrieved documents into a single context string."""
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

# --- Main Application Logic ---

def process_youtube_video(youtube_url: str, query: str, languages: List[str]) -> str:
    """Processes a YouTube video URL and a query, returning the answer."""
    if not languages:
        return "Please select at least one language for the transcript."

    transcript = extract_transcript_from_url(youtube_url, languages)
    if not transcript:
        return "Failed to retrieve transcript."

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    template = """
    You are a helpful assistant.
    Answer ONLY from the following transcript context.
    If the context is insufficient to answer the question, just say "I don't know."
    If possible, cite the specific part of the context that supports your answer.

    Transcript Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    main_chain = parallel_chain | prompt | llm | StrOutputParser()

    answer = main_chain.invoke(query)
    return answer

# --- Gradio UI ---

def gradio_interface(youtube_url: str, query: str, languages_str: str) -> str:
    """Gradio interface function."""
    try:
        languages = [lang.strip() for lang in languages_str.split(",") if lang.strip()]
        result = process_youtube_video(youtube_url, query, languages)
        return result
    except Exception as e:
        return f"An error occurred: {e}"

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="YouTube URL", placeholder="Enter YouTube URL"),
        gr.Textbox(label="Your Question", placeholder="Enter your question about the video"),
        gr.Textbox(label="Preferred Transcript Languages (comma separated)", placeholder="e.g., en,de,fr"),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="TubeTalk",
    description="Ask questions about YouTube videos and get answers from the transcript.\n\nSelect at least one language (e.g., 'en' for English, 'de' for German).",
)

if __name__ == "__main__":
    iface.launch()
