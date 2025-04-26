# TubeTalk: Extracting Information from YouTube Videos with RAG

## Goal

My goal is to leverage the Retrieval-Augmented Generation (RAG) technique within the LangChain framework to enable users to efficiently extract information from long YouTube videos. This involves retrieving relevant segments based on user queries to generate concise summaries and accurately determine if specific topics are discussed within the video content.

## Key Features

-   **Efficient Information Extraction:** Quickly find answers to your questions within lengthy video content.
-   **Concise Summarization:** Get brief summaries of relevant video segments.
-   **Topic Detection:** Determine if specific topics are discussed in a video.

## How it Works

The application uses the following steps:

1.  **Text Extraction:** Extracts the transcript from a YouTube video.
2.  **Indexing:**
    * Splits the transcript into smaller chunks.
    * Generates embeddings for each chunk.
    * Stores the embeddings in a vector store (FAISS).
3.  **Retrieval:** Retrieves the most relevant chunks from the vector store based on a user's query.
4.  **Augmentation:** Combines the retrieved chunks with the original query to create a prompt for the language model.
5.  **Generation:** The language model generates an answer based on the augmented prompt.

## Technologies Used

-   LangChain:  Framework for developing applications powered by language models.
-   YouTube API:  For direct extraction of video transcripts.
-   OpenAI Embeddings:  For generating text embeddings.
-   FAISS:  For efficient vector storage and similarity search.
-   Gradio:  For creating the user interface.