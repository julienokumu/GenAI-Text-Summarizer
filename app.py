# Import Libraries
import streamlit as st
from transformers import pipeline
import torch

# Load the summarizer model using Hugging Face's pipeline with PyTorch backend
@st.cache_resource
def load_summarizer():
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    return summarizer

summarizer = load_summarizer()

# Function to split long text into smaller chunks
def split_text(text, max_chunk_length=1024):
    """Splits text into smaller chunks of a specified maximum length."""
    sentences = text.split('. ')
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        if current_length + len(sentence) <= max_chunk_length:
            current_chunk.append(sentence)
            current_length += len(sentence) + 1  # +1 for the period
        else:
            yield '. '.join(current_chunk) + '.'
            current_chunk = [sentence]  # Start a new chunk with the current sentence
            current_length = len(sentence) + 1
    if current_chunk:
        yield '. '.join(current_chunk) + '.'

# Build the Streamlit Interface
def main():
    st.title("GenAI Text Summarizer | CodewithJulien")
    st.write("Enter the text you want to summarize:")

    # Text input
    user_input = st.text_area("Input Text")

    # Button to generate summary
    if st.button("Summarize"):
        if user_input.strip() == "":
            st.warning("Please enter some text to summarize.")
        else:
            # Break the input into smaller chunks
            chunks = list(split_text(user_input))
            summaries = []
            for chunk in chunks:
                # Generate summary for each chunk
                summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])

            # Combine summaries
            full_summary = ' '.join(summaries)
            st.subheader("Generated Summary")
            st.write(full_summary)

if __name__ == "__main__":
    main()