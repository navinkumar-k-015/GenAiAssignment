import os
# from sentence_transformers import SentenceTransformer
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
import re
import yaml
from langchain.schema import Document
# Initialize components
from langchain_huggingface import HuggingFaceEmbeddings

import torch
torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler._LRScheduler

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

def extract_front_matter(content):
    """Extract YAML front matter from markdown content."""
    front_matter_regex = r"^---\n(.*?)\n---"
    match = re.search(front_matter_regex, content, re.DOTALL)
    if match:
        front_matter_content = match.group(1)
        return yaml.safe_load(front_matter_content)
    return {}

def process_directory_for_files(base_dir):
    documents = []
    metadata_list = []

    for root, _, files in os.walk(base_dir):
        path_parts = os.path.relpath(root, start=base_dir).split(os.sep)
        # Collect directory metadata
        if '.ipynb_checkpoints' in path_parts:
            continue
        section = path_parts[-1] if path_parts else 'root'
        topic = path_parts[-2] if len(path_parts) > 1 else None

        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf8') as f:
                    content = f.read()
                    
                    # Extract and parse front matter
                    front_matter = extract_front_matter(content)
                    title = front_matter.get("title", "Untitled")

                    # Remove the front matter from content
                    content_body = re.sub(r"^---\n(.*?)\n---", "", content, count=1, flags=re.DOTALL)
                    
                    # Capture header-based content
                    header_splits = markdown_splitter.split_text(content_body)

                    for split in header_splits:
                        # Combine all types of metadata into enriched chunks
                        enriched_metadata = f"Title: {title}, Section: {section}, Topic: {topic or 'General'},  Header: {split.metadata}."
                        enriched_chunk = f"{enriched_metadata}<META>\n{split.page_content}"

                        # Append enriched chunk to documents

                        # Save metadata for retrieval context
                        metadata = {
                            'title': title,
                            'topic': topic,
                            'section': section,
                            'header': " ".join([f"{header}: {text}" for header, text in split.metadata.items()]),
                            'file': file_path,
                        }
                        document = Document(page_content=enriched_chunk, metadata=metadata)
                        documents.append(document)

    # Create FAISS index using the enriched documents
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore, metadata_list

# Base directory where your markdown files are located
base_directory = "demo_bot_data/ubuntu-docs/"
vectorstore, metadata = process_directory_for_files(base_directory)

# Create a retriever to query the FAISS index
faiss_retriever = vectorstore.as_retriever(search_kwargs={'k': 7,'fetch_k': 50})

# Function to perform retrieval
def get_context(query):
    retrieved_docs = faiss_retriever.invoke(query)
    files = []
    context = ""
    for result in retrieved_docs[:5]:
        files.append(result.metadata['file'])
    max_file_name = max(files,key=files.count)
    with open(max_file_name, 'r', encoding='utf8') as f:
            full_content = f.read()
    for result in retrieved_docs:
        chunk = result.page_content.split("<META>")[1]
        context += f"{result.metadata}" + " \n " + chunk + " \n "
        headers = result.metadata.get('header', 'N/A').replace(' Header ', '\n  - ')
    # print(f"Full content:{full_content}")


    context += full_content
    return context