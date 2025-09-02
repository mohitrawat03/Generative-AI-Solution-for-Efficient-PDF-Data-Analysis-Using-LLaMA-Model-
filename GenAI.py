# Install required packages
#!pip install transformers torch pandas numpy huggingface_hub chromadb PyPDF2

# Import necessary libraries
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from huggingface_hub import login
import chromadb
import PyPDF2
from io import StringIO
import transformers

# Seed for reproducibility
np.random.seed(400)

# Initialize Hugging Face login to access models
login()

# Define the model name and initialize the tokenizer
model_name = 'meta-llama/Llama-2-13b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the model and set up the text-generation pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model = pipeline.model

# Initialize ChromaDB client
client = chromadb.Client()
index_name = "pdf_data"
collection = client.create_collection(index_name)

def compute_embedding(text):
    """Compute the embedding for a given text using the LLaMA model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the appropriate device
    inputs = tokenizer(text, return_tensors="pt").to(device)  # Tokenize the input text
    with torch.no_grad():
        outputs = model(**inputs)  # Generate model outputs
        logits = outputs.logits  # Extract logits from the output
        # Compute the mean of logits to get a single embedding vector
        embedding = logits.mean(dim=1).cpu().numpy().flatten()
    return embedding.tolist()  # Convert embedding to a list

def read_pdf(file_path):
    """Read text from a PDF file."""
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)  # Create a PDF reader object
        text = ""
        for page_num in range(len(pdf_reader.pages)):  # Iterate through each page
            page = pdf_reader.pages[page_num]
            text += page.extract_text() if page.extract_text() else ""  # Extract text from the page
    return text

# Path to your PDF file
pdf_path = "A_B Testing - Notes.pdf"

# Read data from PDF
pdf_text = read_pdf(pdf_path)

# Convert the string data into a DataFrame
data = StringIO(pdf_text)  # Use StringIO to convert text to a file-like object
try:
    df = pd.read_csv(data, on_bad_lines='skip')  # Read the data into a DataFrame, skipping bad lines
    print("DataFrame loaded successfully:")
    print(df.head())  # Print the first few rows of the DataFrame
except pd.errors.ParserError as e:
    print(f"ParserError: {e}")  # Handle parsing errors
except Exception as e:
    print(f"An error occurred: {e}")  # Handle other exceptions

# Embed each row of the DataFrame and store in ChromaDB
for index, row in df.iterrows():
    text = f"Year: {row['Year']}, Sales: {row['Sales']}, Profit: {row['Profit']}, Margin: {row['Margin']:.2f}%"
    embedding = compute_embedding(text)  # Compute embedding for the text
    collection.add(
        ids=[str(index)],
        embeddings=[embedding]
    )

def generate_response(prompt):
    """Generate a response to a given prompt using the LLaMA model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the appropriate device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Tokenize the prompt
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_length=3000,  # Adjust length as needed
            num_return_sequences=1
        )
    response = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)  # Decode the output
    return response

# Example queries
queries = [
    "What were the sales in 2021?",
    "What was the profit in 2022?",
    "Which year had the highest sales?",
]

for query_text in queries:
    print(f"\nQuery: {query_text}")
    query_embedding = compute_embedding(query_text)  # Compute embedding for the query
    # Determine the number of elements in the collection
    num_elements = len(collection.get()["ids"])
    n_results = min(10, num_elements)  # Limit results to the number of available elements
    # Search for the most similar data in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    # Print and process the results
    print("Query results:")
    print(results)
    retrieved_texts = []
    for doc in results['documents'][0]:
        retrieved_texts.append(doc)  # Collect retrieved texts
    combined_text = " ".join(retrieved_texts)  # Combine texts if necessary
    # Generate and print the response based on the combined results
    response = generate_response(f"Here is the data: {combined_text}. Answer the following question: {query_text}")
    print("Generated Response:")
    print(response)
