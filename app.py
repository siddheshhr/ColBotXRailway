
import os
import uvicorn
from fastapi import FastAPI, Request
from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.middleware.cors import CORSMiddleware
import torch
from typing import Tuple, List

# HuggingFace Authentication (if necessary)
os.environ["HF_TOKEN"] = "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"

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load the model and tokenizer only once during startup
model_name = "AnishaShende/tinyllama-unsloth-merged_1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# MongoDB connection and initialization of the vector store
client = MongoClient("mongodb+srv://anisha22320184:YT0nSJqiOneznlFW@cluster0.mxq6c.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["scrape_data"]
collection = db["acade_calen"]

# Initialize the embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model_name="baai/bge-large-en-v1.5")
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embedding_model,
    index_name="vector_index",
    embedding_key="embeddings",
    text_key="answer",
    relevance_score_fn="cosine",
)




# Function to retrieve context using MongoDB vector store with similarity scores
def get_context_from_vector_store(query: str, top_k: int = 3) -> List[Tuple[str, float]]:
    results = vector_store.similarity_search_with_score(query, k=top_k)
    context = ""

    if not results:
        return []  # Return an empty list if no results are found

    # Print each document's content and score
    for doc, score in results:
        print(f"docs: {doc}")
        print(f"SIM= {score:3f}")
        context += doc.page_content + " "
    print(f"context: {context}")

    # Collect the top-k results along with their similarity scores
    context_with_scores = [
        (doc.page_content, score)  # Accessing the score directly from the doc object
        for doc, score in results
    ]

    return context_with_scores



# LLM inference function that handles context relevance threshold
def llm_inference(question: str, max_length: int = 500, threshold: float = 0.7) -> str:
    # Retrieve context and their similarity scores
    context_with_scores = get_context_from_vector_store(question)

    relevant_context = None

    # Iterate through context and scores to find the most relevant one
    for context, score in context_with_scores:
        if score >= threshold:
            relevant_context = context
            break

    # Check if we found a relevant context
    if relevant_context:
        # If context is relevant, use it in the prompt
        input_prompt = f"""<s>[INST]
        Instruction: The answer must be based only on the provided context.
        Do not use any prior knowledge or assumptions.

        Context: {relevant_context}

        Question: {question}
        [/INST]
        """
    else:
        # If context is not relevant, fall back to the fine-tuned model's internal knowledge
        input_prompt = f"""<s>[INST]
        Instruction: Answer the following question based on your knowledge. If relevant, use any available data you have been trained on.

        Question: {question}
        [/INST]
        """

    # Generate the answer using the model
    inputs = tokenizer(input_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(inputs["input_ids"], max_length=max_length)

    # Decode and clean the response
    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract just the answer part
    answer_start = decoded_output.find('Answer:')
    if answer_start != -1:
        cleaned_answer = decoded_output[answer_start + len('Answer:'):].strip()
    else:
        cleaned_answer = decoded_output.strip()

    return cleaned_answer

# API route for chatbot responses
@app.post("/chat")
async def get_response(request: Request):
    data = await request.json()
    user_input = data.get("input_text")

    if not user_input:
        return {"error": "No input_text provided"}

    # Generate a response using the LLM
    generated_answer = llm_inference(user_input)

    print(f"User Input: {user_input}, Bot Response: {generated_answer}")  # Debugging log

    return {"input": user_input, "response": generated_answer}

# Root route for testing if the server is running
@app.get("/")
async def root():
    return {"message": "Chatbot is running!"}

# Main function to run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT",8000)))
