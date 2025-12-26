# app/ai_models/llm_service.py

# Import Facebook AI Similarity Search (FAISS) library; used for fast vector search.
import faiss

# Import Pandas library for data manipulation and reading CSV files.
import pandas as pd

# Import NumPy library for numerical operations and array management.
import numpy as np

# Import SentenceTransformer to convert texts into vectors (numerical arrays).
from sentence_transformers import SentenceTransformer

# Import 'os' module to check file paths and perform operating system operations.
import os

# --- 1. DEFINE FILE PATHS AND PARAMETERS ---
# File path for the pre-built vector database (FAISS index).
INDEX_PATH = 'data/vectors/patent_embeddings.faiss'

# Path to the processed CSV file containing patent titles and texts.
CSV_PATH = 'data/processed/patentAI.csv'

# Name of the AI model used to convert text to vectors.
# 'all-MiniLM-L6-v2' is often preferred because it is fast and lightweight.
MODEL_NAME = 'all-MiniLM-L6-v2'

# The number determining how many similar patents to retrieve in the search results.
TOP_K = 3 

# --- 2. LOAD MODELS AND DATA GLOBALLY ---
# This function ensures models are loaded ONLY ONCE when the app starts, 
# not every time the API is called. This is critical for performance.

def load_data():
    """Loads data and AI models into computer memory (RAM)."""
    
    # Print output to the console to inform the user.
    print(f"Loading FAISS index: {INDEX_PATH}...")
    
    # Check if the FAISS file exists at the specified path.
    if not os.path.exists(INDEX_PATH):
        # Print error message if file does not exist.
        print(f"ERROR: FAISS index not found: {INDEX_PATH}")
        print("Please ensure you have run the 'data/vectors/vectors.py' script first.")
        # Return None since nothing could be loaded.
        return None, None, None
    
    # Read the FAISS index from disk into memory.
    index = faiss.read_index(INDEX_PATH)
    
    # Inform the user about CSV loading.
    print(f"Loading CSV data: {CSV_PATH}...")
    
    # Check for the existence of the CSV file.
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV file not found: {CSV_PATH}")
        return None, None, None
            
    # FAISS search only gives us an 'ID' (e.g., 5, 12, 100).
    # We need to read the CSV to know which patent title corresponds to that ID.
    try:
        # Read the CSV file as a Pandas DataFrame.
        df = pd.read_csv(CSV_PATH)
        
        # Check for the existence of the 'title' column; error if missing.
        if 'title' not in df.columns:
            print("ERROR: 'title' column not found in CSV file.")
            return None, None, None
            
        # Fill empty (NaN) titles with an empty string '' to prevent errors.
        df['title'] = df['title'].fillna('')
        
        # Convert titles to a Python list (for faster access).
        patent_titles = df['title'].tolist()
        
    except Exception as e:
        # Print to console if an unexpected error occurs during reading.
        print(f"CSV reading error: {e}")
        return None, None, None

    # Inform the user about language model loading.
    print(f"Loading language model: {MODEL_NAME}...")
    
    # Download or load the SentenceTransformer model from cache.
    model = SentenceTransformer(MODEL_NAME)
    
    # Report that the loading process was successful.
    print("All models and data loaded successfully.")
    
    # Return the three important components (index, model, title list).
    return index, model, patent_titles

# This line runs when the application starts and calls the 'load_data' function.
# Results are assigned to global variables so they are accessible from everywhere.
faiss_index, sentence_model, patent_titles = load_data()

# --- 3. UPDATED ANALYSIS FUNCTION ---

def get_analysis(text: str) -> dict:
    """
    Takes text from the user, converts it to a vector, and searches for similar patents.
    Note: Used 'def' instead of 'async def' because this operation is CPU-intensive (blocking operation).
    """
    
    # If models failed to load in the step above (e.g., file missing), stop the process.
    if faiss_index is None:
        return {
            "novelty_score": 0.0,
            "summary": "ERROR: Server-side model or data could not be loaded. Please contact the administrator."
        }

    # Print the first 20 characters of the incoming text for debugging.
    print(f"Real Similarity Search Started... (Text: {text[:20]}...)")

    # STEP 1: Convert the incoming text into a vector (numbers).
    # The model takes the text and converts it into a 384-dimensional list (depending on model).
    query_vector = sentence_model.encode([text])
    
    # FAISS works with NumPy arrays in float32 format, so we convert it.
    query_vector = np.array(query_vector).astype('float32')

    # STEP 2: Search for the most similar vectors in the FAISS database.
    # The search function returns two things:
    # distances: The distance of found patents to our text (Smaller means more similar).
    # indices: The row numbers (IDs) of the found patents in the database.
    distances, indices = faiss_index.search(query_vector, TOP_K)
    
    # STEP 3: Convert found IDs (indices) into actual patent titles.
    found_patents = []
    
    # Loop through each ID in the indices[0] list.
    for i in indices[0]:
        # If ID is -1, it means FAISS couldn't find a valid result, skip it.
        if i != -1: 
            # Use the ID to get the title from the 'patent_titles' list and add to found list.
            found_patents.append(patent_titles[i])
    
    # Print to console how many similar patents were found.
    print(f"Found {len(found_patents)} similar patents.")

    # STEP 4: Format the result to be presented to the user.
    # If the list is empty (no similar patents):
    if not found_patents:
        summary = "Analysis complete. No similar patents were found in our database. This may indicate high novelty potential!"
    else:
        # If similar patents exist, convert them into a bulleted string.
        # "\n* ".join(...) turns the list into a bulleted list one under another.
        summary = "Analysis complete. Patents similar to your idea were found:\n\n* " + "\n* ".join(found_patents)

    # Calculate a simple score.
    # distances[0][0] is the distance of the nearest patent.
    # The smaller the distance, the higher the similarity. 
    # The formula (1.0 - distance) provides a simple "novelty" score estimate.
    novelty_score = (1.0 - distances[0][0]) * 100 

    # Return the results as a dictionary.
    return {
        "novelty_score": novelty_score,
        "summary": summary
    }