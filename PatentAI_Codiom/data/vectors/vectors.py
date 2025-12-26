# VECTOR DATA - Data optimized for AI
# patent_embeddings.faiss - FAISS vector database (stores the vectors)
# embedding_cache.pkl - Pre-computed embeddings (to avoid re-calculating)
# vector_index/ - Indexed vectors for fast searching
# similarity_matrix/ - Similarity matrices

import pandas as pd
from sentence_transformers import SentenceTransformer # Library to generate embeddings from text
import faiss # Facebook AI Similarity Search library for efficient similarity search
import numpy as np
import os

# --- 1. Define File Paths ---
# Set paths relative to the project's root directory
# (This script should be run from the main folder: patent-ai-backend)
CSV_PATH = 'data/processed/patentAI.csv'
INDEX_PATH = 'data/vectors/patent_embeddings.faiss'

def create_vector_database():
    """
    Reads the 'patentAI.csv' file, converts the 'title' column into vectors (embeddings),
    and saves them as a FAISS index file.
    """
    
    # --- 2. Read Data ---
    print(f"Veri okunuyor: {CSV_PATH}")
    
    # Check if the CSV file actually exists at the specified path
    if not os.path.exists(CSV_PATH):
        print(f"HATA: CSV dosyası bulunamadı: {CSV_PATH}")
        print("Lütfen script'i projenin ana klasöründen (patent-ai-backend) çalıştırdığınızdan emin olun.")
        return

    try:
        # Load the dataset into a Pandas DataFrame
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"CSV okuma hatası: {e}")
        return
        
    # Ensure the 'title' column exists in the dataframe
    if 'title' not in df.columns:
        print("HATA: CSV dosyasında 'title' sütunu bulunamadı.")
        return
        
    # Clean the 'title' column: replace missing (NaN) values with empty strings to prevent errors
    df['title'] = df['title'].fillna('')
    
    # Convert the column data into a standard Python list
    texts = df['title'].tolist()
    
    print(f"Toplam {len(texts)} adet patent metni okundu.")

    # --- 3. Load Model and Generate Vectors ---
    # We use the 'all-MiniLM-L6-v2' model as specified in the Readme
    # This model maps sentences to a 384-dimensional dense vector space
    print("Dil modeli yükleniyor (all-MiniLM-L6-v2)... Bu işlem biraz sürebilir.")
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    print("Metinler vektörlere dönüştürülüyor (Embedding)...")
    # Encode the texts into vectors. show_progress_bar=True displays a progress bar in the terminal
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Convert embeddings to a NumPy array of type 'float32'
    # Important: FAISS specifically requires float32 (not float64) to work correctly
    embeddings = np.array(embeddings).astype('float32')

    print(f"Vektörler oluşturuldu. Boyut: {embeddings.shape}")

    # --- 4. Create and Save FAISS Index ---
    d = embeddings.shape[1]  # Dimension of the vectors (e.g., 384 for MiniLM)
    
    # Create a Flat L2 Index
    # Flat: Brute-force search (exact results, slower on massive data but fine here)
    # L2: Uses Euclidean distance to measure similarity
    index = faiss.IndexFlatL2(d) 
    
    print("Vektörler FAISS indeksine ekleniyor...")
    # Add the generated vectors to the FAISS index
    index.add(embeddings)
    
    print(f"İndeks FAISS dosyasına kaydediliyor: {INDEX_PATH}")
    # Write the index to a file on the disk so it can be loaded later without re-calculating
    faiss.write_index(index, INDEX_PATH)
    
    print("\nİşlem Tamamlandı!")
    print(f"'{INDEX_PATH}' başarıyla oluşturuldu.")

# If this script is run directly (not imported as a module), execute the function
if __name__ == "__main__":
    create_vector_database()