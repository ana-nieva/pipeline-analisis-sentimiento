import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def ejecutar_embeddings():
    path_entrada = "datasets/amazon_reviews_clean.csv"
    
    print("Cargando dataset preprocesado...")
    df = pd.read_csv(path_entrada)
    df['cleaned_text'] = df['cleaned_text'].fillna("")
    
    # Para la demostración y no saturar la RAM, tomamos una muestra
    # En producción o si tenés tiempo, sacá el .sample()
    print("Cargando modelo de Hugging Face (puede tardar en descargar)...")
    modelo_hf = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generando Embeddings (esto tomará unos minutos)...")
    # Genera vectores densos (embeddings) de 384 dimensiones
    embeddings = modelo_hf.encode(df['cleaned_text'].tolist(), show_progress_bar=True)
    
    print("Guardando matriz de embeddings...")
    np.save("datasets/X_embeddings_hf.npy", embeddings)
    
    # Actualizamos el reporte
    with open("entregables/reporte_vocabulario.md", "a", encoding="utf-8") as f:
        f.write(f"| Hugging Face (MiniLM) | {embeddings.shape[1]} dimensiones (Vectores Densos) |\n")
        
    print("¡Embeddings generados y guardados con éxito!")

if __name__ == "__main__":
    ejecutar_embeddings()