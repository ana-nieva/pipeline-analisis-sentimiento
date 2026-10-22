import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse

def ejecutar_vectorizacion():
    path_entrada = "datasets/amazon_reviews_clean.csv"
    
    print("Cargando dataset preprocesado...")
    df = pd.read_csv(path_entrada)
    
    # Nos aseguramos de no tener valores nulos post-limpieza
    df['cleaned_text'] = df['cleaned_text'].fillna("")
    
    print("Realizando partición Train/Test (80/20)...")
    # Estratificamos para mantener la misma proporción de clases positivas y negativas
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['cleaned_text'], 
        df['sentiment'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['sentiment']
    )
    
    print("Guardando etiquetas (y_train, y_test)...")
    np.save("datasets/y_train.npy", y_train)
    np.save("datasets/y_test.npy", y_test)
    
    reporte_vocabulario = {}
    
    # 1. Bag of Words (BoW)
    print("\nEntrenando vectorizador Bag of Words (BoW)...")
    bow_vectorizer = CountVectorizer(min_df=5)
    X_train_bow = bow_vectorizer.fit_transform(X_train_text)
    X_test_bow = bow_vectorizer.transform(X_test_text)
    
    reporte_vocabulario['BoW'] = len(bow_vectorizer.vocabulary_)
    sparse.save_npz("datasets/X_train_bow.npz", X_train_bow)
    sparse.save_npz("datasets/X_test_bow.npz", X_test_bow)
    
    # 2. TF-IDF Unigrama
    print("Entrenando vectorizador TF-IDF (Unigrama)...")
    tfidf_uni_vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1, 1))
    X_train_tfidf_uni = tfidf_uni_vectorizer.fit_transform(X_train_text)
    X_test_tfidf_uni = tfidf_uni_vectorizer.transform(X_test_text)
    
    reporte_vocabulario['TF-IDF Unigrama'] = len(tfidf_uni_vectorizer.vocabulary_)
    sparse.save_npz("datasets/X_train_tfidf_uni.npz", X_train_tfidf_uni)
    sparse.save_npz("datasets/X_test_tfidf_uni.npz", X_test_tfidf_uni)
    
    # 3. TF-IDF Bigrama
    print("Entrenando vectorizador TF-IDF (Bigrama)...")
    tfidf_bi_vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1, 2))
    X_train_tfidf_bi = tfidf_bi_vectorizer.fit_transform(X_train_text)
    X_test_tfidf_bi = tfidf_bi_vectorizer.transform(X_test_text)
    
    reporte_vocabulario['TF-IDF Bigrama'] = len(tfidf_bi_vectorizer.vocabulary_)
    sparse.save_npz("datasets/X_train_tfidf_bi.npz", X_train_tfidf_bi)
    sparse.save_npz("datasets/X_test_tfidf_bi.npz", X_test_tfidf_bi)
    
    print("\nGenerando reporte de entregables...")
    path_reporte = "entregables/reporte_vocabulario.md"
    with open(path_reporte, "w", encoding="utf-8") as f:
        f.write("# Reporte de Tamaños de Vocabulario\n\n")
        f.write("Tamaño del vocabulario generado por cada técnica ajustada sobre el conjunto de entrenamiento:\n\n")
        f.write("| Técnica de Vectorización | Tamaño del Vocabulario |\n")
        f.write("| :--- | :--- |\n")
        for tecnica, tamaño in reporte_vocabulario.items():
            f.write(f"| {tecnica} | {tamaño:,} tokens |\n")
            
    print(f"¡Vectorización finalizada! Reporte guardado en {path_reporte}")

if __name__ == "__main__":
    ejecutar_vectorizacion()