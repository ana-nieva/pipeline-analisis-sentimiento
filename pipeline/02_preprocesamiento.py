import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Descargamos los modelos de lenguaje de NLTK la primera vez que se corre
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

def expandir_contracciones(texto):
    # Expandimos contracciones negativas para evitar que la puntuación elimine el "not"
    texto = re.sub(r"won\'t", "will not", texto)
    texto = re.sub(r"can\'t", "can not", texto)
    texto = re.sub(r"n\'t", " not", texto)
    return texto

def limpiar_texto(texto, stop_words, lematizador):
    if not isinstance(texto, str):
        return ""
        
    # Llevamos todo a minúsculas y separamos contracciones
    texto = texto.lower()
    texto = expandir_contracciones(texto)
    
    # Eliminamos caracteres especiales, números y puntuación
    texto = re.sub(r'[^a-z\s]', '', texto)
    
    # Tokenización
    tokens = word_tokenize(texto)
    
    # Lematizamos y filtramos stopwords (ignorando tokens de una sola letra)
    tokens_limpios = [
        lematizador.lemmatize(palabra) 
        for palabra in tokens 
        if palabra not in stop_words and len(palabra) > 1
    ]
    
    return " ".join(tokens_limpios)

def ejecutar_preprocesamiento():
    path_entrada = "datasets/amazon_reviews_raw.csv"
    path_salida = "datasets/amazon_reviews_clean.csv"
    
    print(f"Cargando dataset original desde {path_entrada}...")
    df = pd.read_csv(path_entrada)
    
    print("Descartando reseñas neutrales y binarizando etiquetas...")
    # Nos quedamos solo con las positivas y negativas (descartamos label 2)
    df = df[df['label'] != 2].copy()
    
    # Binarización: 0 para negativas (0 y 1), 1 para positivas (3 y 4)
    df['sentiment'] = df['label'].apply(lambda x: 1 if x > 2 else 0)
    
    # Configuramos el lematizador y las stopwords
    lematizador = WordNetLemmatizer()
    stop_words_ingles = set(stopwords.words('english'))
    
    # Excluimos las negaciones de la lista de stopwords
    palabras_a_conservar = {'not', 'no', 'nor'}
    stop_words_ingles = stop_words_ingles - palabras_a_conservar
    
    # Activamos progress_apply de tqdm para ver el avance en pandas
    tqdm.pandas(desc="Procesando texto")
    
    print("Iniciando limpieza y lematización (esto tomará unos minutos)...")
    # Generamos la nueva columna cleaned_text manteniendo la original intacta
    df['cleaned_text'] = df['text'].progress_apply(
        lambda x: limpiar_texto(x, stop_words_ingles, lematizador)
    )
    
    print(f"Exportando dataset procesado a {path_salida}...")
    df.to_csv(path_salida, index=False)
    print("¡Preprocesamiento completado con éxito!")

if __name__ == "__main__":
    ejecutar_preprocesamiento()