import time
import requests
import pandas as pd
from tqdm import tqdm

def extraer_resenas_api():
    url_api = "https://amazon-reviews-api-g5ae.onrender.com/reviews"
    limite_por_pagina = 1000
    offset_actual = 0
    total_estimado = 210000 
    
    datos_recolectados = []

    print("Iniciando la conexión con la API de reseñas...")
    barra_progreso = tqdm(total=total_estimado, desc="Descargando registros")

    while True:
        try:
            # Consulta a la API usando paginación por offset
            respuesta = requests.get(
                url_api, 
                params={"limit": limite_por_pagina, "offset": offset_actual}, 
                timeout=15
            )
            
            # Control de errores HTTP (ej. 502 Bad Gateway por saturación)
            if respuesta.status_code != 200:
                print(f"\n[Advertencia] Error HTTP {respuesta.status_code} en offset {offset_actual}. Reintentando...")
                time.sleep(5)
                continue
                
            json_data = respuesta.json()
            lote_resenas = json_data.get("data", [])
            
            # Condición de corte: si no hay más datos, terminamos el bucle
            if len(lote_resenas) == 0:
                break 
                
            datos_recolectados.extend(lote_resenas)
            offset_actual += limite_por_pagina
            
            barra_progreso.update(len(lote_resenas))
            
        except requests.exceptions.RequestException as e:
            # Captura cortes de internet o demoras en la respuesta del servidor
            print(f"\n[Error de red] {e}. Esperando 5 segundos antes del próximo intento...")
            time.sleep(5)

    barra_progreso.close()
    print(f"Descarga finalizada. Total de registros: {len(datos_recolectados)}")
    
    # Exportación del dataset crudo
    df_crudo = pd.DataFrame(datos_recolectados)
    path_salida = "datasets/amazon_reviews_raw.csv"
    df_crudo.to_csv(path_salida, index=False)
    print(f"Archivo guardado exitosamente en: {path_salida}")

if __name__ == "__main__":
    extraer_resenas_api()