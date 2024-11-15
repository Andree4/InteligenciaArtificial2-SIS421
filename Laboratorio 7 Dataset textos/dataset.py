import glob
import pandas as pd

# Ruta donde están tus archivos de normativa universitaria
ruta_archivos = "TextosLimpios/*.txt"  # Cambia esta ruta

# Listas para almacenar los segmentos y las etiquetas ingresadas manualmente
segmentos = []
categorias = []

# Definir el tamaño del segmento (en número de caracteres o palabras)
segment_size = 10000  # Puedes ajustar este valor según la longitud de los segmentos que desees

# Instrucciones para el etiquetado
print("Clasificación por segmento: Ingrese 'Academico', 'Administrativo', 'Conducta', o 'Recursos' para cada fragmento.\n")

# Leer cada archivo y dividir en segmentos
for archivo in glob.glob(ruta_archivos):
    with open(archivo, 'r', encoding='utf-8') as f:
        contenido = f.read().replace('\n', ' ').strip()
        
        # Dividir el contenido en segmentos
        for i in range(0, len(contenido), segment_size):
            segmento = contenido[i:i+segment_size]
            print(f"\nSegmento del archivo {archivo} (muestra {i//segment_size + 1}):\n")
            print(segmento)
            
            # Ingresar la categoría para el segmento
            etiqueta = input("\nIngrese la categoría para este segmento (Academico, Administrativo, Conducta, Recursos): ").strip()
            
            # Validar entrada
            while etiqueta not in ["Academico", "Administrativo", "Conducta", "Recursos"]:
                print("Entrada inválida. Ingrese una de las opciones válidas: Academico, Administrativo, Conducta, Recursos")
                etiqueta = input().strip()
            
            # Agregar el segmento y la etiqueta a las listas
            segmentos.append(segmento)
            categorias.append(etiqueta)

# Crear el DataFrame con los segmentos y las etiquetas ingresadas manualmente
df = pd.DataFrame({
    'Segmento': segmentos,
    'Categoria': categorias
})

# Convertir las categorías en etiquetas numéricas
label_mapping = {"Academico": 0, "Administrativo": 1, "Conducta": 2, "Recursos": 3}
df['Label'] = df['Categoria'].map(label_mapping)

# Guardar el dataset en un archivo CSV
df.to_csv('CorpusClasificado.csv', index=False, encoding='utf-8')

print("Etiquetado completo. Dataset guardado como 'CorpusClasificado.csv'")
