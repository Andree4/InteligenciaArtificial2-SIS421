import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianTokenizer, MarianMTModel

# 1. Configurar el modelo y tokenizer CodeT5
def get_codet5_model_and_tokenizer():
    """
    Cargar el modelo y tokenizer de CodeT5.
    """
    model_name = "Salesforce/codet5-base"  # Modelo para generación de código
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

# 2. Preprocesar el texto
def preprocess_text(prompt, tokenizer, max_input_length=128):
    """
    Preprocesar el texto para el modelo.
    """
    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_input_length, truncation=True, padding="max_length")
    return inputs

# 3. Generar código a partir del prompt
def generate_code_from_prompt(prompt, model, tokenizer, max_output_length=256):
    """
    Generar código a partir de un prompt en lenguaje natural.
    """
    print(f"Prompt: {prompt}")
    # Preprocesar el texto
    inputs = preprocess_text(prompt, tokenizer)
    # Generar código
    outputs = model.generate(
        **inputs,
        max_length=max_output_length,
        num_beams=5,
        early_stopping=True,
        temperature=0.7
    )
    # Decodificar el código generado
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Código generado:\n{generated_code}")
    return generated_code

# 4. Traducción del español al inglés
def translate_to_english(text, model_name="Helsinki-NLP/opus-mt-es-en"):
    """
    Traducir un texto en español al inglés.
    """
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenización y generación de la traducción
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    print(f"Texto traducido al inglés: {translated_text}")
    return translated_text

# 5. Guardar el modelo
def export_model_and_tokenizer(model, tokenizer, save_directory):
    """
    Guardar el modelo y el tokenizer en el directorio especificado.
    """
    os.makedirs(save_directory, exist_ok=True)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Modelo y tokenizer guardados en: {save_directory}")

# Exportar el modelo
if __name__ == "__main__":
    # Configurar el modelo y tokenizer de CodeT5
    model, tokenizer = get_codet5_model_and_tokenizer()

    # Directorio para guardar el modelo
    save_dir = "./exported_codet5_model"
    export_model_and_tokenizer(model, tokenizer, save_dir)
