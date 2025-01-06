import requests
import json


# Payload de ejemplo para enviar al servidor
payload = {
    "model": "llama2",
    "messages": [
        {"role": "user", "content": "who invented the wheel?"}
    ]
}

# Enviar la solicitud POST al servidor local
try:
    response = requests.post("http://localhost:11434/api/chat", json=payload)

    # Imprimir el contenido de la respuesta
    print("Response text:", response.text)

    # Intentar procesar los fragmentos de JSON
    response_text = response.text
    messages = []
    final_message = ""

    # Procesar cada línea de la respuesta
    for line in response_text.splitlines():
        try:
            # Intentamos parsear cada línea como JSON
            json_fragment = json.loads(line)
            content = json_fragment.get('message', {}).get('content', '')
            
            # Concatenar solo los fragmentos no vacíos y añadir espacio entre ellos
            if content.strip():
                final_message += content.strip() + " "  # Añadir un espacio

        except json.JSONDecodeError:
            # Si no es un fragmento JSON válido, continuamos
            continue

    # Eliminar el último espacio extra
    final_message = final_message.strip()

    # Mostrar la respuesta final
    if final_message:
        print("Respuesta final:", final_message)
    else:
        print("No se encontró una respuesta válida.")

except requests.exceptions.RequestException as e:
    print("Error en la solicitud HTTP:", e)
