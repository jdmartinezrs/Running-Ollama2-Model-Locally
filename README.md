
1. Es documentar en un readme como correr un modelo en local, con sus configuraciones y un ejemplo

# 1.Ejecución del Modelo Llama2 en un Servidor Local

Este proyecto realiza una solicitud HTTP `POST` a un servidor local en la URL `http://localhost:11434/api/chat`, enviando un payload que incluye un modelo llamado "llama2" y un mensaje con el contenido "who invented hiphop music?". Luego procesa la respuesta del servidor, que probablemente contiene una serie de fragmentos JSON, y los une en un único mensaje que se imprime al final.

Memory requirements
7b models generally require at least 8GB of RAM


## Detalles del Código

1. **Payload**: El payload es un diccionario de Python que contiene dos claves:
   - `"model"`: Un modelo que se utilizará para procesar la consulta (en este caso, `"llama2"`).
   - `"messages"`: Una lista con un solo mensaje que contiene un rol de usuario y una consulta.
2. **Solicitud POST**: Se usa el módulo `requests` para enviar una solicitud `POST` al servidor local. Los datos se envían en formato JSON (mediante el parámetro `json=payload`).
3. **Procesamiento de la respuesta**: La respuesta del servidor se recibe en `response.text`, que se divide en líneas. Cada línea se intenta convertir en un fragmento JSON. Si es válido, se extrae el contenido del mensaje y se concatena con el contenido de otros fragmentos, formando la respuesta final.

------

## Requisitos

Asegúrate de tener **Python** y **pip** (el gestor de paquetes de Python) instalados en tu sistema.

------

## 1. Instalar Dependencias Necesarias

El código utiliza la librería `requests` para enviar una solicitud HTTP y procesar la respuesta. Por lo tanto, **necesitarás tener esta librería instalada**.

Para instalar `requests`, ejecuta el siguiente comando:

```python
pip install requests
```

------

## 2. Asegúrate de Tener el Servidor Local Corriendo

El código realiza una solicitud `POST` a la URL `http://localhost:11434/api/chat`. Esto significa que necesitas tener un servidor corriendo localmente en ese puerto. Este servidor es el que proporciona el modelo `llama2` o un servicio que procesa la solicitud y genera la respuesta.

### ¿Cómo Asegurar que el Servidor Está Corriendo?

1. Si tienes configurado un **servidor local** que ejecuta el modelo `llama2`, asegúrate de que esté en funcionamiento en el puerto `11434`. Esto puede ser un servidor basado en Flask, FastAPI o alguna implementación personalizada que hayas hecho. Si el servidor no está corriendo, tu código no podrá hacer la solicitud correctamente.
2. Para **verificar si el servidor está corriendo** en ese puerto, puedes intentar ejecutar el siguiente comando:

```python
curl http://localhost:11434/
```

Si el servidor está activo, debería devolverte una respuesta, como un mensaje de bienvenida o algún tipo de información sobre el servidor. Si no hay respuesta, significa que el servidor no está corriendo y deberías iniciar el servicio que maneja las solicitudes del modelo `llama2`.

------

## 3. Modelo `Llama2` en el Servidor Local

El modelo `llama2` debe estar disponible en el servidor local. Asegúrate de que el servidor que estás ejecutando tenga acceso al modelo `llama2` y esté configurado correctamente para responder a las solicitudes en el endpoint `/api/chat`.

Dependiendo de cómo hayas configurado tu servidor, puede ser necesario que descargues el modelo, lo cargues en memoria y lo sirvas desde allí.

------

## 4. Comandos para la Configuración del Servidor Local

```python
`ollama pull llama2`
```

Este comando **no es necesario** para el código Python que has proporcionado, a menos que estés usando una herramienta específica para gestionar el modelo `llama2` y quieras descargarlo o asegurarte de que esté disponible localmente para el servidor.

------

## Ejecutar el Código

Una vez que hayas asegurado que el servidor esté corriendo y configurado correctamente, puedes ejecutar el script Python.

Para ejecutar el script, usa el siguiente comando:

```python
python localollama2.py
```

Este comando enviará la solicitud al servidor y procesará la respuesta. El resultado final se imprimirá en la terminal.



**RETORNA**

```
Hip hop music originated in the Bronx, New York City in the 1970s and early 1980s. The genre was created by African American and Latino youth who were living in poverty and facing social and economic challenges. The origins of hip hop can be traced back to four main elements: DJing, MCing (rapping), breaking (dancing), and graffiti art.

The earliest pioneers of hip hop include:

DJ Kool Herc: Known as the "father of hip hop," DJ Kool Herc was a Jamaican-American DJ who started hosting parties in the Bronx in the early 1970s. He is credited with inventing the breakbeat, which is a key element of hip hop music.

Afrika Bambaataa: Afrika Bambaataa is a hip-hop DJ and founder of the Zulu Nation, a group that helped popularize hip hop culture. He is known for his innovative mixing techniques and his efforts to promote unity and diversity within the hip-hop community.

Grandmaster Flash: Grandmaster Flash is a pioneering DJ and producer who is known for his innovative use of scratching and sampling. He is also credited with developing the "backspin," a technique that involves reversing a recorded segment of music to create a new sound.

The Sugarhill Gang: The Sugarhill Gang is a hip-hop group from Newark, New Jersey, credited with creating the first hip-hop single, "Rapper's Delight," in 1979.

Kurtis Blow: Kurtis Blow is a hip-hop artist and producer who was one of the first to record and release hip-hop music. His debut album, "Kurtis Blow," was released in 1980 and is considered one of the earliest hip-hop albums.

Run-D.M.C.: Run-D.M.C. is a hip-hop group from Queens, New York, known for their innovative use of sampling and groundbreaking music videos. They are credited with helping to shape the genre's development and evolution. Their contributions helped shape the genre into what it is today, and their legacy continues to inspire new generations of hip-hop artists.
```

![ollama](https://i.pinimg.com/1200x/50/ca/c8/50cac8aec3b153c6279458797b9aa938.jpg)


2. Es usando la API crear como una interfaz de chat donde pueda escribir a un modelo y este te responda.

## 2.Proyecto aplicación

Chat con API de Modelo de Lenguaje Este proyecto crea una interfaz de chat interactiva donde los usuarios pueden escribir mensajes y obtener respuestas generadas por un modelo de lenguaje. Utilizando Streamlit, esta aplicación permite comunicarte con el modelo de forma sencilla y eficaz.

```
Es recomendable crear un entorno virtual para evitar conflictos con otras dependencias de Python. Puedes hacerlo de la siguiente manera:

python -m venv venv
```

Una vez creado el entorno, actívalo con el siguiente comando:

- En Windows:

  ```
  venv\Scripts\activate
  ```

## 1. Instalar los requisitos

Instala las dependencias necesarias utilizando el archivo `requirements.txt`. Para ello, ejecuta el siguiente comando:

```
pip install -r requirements.txt
```

## 2. Ejecutar la aplicación

Una vez instaladas las dependencias, puedes ejecutar la aplicación con el siguiente comando:

```
streamlit run app.py
```

Esto abrirá la aplicación en tu navegador predeterminado.

## 3. Imagen de referencia

Aquí puedes ver una imagen relacionada con el proyecto:

![llama2 chat](https://i.pinimg.com/1200x/29/db/93/29db93cf3d9bd177519c591a87115afa.jpg)


3. Usando múltiples moldes (recomendamos la API) comunicarlos entre ellos, por ejemplo que un LLM genere el prompt para un modelo de generación de imágenes.


Uso en Google Colab

Este código está diseñado para ejecutarse de manera eficiente en Google Colab, aprovechando el uso de GPU. Sigue los siguientes pasos para ejecutar el código correctamente:

**Generación de Prompt Mejorado (con Phi):**

- Se utiliza el modelo `microsoft/Phi-3.5-mini-instruct` para generar un texto mejorado (prompt) a partir de un texto inicial. Esto se hace con el fin de crear un prompt más detallado y efectivo para la generación de imágenes.

**Generación de Imágenes (con Stable Diffusion):**

- Una vez que se tiene el prompt generado, este se utiliza para crear una imagen utilizando el modelo de **Stable Diffusion** (versión 1.5).

- **`transformers`**: Carga y usa modelos preentrenados de NLP (como Phi).
- **`diffusers`**: Genera imágenes a partir de texto usando modelos de difusión (Stable Diffusion).
- **`torch`**: Utiliza PyTorch para operaciones de aprendizaje profundo y acceso a GPU.
- **`Pillow`**: Manipula y guarda imágenes.
- **`bitsandbytes`**: Optimiza el uso de memoria al cargar modelos grandes en 8 bits.
- **`accelerate`**: Facilita la ejecución eficiente en múltiples dispositivos (GPU, CPU).

```python
!pip install transformers diffusers torch Pillow bitsandbytes accelerate
```

1. **Carga y configura el modelo Phi** de Microsoft para generar texto mejorado (prompt) usando el modelo `AutoModelForCausalLM` y `AutoTokenizer`.
2. **Configura Stable Diffusion** para generar imágenes a partir de un texto dado usando la clase `StableDiffusionPipeline`.
3. Métodos
   - `generate_prompt`: Genera un prompt mejorado a partir de un texto de entrada utilizando Phi.
   - `generate_image`: Usa el prompt para generar una imagen con Stable Diffusion.

Utiliza GPU si está disponible.



```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from diffusers import StableDiffusionPipeline  # Asegúrate de que esta línea esté incluida
from PIL import Image

class ImageGenerationPipeline:
    def __init__(self):
        # Phi model setup
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )

        # Load Phi tokenizer and model
        self.phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
        self.phi_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3.5-mini-instruct",
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.float16
        )

        # Stable Diffusion setup
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to("cuda")  # Usa "cpu" si no tienes GPU

    def generate_prompt(self, input_text, max_length=100):
        # Generate enhanced prompt using Phi model
        inputs = self.phi_tokenizer(input_text, return_tensors="pt").to(self.phi_model.device)
        outputs = self.phi_model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.phi_tokenizer.eos_token_id
        )
        return self.phi_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_image(self, prompt):
        # Generate image using Stable Diffusion
        image = self.sd_pipeline(prompt).images[0]
        return image

```

**Crea una instancia de `ImageGenerationPipeline`**: Se inicializa el pipeline que contiene los modelos Phi y Stable Diffusion.

**Define un texto de entrada (`input_text`)**: Se especifica el texto base que se usará para generar un prompt mejorado. En este caso, el texto es "Give me a prompt to create an image of a magic beach."

**Genera un prompt mejorado (`enhanced_prompt`)**: Usando el método `generate_prompt`, el pipeline genera un prompt más detallado y descriptivo a partir del texto base dado.

**Muestra el prompt generado**: Imprime el prompt mejorado que fue generado por el modelo Phi.

```python
# Crear una instancia del pipeline
pipeline = ImageGenerationPipeline()

# Define el texto original para generar el prompt
input_text = "Give me a prompt to create an image of a magic beach."

# Generar el prompt mejorado
enhanced_prompt = pipeline.generate_prompt(input_text)

# Mostrar el prompt generado
print(f"Prompt Generado: {enhanced_prompt}")
```

El prompt generado en este caso es:

```
\#  Answer Prompt for Image Generation: Create a captivating digital artwork that depicts a magical beach scene. The beach is bathed in ethereal light, emanating from a setting sun that casts a warm golden hue across the scene. The sand sparkles with a touch of magic, shimmering with tiny, iridescent crystals that twink
```

**Genera la imagen**: Utiliza el **prompt mejorado** (`enhanced_prompt`) para generar una imagen usando el método `generate_image` del pipeline.

**Muestra la ruta de guardado**: Imprime la ruta en la que la imagen ha sido guardada, para que el usuario sepa dónde encontrarla en su entorno.

```python
import os  # Import the os module

# Usamos el prompt generado para crear la imagen
image = pipeline.generate_image(enhanced_prompt)


image.show()

download_folder = './Descargas'
os.makedirs(download_folder, exist_ok=True)


image_path = os.path.join(download_folder, "beach.png")
image.save(image_path)

# Mostrar la ruta donde se guardó la imagen
print(f"Imagen guardada en: {image_path}")
```

**Importa la librería `files` de Google Colab**: La librería `files` se utiliza para interactuar con archivos dentro de Google Colab.

**Descarga la imagen**: `files.download(image_path)` inicia la descarga del archivo ubicado en `image_path`, que es la ruta del archivo de imagen guardado en el directorio `Descargas` (por ejemplo, `beach.png`).

- Esto hará que el navegador del usuario descargue el archivo directamente a la carpeta predeterminada de descargas de su computadora.

  ```python
  from google.colab import files
  
  # Descargar la imagen a tu computadora
  files.download(image_path)  # Esto descargará la imagen a tu computadora.
  ```

  

  **##  Imagen de referencia**

  Aquí puedes ver la imágen generada, basada en el prompt:

  
  ![Beach image](https://i.pinimg.com/1200x/80/27/74/802774462e3cc682e57625718d6e38ad.jpg)