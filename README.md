# Guía de Ejecución del Modelo Local Llama2 y Generación de Imágenes con Phi + Stable Diffusion

Este documento te guiará en el proceso de ejecutar un modelo Llama2 localmente en un servidor, interactuar con él a través de una API y crear imágenes utilizando una combinación de modelos de lenguaje y generación visual.

------

## **1. Ejecución del Modelo Llama2 Local**

Este proyecto realiza una solicitud HTTP `POST` a un servidor local en la URL `http://localhost:11434/api/chat`, enviando un modelo `llama2` para procesar la consulta "¿Who invented hip-hop music?" y luego muestra la respuesta procesada.

### Requisitos del Sistema

- **Memoria RAM**: Se recomienda al menos **8 GB de RAM** para modelos de 7B.
- **Python**: Versión 3.6 o superior.

### Configuración del Entorno

Se recomienda crear un entorno virtual para evitar conflictos con otras dependencias. Para hacerlo, sigue estos pasos:

1. **Crear un entorno virtual**:

   ```
   python -m venv venv
   ```

2. **Activar el entorno virtual**:

   - En **Windows**:

     ```
     venv\Scripts\activate
     ```

   - En **Linux/macOS**:

     ```
     source venv/bin/activate
     ```

### Instalación de Dependencias

Para instalar las dependencias necesarias, ejecuta el siguiente comando:

```
pip install -r requirements.txt
```

### Ejecutando el Servidor Local

1. **Verifica que tu servidor local está corriendo**:

   Asegúrate de tener un servidor en funcionamiento en el puerto `11434`. Puedes probarlo con el siguiente comando:

   ```
   curl http://localhost:11434/
   ```

   Si el servidor está activo, recibirás una respuesta, como un mensaje de bienvenida:"Ollama is running"

2. **Arrancar el servidor que aloja el modelo `llama2`**:

   ```
   ollama pull llama2
   ```

### Ejecución del Script

Una vez que el servidor esté en marcha, puedes ejecutar el script de Python para hacer la solicitud y obtener la respuesta:

```
python localollama2.py
```

Esto enviará una solicitud al servidor y te devolverá una respuesta en tu terminal.

**Ejemplo de respuesta obtenida**:

```
Hip hop music originated in the Bronx, New York City in the 1970s and early 1980s. The genre was created by African American and Latino youth who were living in poverty and facing social and economic challenges. The origins of hip hop can be traced back to four main elements: DJing, MCing (rapping), breaking (dancing), and graffiti art.

The earliest pioneers of hip hop include:

DJ Kool Herc: Known as the "father of hip hop," DJ Kool Herc was a Jamaican-American DJ who started hosting parties in the Bronx in the early 1970s. He is credited with inventing the breakbeat, which is a key element of hip hop music.

Afrika Bambaataa: Afrika Bambaataa is a hip-hop DJ and founder of the Zulu Nation, a group that helped popularize hip hop culture. He is known for his innovative mixing techniques and his efforts to promote unity and diversity within the hip-hop community.

Grandmaster Flash: Grandmaster Flash is a pioneering DJ and producer who is known for his innovative use of scratching and sampling. He is also credited with developing the "backspin," a technique that involves reversing a recorded segment of music to create a new sound.

The Sugarhill Gang: The Sugarhill Gang is a hip-hop group from Newark, New Jersey, credited with creating the first hip-hop single, "Rapper's Delight," in 1979.

Kurtis Blow: Kurtis Blow is a hip-hop artist and producer who was one of the first to record and release hip-hop music. His debut album, "Kurtis Blow," was released in 1980 and is considered one of the earliest hip-hop albums.
```

------



## **2. Aplicación de Chat con la API del Modelo de Lenguaje**

Este proyecto utiliza **Streamlit** para crear una interfaz de chat donde puedes interactuar con el modelo de lenguaje y obtener respuestas en tiempo real.

### Ejecutar la Aplicación de Chat

Una vez que las dependencias estén instaladas, ejecuta la aplicación con el siguiente comando:

```python
streamlit run app.py
```

Esto abrirá la aplicación en tu navegador predeterminado y podrás comenzar a interactuar con el modelo de lenguaje.

![llama2 chat](https://i.pinimg.com/1200x/29/db/93/29db93cf3d9bd177519c591a87115afa.jpg)



------

## **3. Generación de Imágenes Usando Modelos de Lenguaje y Estilo Visual**

\- Se utiliza el modelo `microsoft/Phi-3.5-mini-instruct` para generar un texto mejorado (prompt) a partir de un texto inicial. Esto se hace con el fin de crear un prompt más detallado y efectivo para la generación de imágenes.

***\*Generación de Imágenes (con Stable Diffusion):\****

\- Una vez que se tiene el prompt generado, este se utiliza para crear una imagen utilizando el modelo de ***\*Stable Diffusion\**** (versión 1.5).

### Requisitos

Para ejecutar este código en **Google Colab**, necesitarás las siguientes librerías:

```python
!pip install transformers diffusers torch Pillow bitsandbytes accelerate
```

**Obtén tu token de Hugging Face**

Para usar modelos de Hugging Face, necesitarás un **HF_TOKEN**. Aquí te mostramos cómo obtenerlo:

- Ve a [Hugging Face](https://huggingface.co/).
- Regístrate o inicia sesión en tu cuenta.
- Dirígete a **Settings** (Configuraciones) > **Access Tokens**.
- Crea un nuevo token con los permisos que necesites (por ejemplo, "read").
- Guarda este token para usarlo en tu código

### Código para Generación de Prompts e Imágenes

A continuación se presenta el código completo para la creación de una imagen a partir de un texto generado por el modelo de lenguaje Phi:

```python
pythonCopiar códigoimport torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from diffusers import StableDiffusionPipeline
from PIL import Image

class ImageGenerationPipeline:
    def __init__(self):
        # Configuración del modelo Phi
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )

        # Cargar el modelo y tokenizer Phi
        self.phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
        self.phi_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3.5-mini-instruct",
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.float16
        )

        # Configuración de Stable Diffusion
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to("cuda")  # Usa "cpu" si no tienes GPU

    def generate_prompt(self, input_text, max_length=100):
        # Generar un prompt mejorado usando el modelo Phi
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
        # Generar la imagen usando Stable Diffusion
        image = self.sd_pipeline(prompt).images[0]
        return image
```

### Ejecución del Pipeline

1. **Generar un prompt mejorado**:

   Usamos el modelo Phi para mejorar el prompt a partir de un texto inicial:  

   "Give me a prompt to create an image of a magic beach."

   ```python
   pythonCopiar códigoinput_text = "Give me a prompt to create an image of a magic beach."
   pipeline = ImageGenerationPipeline()
   enhanced_prompt = pipeline.generate_prompt(input_text)
   print(f"Prompt Generado: {enhanced_prompt}")
   ```

   **Prompt mejorado**:

   ```
   \#  Answer Prompt for Image Generation: Create a captivating digital artwork that 
   depicts a magical beach scene. The beach is bathed in ethereal light, emanating from a
   setting sun that casts a warm golden hue across the scene. The sand sparkles with a touch
   of magic, shimmering with tiny, iridescent crystals that twink
   ```

   

2. **Generar la imagen**:

   Utilizando el prompt mejorado, generamos una imagen con **Stable Diffusion**:

   ```
   pythonCopiar códigoimage = pipeline.generate_image(enhanced_prompt)
   image.show()
   
   # Guardar la imagen en la carpeta de descargas
   image_path = './Descargas/beach.png'
   image.save(image_path)
   print(f"Imagen guardada en: {image_path}")
   ```

### Descargar la Imagen (Google Colab)

Si estás trabajando en **Google Colab**, puedes descargar la imagen generada usando:

```
pythonCopiar códigofrom google.colab import files
files.download(image_path)  # Esto descargará la imagen a tu computadora
```

**Imágenes de referencia**:

**Imágen generada **:

![Beach image](https://i.pinimg.com/1200x/80/27/74/802774462e3cc682e57625718d6e38ad.jpg)






**BONUS**:

[Pose estimation](https://youtu.be/NAwsCyCkQbA?si=DkTBEBpD6Oye8ovc)