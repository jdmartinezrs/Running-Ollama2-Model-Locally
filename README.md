# Ejecución del Modelo Llama2 en un Servidor Local

Este proyecto realiza una solicitud HTTP `POST` a un servidor local en la URL `http://localhost:11434/api/chat`, enviando un payload que incluye un modelo llamado "llama2" y un mensaje con el contenido "who invented hiphop music?". Luego procesa la respuesta del servidor, que probablemente contiene una serie de fragmentos JSON, y los une en un único mensaje que se imprime al final.

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