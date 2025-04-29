### Que es Flask
Flask es un framework web para Python que permite crear aplicaciones web.

### Instalacion
`pip install flask`

### Ejemplo de una aplicacion web

```python   
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Hello World"})

@app.route("/message", methods=["POST"])
def enviar_mensaje():
    data = request.json
    message = data.get("message")
    return jsonify({"message": "Data received", "data": data})

if __name__ == "__main__":
    app.run(debug=True)
```

### Rutas
    - GET: Obtiene datos
    - POST: Envía datos
    - PUT: Actualiza datos
    - DELETE: Elimina datos

### Parametros
    - Query parameters: Parametros que se envian en la url
    - Path parameters: Parametros que se envian en la url
    - Body parameters: Parametros que se envian en el body

### Respuestas HTTP
    - 200: OK
    - 201: Created
    - 204: No content
    - 400: Bad request
    - 401: Unauthorized
    - 403: Forbidden
    - 404: Not found
    - 500: Internal server error    
