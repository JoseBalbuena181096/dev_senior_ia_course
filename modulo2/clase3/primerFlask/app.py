from flask import Flask, jsonify, request

app = Flask(__name__)

# Route for home
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Hello World"})

# Ruta enviar dato(post)
@app.route("/message", methods=["POST"])
def enviar_mensaje():
    data = request.json
    message = data.get("message")
    return jsonify({"response": "Data received", "data": data}),201


# Iniciar la aplicacion
if __name__ == "__main__":
    app.run(debug=True)
