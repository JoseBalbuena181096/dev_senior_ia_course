from flask import Flask, jsonify, request, render_template
import pandas as pd

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        try:
            # Obtener datos del formulario Timestamp,Open,High,Low,Close,Volume,datetime
            timestamp = request.form["timestamp"]
            open_price = request.form["open_price"]
            high_price = request.form["high_price"]
            low_price = request.form["low_price"]
            close_price = request.form["close_price"]
            volume = request.form["volume"]
            datetime = request.form["datetime"]
            volume_currency = request.form["volume_currency"]
            weighted_price = request.form["weighted_price"]

            #  convertir el timestap a fecha legible    
            date_time = datetime.datetime.fromtimestamp(timestamp)

            # crear un dataframe con los datos
            data = {
                'timestamp': [timestamp],
                'Data': [date_time.strftime('%Y-%m-%d %H:%M:%S')],
                'Open': [open_price],
                'High': [high_price],
                'Low': [low_price],
                'Close': [close_price],
                'Volume_(BTC)': [volume],
                'Volumen_(Currency)': [volume_currency],
                'Weighted_Price': [weighted_price],
            }

            # Crear un dataframe con los datos
            df = pd.DataFrame(data)

            # Calculo estadisticas simples
            stats = {
                "Precio Promedio": df[['Open', 'High', 'Low', 'Close']].mean(axis=1).iloc[0],
                'Precio Maximo': df['High'].max(),
                'Precio Minimo': df['Low'].min(),
                'Precio de Cierre': df['Close'].iloc[0],
            }
            return render_template("input_form.html", stats=stats)
            
        except Exception as e:
            error_message = f"Error al procesar los datos: {str(e)}"
            return render_template("input_form.html", error=error_message)
    return render_template("input_form.html")


if __name__ == "__main__":
    app.run(debug=True)
