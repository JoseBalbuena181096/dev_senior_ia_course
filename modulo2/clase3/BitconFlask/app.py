from flask import Flask, jsonify, request, render_template
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from matplotlib.figure import Figure

app = Flask(__name__)

def clean_data(df):
    original_shape = df.shape[0]
    
    # Remove rows with any empty values
    df = df.dropna()
    
    # Remove rows where Volume is negative or not numeric
    df = df[pd.to_numeric(df['Volume'], errors='coerce').notna()]
    df = df[df['Volume'] >= 0]
    
    # Remove rows where prices are not numeric or invalid
    numeric_columns = ['Open', 'High', 'Low', 'Close']
    for col in numeric_columns:
        df = df[pd.to_numeric(df[col], errors='coerce').notna()]
    
    # Remove rows where Close price is 0 or negative
    df = df[df['Close'] > 0]
    
    # Ensure High is greater than Low
    df = df[df['High'] >= df['Low']]
    
    # Ensure High is greater than or equal to Close and Open
    df = df[df['High'] >= df['Close']]
    df = df[df['High'] >= df['Open']]
    
    # Ensure Low is less than or equal to Close and Open
    df = df[df['Low'] <= df['Close']]
    df = df[df['Low'] <= df['Open']]
    
    # Remove rows with invalid timestamps
    df = df[pd.to_datetime(df['datetime'], errors='coerce').notna()]
    
    clean_shape = df.shape[0]
    removed_rows = original_shape - clean_shape
    
    cleaning_stats = {
        "Filas originales": original_shape,
        "Filas después de limpieza": clean_shape,
        "Filas eliminadas": removed_rows,
        "Porcentaje eliminado": f"{(removed_rows/original_shape)*100:.2f}% de datos"
    }
    
    return df, cleaning_stats

def analyze_data(df):
    # Calcular estadísticas básicas
    stats = {
        "Estadísticas generales": {
            "Período de tiempo": f"{df['datetime'].min()} a {df['datetime'].max()}",
            "Días analizados": (df['datetime'].max() - df['datetime'].min()).days,
            "Número de registros": len(df)
        },
        "Precio": {
            "Precio mínimo": f"${df['Low'].min():.2f}",
            "Precio máximo": f"${df['High'].max():.2f}",
            "Precio promedio": f"${df['Close'].mean():.2f}",
            "Precio de cierre inicial": f"${df['Close'].iloc[0]:.2f}",
            "Precio de cierre final": f"${df['Close'].iloc[-1]:.2f}",
            "Cambio total": f"{((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100):.2f}%"
        },
        "Volumen": {
            "Volumen total": f"{df['Volume'].sum():.2f}",
            "Volumen promedio diario": f"{df['Volume'].mean():.2f}",
            "Día con mayor volumen": str(df.loc[df['Volume'].idxmax(), 'datetime'].date())
        }
    }
    
    # Crear gráficos para análisis
    fig = Figure(figsize=(14, 16))
    
    # Calcular retornos diarios
    df['Returns'] = df['Close'].pct_change() * 100
    
    # Gráfico 1: Movimiento de precios
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(df['datetime'], df['Close'], color='blue')
    ax1.set_title('Movimiento de Precio')
    ax1.set_ylabel('Precio USD')
    ax1.grid(True)
    
    # Gráfico 2: Volumen
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.bar(df['datetime'], df['Volume'], color='green', alpha=0.7)
    ax2.set_title('Volumen a lo largo del tiempo')
    ax2.set_ylabel('Volumen')
    ax2.grid(True)
    
    # Gráfico 3: Distribución de precios
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.hist(df['Close'], bins=50, color='purple', alpha=0.7)
    ax3.set_title('Distribución de Precios')
    ax3.set_xlabel('Precio')
    ax3.set_ylabel('Frecuencia')
    ax3.grid(True)
    
    # Gráfico 4: Distribución de componentes de precio (boxplot)
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.boxplot([df['Open'], df['High'], df['Low'], df['Close']])
    ax4.set_title('Distribución de Componentes de Precio')
    ax4.set_xticklabels(['Open', 'High', 'Low', 'Close'])
    ax4.grid(True)
    
    # Gráfico 5: Candlestick chart
    ax5 = fig.add_subplot(3, 2, 5)
    
    # Simplificamos para mostrar solo algunos puntos si hay demasiados datos
    if len(df) > 30:
        sample_size = 30
        sample_indices = np.linspace(0, len(df)-1, sample_size, dtype=int)
        df_sample = df.iloc[sample_indices]
    else:
        df_sample = df
    
    # Datos para candlestick
    width = 0.6
    width2 = width * 0.8
    
    # Para cada punto, creamos un candlestick
    for i, (idx, row) in enumerate(df_sample.iterrows()):
        # Cuerpo del candlestick
        if row['Close'] >= row['Open']:
            color = 'green'  # Día positivo
        else:
            color = 'red'    # Día negativo
            
        # Dibujamos el cuerpo (rect) entre open y close
        rect = plt.Rectangle((i - width/2, min(row['Open'], row['Close'])), 
                             width, abs(row['Close'] - row['Open']),
                             facecolor=color, alpha=0.5)
        ax5.add_patch(rect)
        
        # Dibujamos las mechas (líneas) para High y Low
        ax5.plot([i, i], [row['Low'], min(row['Open'], row['Close'])], color='black', linewidth=1)
        ax5.plot([i, i], [max(row['Open'], row['Close']), row['High']], color='black', linewidth=1)
    
    # Configuración del gráfico
    ax5.set_title('Candlestick Chart (muestra)')
    ax5.set_xlabel('Índice de tiempo')
    ax5.set_ylabel('Precio')
    ax5.grid(True)
    ax5.set_xlim(-1, len(df_sample))
    
    # Gráfico 6: Retornos diarios y volatilidad
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.plot(df['datetime'], df['Returns'].rolling(window=7).std(), color='red', linewidth=2)
    ax6.set_title('Volatilidad (Desviación estándar de retornos, ventana 7 días)')
    ax6.set_ylabel('Volatilidad (%)')
    ax6.grid(True)
    
    # Ajustar el layout
    fig.tight_layout()
    
    # Convertir el gráfico a una imagen codificada en base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    plot_url = f"data:image/png;base64,{img_data}"
    
    # Estadísticas adicionales
    stats["Análisis avanzado"] = {
        "Volatilidad (desv. estándar de retornos)": f"{df['Returns'].std():.2f}%",
        "Retorno diario promedio": f"{df['Returns'].mean():.2f}%",
        "Volatilidad anualizada": f"{df['Returns'].std() * np.sqrt(365):.2f}%",
        "Sharpe Ratio (asumiendo tasa libre de riesgo 0)": f"{df['Returns'].mean() / df['Returns'].std():.2f}"
    }
    
    return stats, plot_url

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        try:
            # Verificar si el archivo está en la solicitud
            if 'file' not in request.files:
                return render_template("input_form.html", error="No se ha subido ningún archivo")
            
            file = request.files['file']
            
            # Si el usuario no selecciona un archivo
            if file.filename == '':
                return render_template("input_form.html", error="No se ha seleccionado ningún archivo")
            
            # Verificar que sea un archivo CSV
            if not file.filename.endswith('.csv'):
                return render_template("input_form.html", error="Por favor, sube un archivo CSV válido")
            
            # Suprimir advertencias de tipos mixtos
            warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
            
            # Leer el archivo CSV con tipos específicos
            try:
                df = pd.read_csv(file, low_memory=False, dtype={
                    'Open': float,
                    'High': float,
                    'Low': float,
                    'Close': float,
                    'Volume': float,
                    'Timestamp': float
                })
                
                # Convertir timestamp a datetime
                df['datetime'] = pd.to_datetime(df['datetime'])
                
            except Exception as e:
                # Si falla con tipos específicos, intentar lectura genérica
                file.seek(0)
                df = pd.read_csv(file, low_memory=False)
                
                # Comprobar si tiene las columnas esperadas
                expected_columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'datetime']
                if not all(col in df.columns for col in expected_columns):
                    return render_template("input_form.html", error="El CSV no tiene el formato esperado. Se esperan las columnas: Timestamp, Open, High, Low, Close, Volume, datetime")
                
                # Convertir columnas a tipos adecuados
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            
            # Limpiar los datos
            df_clean, cleaning_stats = clean_data(df)
            
            # Verificar si quedaron suficientes datos después de la limpieza
            if len(df_clean) < 10:
                return render_template("input_form.html", error="No hay suficientes datos válidos después de la limpieza. Verifica el formato del CSV.")
            
            # Analizar los datos
            stats, plot_url = analyze_data(df_clean)
            
            # Añadir estadísticas de limpieza
            stats["Limpieza de datos"] = cleaning_stats
            
            # Mostrar solo los primeros 10 elementos para la vista previa
            df_display = df_clean.head(10).copy()
            
            # Formatear columnas para mejorar visualización
            df_display['Open'] = df_display['Open'].map('${:.2f}'.format)
            df_display['High'] = df_display['High'].map('${:.2f}'.format)
            df_display['Low'] = df_display['Low'].map('${:.2f}'.format)
            df_display['Close'] = df_display['Close'].map('${:.2f}'.format)
            
            return render_template(
                "input_form.html", 
                data=df_display.to_html(classes="w-full table-auto text-sm", index=False, border=0),
                stats=stats,
                plot_url=plot_url
            )
            
        except Exception as e:
            error_message = f"Error al procesar los datos: {str(e)}"
            return render_template("input_form.html", error=error_message)
    return render_template("input_form.html")


if __name__ == "__main__":
    app.run(debug=True)