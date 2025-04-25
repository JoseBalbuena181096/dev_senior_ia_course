import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

movies = {
    '1': {
        'title': 'The Dark Knight',
        'year': 2008,
        'rating': 9.0,
        'duration': 152,
    },
    '2': {
        'title': 'Pulp Fiction',
        'year': 1994,
        'rating': 8.9,
        'duration': 154,
    },
    '3': {
        'title': 'El Padrino',
        'year': 1972,
        'rating': 9.2,
        'duration': 175,
    },
    '4': {
        'title': 'Forrest Gump',
        'year': 1994,
        'rating': 8.8,
        'duration': 142,
    },
    '5': {
        'title': 'Matrix',
        'year': 1999,
        'rating': 8.7,
        'duration': 136,
    },
    '6': {
        'title': 'El Señor de los Anillos',
        'year': 2001,
        'rating': 8.8,
        'duration': 178,
    },
    '7': {
        'title': 'Titanic',
        'year': 1997,
        'rating': 7.9,
        'duration': 194,
    },
    '8': {
        'title': 'Inception',
        'year': 2010,
        'rating': 8.8,
        'duration': 148,
    },
    '9': {
        'title': 'El Rey León',
        'year': 1994,
        'rating': 8.5,
        'duration': 88,
    },
    '10': {
        'title': 'Gladiador',
        'year': 2000,
        'rating': 8.5,
        'duration': 155,
    },
    '11': {
        'title': 'Avatar',
        'year': 2009,
        'rating': 7.8,
        'duration': 162,
    },
    '12': {
        'title': 'Jurassic Park',
        'year': 1993,
        'rating': 8.1,
        'duration': 127,
    },
    '13': {
        'title': 'El Silencio de los Corderos',
        'year': 1991,
        'rating': 8.6,
        'duration': 118,
    },
    '14': {
        'title': 'Interestelar',
        'year': 2014,
        'rating': 8.6,
        'duration': 169,
    },
    '15': {
        'title': 'Los Vengadores',
        'year': 2012,
        'rating': 8.0,
        'duration': 143,
    },
    '16': {
        'title': 'El Club de la Pelea',
        'year': 1999,
        'rating': 8.8,
        'duration': 139,
    },
    '17': {
        'title': 'Regreso al Futuro',
        'year': 1985,
        'rating': 8.5,
        'duration': 116,
    },
    '18': {
        'title': 'El Pianista',
        'year': 2002,
        'rating': 8.5,
        'duration': 150,
    },
    '19': {
        'title': 'La Lista de Schindler',
        'year': 1993,
        'rating': 8.9,
        'duration': 195,
    },
    '20': {
        'title': 'El Laberinto del Fauno',
        'year': 2006,
        'rating': 8.2,
        'duration': 118,
    },
    '21': {
        'title': 'La Cenicienta',
        'year': 2015,
        'rating': 7.4,
        'duration': 105,
    },
}

def create_dataframe(movies):
    """
    Un dataframe es una tabla de datos que se puede manipular con pandas
    En este caso, se crea un dataframe a partir de un diccionario de peliculas
    """
    df = pd.DataFrame.from_dict(movies, orient='index')
    return df

def mostrar_dataframe(df):
    """
    Muestra el dataframe en una tabla
    """
    print(df)

def mostrar_info(df):
    print('Primeras 10 peliculas:')
    print(df.head(10))

def mostrar_estadisticas(df):
    print('Estadisticas del dataframe:')
    print(df.describe())

def mostrar_informacion(df):
    print('Información del dataframe')
    print(df.info())

def graficar_rating(df):
    plt.figure(figsize=(10, 6))
    plt.title('Distribución de clasificaciones')
    plt.xlabel('Calificación')
    plt.ylabel('Frecuencia')
    plt.hist(df['rating'], bins=5, color='lightblue', edgecolor='black')
    plt.show()

def graficar_rating3D(df):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    hist, bins = np.histogram(df['rating'], bins=5)
    xs = (bins[:-1] + bins[1:])/2
    
    ax.bar3d(xs, np.zeros_like(xs), np.zeros_like(xs), 
             0.2, 0.2, hist,
             color='lightblue', edgecolor='black')
    
    ax.set_title('Distribución de clasificaciones')
    ax.set_xlabel('Calificación')
    ax.set_ylabel('Eje Y')
    ax.set_zlabel('Frecuencia')
    
    plt.show()

df = create_dataframe(movies)
# mostrar_dataframe(df)
# mostrar_info(df)
# mostrar_estadisticas(df)
# mostrar_informacion(df)
graficar_rating3D(df)