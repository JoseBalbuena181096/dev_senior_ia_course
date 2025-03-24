# Crear una variable
numero = 10.2

# mostrar por consola el valor de la variable
print("El número es ", numero)

# saber el tipo de dato 
print(type(numero))

## Tipos de datos

#   entero
entero = 10
#   flotante
flotante = 10.2

#   cadena de texto
string = "10.2"

#   booleano
booleano = False

## Operadores aritmeticos

numero1 = 10
numero2 = 20

# sumar 
sumar = numero1 + numero2
print(sumar)

# restar
restar = numero1- numero2
print(sumar)

# multiplicación
multiplicacion = numero1 * numero2
print(multiplicacion)

# división
division = numero1 / numero2
print(division)

# exponente
exponente = numero1 ** numero2
print(exponente)

# Operadores logicos

a = True
b = False

# operador and
print(a and b)

#operador  or
print(a or  b)

#operador not
print(not a)


# Estructuras de control y operadores de comparación

cedula = 1000

# evaluar la condición de comparación 
if cedula == 1200:
    # se ejecuta esto si es verdadero
    print("Tu cedula es correcta")

# se ejecuta esto si no es verdadero
else:
    print("Cedula incorrecta")


# "clear terminar jijiji"
print("\n"*100)

# solicitud por consola

cedula = int(input('Ingrese la cedula del usuario: \n'))

if cedula == 1200:
    print("Tu cedula es correcta")
else:
    print("Cedula incorrecta")

print("\n"*100)

# ultilizando operadores de comparación y lógicos con las estructuras de control 

cedula = int(input('Ingrese la cedula del usuario: \n'))
edad = int(input('Ingrese la edad del usuario:\n'))

if edad >= 18 and cedula == 1200:
    print("Cedula y edad correctos ingrese")
else:
    print("Cedula o edad incorrecta intente de nuevo")


# Collecciones en python

# Listas

# Declaración de la lista
numeros = [1,2,3,4,5]
# mostrando el elemento con valor 2 en la posición 1 de la lista 
print(numeros[1])

# modificamos el elemento con valor 2 a un nuevo valor 20
numeros[1] = 20

numero = []