# Introducción a la programación

-   ¿Qué es la programación?
    Escribir instrucciones para la computadora
    Importancia:
    -   Variables, tipos de datos y operadores.


-   ¿Qué son las variables?
    Espacios de memoria para almacenar valores, Almacenar datos que puedan cambiar.
    Tienen: 
    -   Tipo de dato (Numeros, cadenas de texto, booleano, ect.)
    -   Nombre  (La referencia de la variable como la utilizamos por su nombre)
    -   Dato    (El valor que almacena la variable)

-   Tipos de datos primitivos (los más basicos que podemos encontrar en un lenguaje de programación):
    -   Entero(int):
        Números sin decimal 10
    -   Punto flotante(float):
        Número con parte decimal 10.23
    -   Cadenas de texto(string):
        Son cadenas de caracteres "10.23" los caracteres
    -   Booleanos (bool):
        Puede tomar dos valores True 1, False 0

-   Los operadores, son para realizar operaciones con los datos (operaciones aritmeticas, logicas, de comparación, etc). 

-   Operadores aritmeticos:
        -   Suma (+) x + y
        -   Resta (-) x - y
        -   Multiplicación (*)  x * y
        -   Divición (/) x / y
        -   Potencia (**) x ** y
        -   Modulo (%) x % y

-   Operadores de comparación:
        -   == IGUAL que
        -   > Mayor que
        -   < Menor que
        -   >=  Mayor o igual que
        -   <= Menor o igual que

-   Operadores lógicos:
        -   and operador  -> ambos verdaderos para ser verdadero
        -   or operador  -> con uno que se verdadero el resultado de la expreción sera verdadera
        -   not operador de negación -> invierte el estado lógico si es verdadero lo vuelve falso
        -   xor operador -> ambos deben ser diferentes valores lógicos para que el resultado de la expreción sea verdadero

-   Operadores de asignación:
        -   = operador de asignación -> x = y  y es asignado a x, lo del lado derecho se asigna a lo del lado izquierdo
        -   =+  operacion de asignación con incremento  -> a lo que se asigna del lado izquierdo se le incrementa lo del lado derecho
        -   =-  operacion de asignación con decremento  -> a lo que se asigna del lado izquierdo se le decrementa lo del lado derecho  


## Colecciones en python

-   Listas:
    Es una estructura de datos que permite almacenar múlples elementos ordenados en una sola variable. La listas con muy versátiles y tiene las siguientes caracteristicas:
    -   Se declaran usado corchetes []
    -   Contienen diferenres tipos de datos (enteros, cademas, etc).
    -   Son mutables (se puede modificar despues de su creación).
    -   Mantiene un orden especifíco de elemtos
    -   Permite elementos duplicados
    -   Se accede a un elemtos de la lista por nombre de la lista y la posicion del elemento empezando desde el 

    Ejemplo de una lista de números numeros = [1,2,3,4,5]
    si quiero acceder al elemento con valor 2, lo ralizo a numeros[1] -> 2

    Principales funciones para trabajar con listas en python:
    -   1.Crear listas:
        -   lista = [] -> Lista vacia
        -   lista = [1,2,3] -> lista con elementos
        -   lista = list()  -> Crear lista usando constructor
        -   lista = list("abc") -> Convertir iterable a lista

    -   2. Añadir elementos
        -   append(x) -> Añade un elemento al final lista.append(1) -> el último de la lista sera 1
        -   insert(i, x) -> Inserta un elemento en la posición especifica -> lista.insert(2, 'a') inserta en 'a' en la posición 2
        -   extend(iterable) -> Añade todos los elementos a un iterable  -> lista.extend([1,2,3]) añade a la lista los elementos 1,2,3
    -   3. Eliminar elemenetos:
        -   remove(x) -> Elimina la primera aparición del valor -> lista.remove(1) -> Elimina la primera aparición del elemento con valor 1
        -   pop(i) ->  lista.pop(i) -> Elimina el elemento en la posición i lo retorna, por defecto lista.pop() elimina el último elemento y lo retorno
        -   clear() -> lista.clear() -> Elimana todos los elemnto de la lista 
        -   del lista[i]  ->  Elimina el elemento en la posición i de la lista i
    -   4.  Buscar elementos 
        -   index(x)  ->  lista.index(x)  ->  Devuelve el indice de la primera aparición del elemento x
        -   count(x)  ->  lista.count(x)  ->  Cuenta las apariciones del valor x
        -   in  ->  3 in lista  ->  Comprueba  si un elemento está en la lista, devuelve True si esta en la lista

    -   5. Ordenar y reorganizar
        -   sort()   ->   lista.sort()   ->   Ordena la lista en su lugar  
        -   sorted(lista)  ->  Retorna una lista ordenada
        -   reverse()   ->  lista.reverse()  -> Revierte el orden de la lista
        -   lista[::-1]  ->  Revierte el orden de la lista
    -   6.  Copiar listas
        -   copy()  ->  lista.copy()  ->  Crea una copia superficial
        -   lista[:]  ->  Crea una copia mediante slicing
    -   7.  Operaciónes comunes
        -   len(lista)  ->  Obtiene el número de elementos
        -   max(lista)  ->  Encuentra el valor máximo
        -   min(lista)  ->  Encuentra el valor minímo
        -   sum(lista)  ->  Suma todos los elementos numéricos

    -   8.  Slicing (rebanadas)
        -   lista[start:end]  ->  Obtiene una sublista que inicia en el indice start y termina un indice antes de end donde end no esta incluido
        -   lista[start:end:step]  ->  Obtiene una sublista con un paso específico, donde el step indica el salto en el indice 
    -   9. Compresión de listas
        -   [expresion for item in lista]  ->  Crea una lista aplicando una expresión
        -   [expresion for iten in lista if condicion]   ->  Con filtrado dada una condición

    -   10. Funciones útiles para trabajo con listas
        -   map(funcion, lista)   ->  Aplica una función a cada elemento
        -   filter(funcion, lista)  ->  Filtra elementos según una función
        -   zip(lista1, lista2)   ->  Combina elementos de varias listas
        -   enumerate(lista)   ->   Devuelve pares(índice, valor) 

-   Las tuplas:
    Una tupla es una estructura de datos similar a una lista pero con la diferencia fundamental: que las tuplas son inmutables, lo que significa que no se puede modificar despues de su creación.

    Caracteristicas principales:
    -   Se declara usando paréntesis ()
    -   Puede contener valores de diferentes tipos(enteros, cadenas, etc).
    -   Son inmutables (no se pueden modificar una vez creadas).
    -   Mantiene un orden especifíco de elementos.
    -   Permiten elementos duplicados.

    Ejemplo de una tupla de números numeros = (1,2,3)
    si quiero acceder al elemento con valor 3, lo ralizo a numeros[2] -> 3
    si quiero modificar elemento en la posición 3 numeros[2] = 10 -> marca erro por que las tuplas no se pueden modificar. 

    Tiene las mismas funciones y métodos que las listas, excepto por las que añaden o eliminan elementos de la tupla