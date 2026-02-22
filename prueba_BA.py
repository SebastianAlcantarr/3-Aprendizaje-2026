import utileria as ut
import arboles_numericos as an
import bosque_aleatorio as ba
import random
"""
Pruebas de Bosques Aleatorios comparando 

1)Arbol Numerico
2)Bosques Aleatorios con distintos numeros de subconjuntos generados
3)Bosques Aleatorios con distintas numeros de variables en cada nodo
4)Bosques Aleatorios con distintas profundidades del bosque
"""

url = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"

archivo, archivo_datos = "datos/cancer.zip", "datos/wdbc.data"


atributos_nombres = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
datos = ut.lee_csv(archivo_datos, atributos=atributos_nombres)

for d in datos:
    d['Diagnosis'] = 1 if d['Diagnosis'] == 'M' else 0
    for i in range(1, 31): d[f'feature_{i}'] = float(d[f'feature_{i}'])
    del(d['ID'])



objetivo = 'Diagnosis'
random.seed(42)
random.shuffle(datos)
N = int(0.8 * len(datos))
entrenamiento, val = datos[:N], datos[N:]


print('Prueba de Bosques Aleatorios')
arbol = an.entrena_arbol(entrenamiento, objetivo, clase_default=0, max_profundidad=5)
precision_arbol = an.evalua_arbol(arbol, val, objetivo)
print(f"Precision con un solo arbol [{precision_arbol}]")


print('\nPrecision de bosques aleatorios con distintos numeros de subconjuntos generados\n')
for m in [5, 10, 25, 50]:
    bosque = ba.entrena_bosque(
        entrenamiento, objetivo, clase_default=0,
        M=m,
        variables_por_nodo=5,
        max_profundidad=5
    )
    precision_bosque = ba.evalaur_bosques(bosque, val, objetivo)
    print(f"Precision de bosque con [{m}] subconjutnos generados = {precision_bosque}")


print("\nPruebas cambiando el numero de variables en cada nodo")
for var in [2, 5, 10, 20]:
    bosque = ba.entrena_bosque(
        entrenamiento, objetivo, clase_default=0,
        M=50, variables_por_nodo=var, max_profundidad=5
    )
    prec = ba.evalaur_bosques(bosque, val, objetivo)
    print(f"Precision del bosque con {var} Variables: {prec}")


print("\nPruebas cambiando la profundidad del bosque")
for prof in [2, 5, 10, 25]:
    bosque = ba.entrena_bosque(
        entrenamiento, objetivo, clase_default=0,
        M=50, variables_por_nodo=5, max_profundidad=prof
    )
    precision = ba.evalaur_bosques(bosque, val, objetivo)
    print(f"Precision del bosque con una Profundidad de [{prof}] {precision:}")
