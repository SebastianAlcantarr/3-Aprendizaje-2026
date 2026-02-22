import random
from collections import Counter
import arboles_numericos as an

def entrena_bosque(datos, objetivo, clase_default, M=None, variables_por_nodo=None, **kwargs):
    bosque = []
    for _ in range(M):
        subconjuntos = random.choices(datos, k=len(datos))
        arbol = an.entrena_arbol(
            subconjuntos,
            objetivo,
            clase_default,
            variables_seleccionadas=variables_por_nodo,
            **kwargs
        )
        bosque.append(arbol)
    return bosque

def predice_instancia_bosque(bosque, instancia):
    predicciones=[]
    for arbol in bosque:
        pre=arbol.predice(instancia)
        predicciones.append(pre)

    conteo = Counter(predicciones)
    return conteo.most_common(1)[0][0]


def predice_bosque(bosque, datos):
    prediccion=[]
    for d in datos:
        p=predice_instancia_bosque(bosque,d)
        prediccion.append(p)
    return prediccion


def evalaur_bosques(bosque,datos,objetivo):
    predicciones = predice_bosque(bosque, datos)
    aciertos = sum(1 for p, d in zip(predicciones, datos) if p == d[objetivo])
    return aciertos / len(datos)