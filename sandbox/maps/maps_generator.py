import random
import os

def generar_mapa(filas=23, columnas=43, num_indicaciones=10, num_ubicaciones=5, densidad_paredes=0.1):
    # Crear el mapa vacío con bordes de '#'
    mapa = [[' ' for _ in range(columnas)] for _ in range(filas)]
    
    # Rellenar los bordes con '#'
    for i in range(filas):
        mapa[i][0] = '#'
        mapa[i][columnas - 1] = '#'
    for j in range(columnas):
        mapa[0][j] = '#'
        mapa[filas - 1][j] = '#'
    
    # Función para colocar un símbolo en una posición aleatoria válida
    def colocar_simbolo(simbolo, cantidad):
        colocados = 0
        while colocados < cantidad:
            x = random.randint(1, filas - 2)
            y = random.randint(1, columnas - 2)
            if mapa[x][y] == ' ':
                mapa[x][y] = simbolo
                colocados += 1

    # Colocar paredes internas ('#') según la densidad dada
    num_paredes = int(densidad_paredes * (filas - 2) * (columnas - 2))
    colocar_simbolo('#', num_paredes)
    
    # Colocar indicaciones ('!')
    colocar_simbolo('!', num_indicaciones)
    
    # Colocar ubicaciones ('A')
    colocar_simbolo('A', num_ubicaciones)
    
    # Convertir el mapa a un string para guardar
    mapa_str = "\n".join("".join(fila) for fila in mapa)
    return mapa_str

def generar_varios_mapas(cantidad=100, filas=23, columnas=43, num_indicaciones=10, num_ubicaciones=5, densidad_paredes=0.1):
    # Crear un directorio para almacenar los mapas generados
    os.makedirs("mapas_generados", exist_ok=True)
    
    for i in range(20, cantidad + 31):
        mapa = generar_mapa(filas, columnas, num_indicaciones, num_ubicaciones, densidad_paredes)
        nombre_archivo = f"mapa_{i}.txt"
        with open(nombre_archivo, "w") as f:
            f.write(mapa)
        print(f"Mapa {i} generado y guardado en {nombre_archivo}")

# Parámetros para la generación de los mapas
filas = 23
columnas = 43
num_indicaciones = 10
num_ubicaciones = 7
densidad_paredes = 0.20

# Generar 100 mapas
generar_varios_mapas(10, filas, columnas, num_indicaciones, num_ubicaciones, densidad_paredes)
