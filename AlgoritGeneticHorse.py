from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import numpy as npy

'''Variables globales'''
rateMutation = 0.05
numGenerations = 200
population = 100
lenTable = 8
tamTable = 8*8
probabilityCruze = 0.7
minHorse = 5
maxHorse = 20
'''Metodo para obtener las casillas atacadas'''
def getBoxesAtack(tab):
#Creamos una variable de tipo lista donde vamos a almacenar 8 tuplas que equivalen a los ataques del caballo
  posMovHorse = [(-2,1),(-2, -1), (1,2),(1,-2),(-1, 2),(2,1), (2,-1), (-1, -2)]
#Creamos una variable para el numero de casillas atacadas
  numCasillasAtacadas = 0
#Creamos una variable de tipo lista donde va a contener las casillas atacadas ya contadas
  casAtaCons = []
  #Realizamos un ciclo que recorra el tablero que recibe como parametro el metodo gerBoxesAtack
  for posHorse in tab:
  #Realizamos un ciclo donde el desplazamiento en filas va a ser la posicion 0 en la tupla de posMovHorse
    for (desRow, despColumns) in posMovHorse:
      #Se realiza el mapeo de las coordenadas
      rowHorse    = posHorse  // lenTable
      columnHorse = posHorse  %  lenTable
      rowAta    = rowHorse      +  desRow
      columnsAta = columnHorse   +  despColumns
      #creamos una variable que sea la suma de las columnas atacadas por la multiplicacion de las filas atacadas por la longitud de la tabla
      casAtac = columnsAta + rowAta*lenTable

      if casDenTab(rowAta, columnsAta) and casAtac not in casAtaCons:
        numCasillasAtacadas += 1
        casAtaCons.append(casAtac)

  return numCasillasAtacadas
'''funcion para verificar las casillas dentro del tablero'''
def casDenTab(f, c):
    return f >= 0 and f <= lenTable - 1 and c >=0 and c <= lenTable - 1
'''Funcion para obtener las posiciones iniciales del caballo'''
def getPosicionesInicialesCaballos():

  cantidadCaballos = random.randint(minHorse, maxHorse)

  return random.sample(range(tamTable), cantidadCaballos)
'''Funcion para obtener el mejor individuo adaptabilidad'''
def getFitness(tablero):
  casillasAtacadas = getBoxesAtack(tablero)
  return tamTable - casillasAtacadas,

#Funcion para ejecutar el algoritmo genetico
def executeAlgGenetic():
  poblacion    = toolbox.poblacion(n = population)
  stats        = tools.Statistics(lambda tablero: tablero.fitness.values)
  bestTable = tools.HallOfFame(1)

#Metodos de la libreria DEAP de la importacion stats donde podemos sacar el promedio,desviacion estandar,minimo y maximo
  stats.register("Promedio", npy.mean)
  stats.register("DesEstandar", npy.std)
  stats.register("Minimo", npy.min)
  stats.register("Maximo", npy.max)

#Metdodo de la libreria DEAp utilizando la importacion algorithms usando el metodo esSimple
  algorithms.eaSimple(poblacion, toolbox, probabilityCruze, rateMutation, numGenerations, stats=stats, halloffame = bestTable)
  return bestTable

#Metodos de la libreria DEAP para crear el tablero y obtener la adaptabilidad minima
creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
creator.create("Tablero", list, fitness = creator.FitnessMin)
#Creamos la caja de herramientas para crear el tablero y la poblacion
toolbox = base.Toolbox()
#Iniciamos los metodos register donde creamos el tablero y la poblacion
toolbox.register("tablero", tools.initIterate, creator.Tablero, getPosicionesInicialesCaballos)
toolbox.register("poblacion", tools.initRepeat, list, toolbox.tablero)
#Funcion de la caja de herramientas para hacer la seleccion de los individuos
toolbox.register("select", tools.selTournament, tournsize = 3)
toolbox.register("mate", tools.cxTwoPoint)
#Funcion de la caja de herramientas para hacer la mutacion de los individuos
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
#Funcion de la caja de herramientas para hacer la evaluacion don el getFitness
toolbox.register("evaluate", getFitness)


#Funcion para habilitar el codigo genetico verifica que no esta vacio
def Iniciar():
    bestTable = executeAlgGenetic()
    print(bestTable)

#verifica que la poblacion no esta vacia
Iniciar()
