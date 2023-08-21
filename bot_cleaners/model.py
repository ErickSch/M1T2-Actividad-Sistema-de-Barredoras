from mesa.model import Model
from mesa.agent import Agent
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector

import matplotlib.pyplot as plt
import matplotlib as mlp

import numpy as np
import math
import time
import datetime


class Celda(Agent):
    def __init__(self, unique_id, model, suciedad: bool = False):
        super().__init__(unique_id, model)
        self.sucia = suciedad

class Mueble(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

# Clase de agente estación de carga
class EstacionDeCarga(Agent):
    posiciones = []
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.ocupada = False
        if len(EstacionDeCarga.posiciones) <= unique_id:
            EstacionDeCarga.posiciones.append(self.pos)

            
        # if isinstance(self.enCarga, RobotLimpieza):
        #     self.enCarga.carga = 100
        #     self.ocupada = False

class RobotLimpieza(Agent):

    funcionando = []
    celdas_limpias = []
    busca_contrato = -1
    contratos = []
    contratado = -1

    def __init__(self, unique_id, model, mueblesPos, recorrido):
        super().__init__(unique_id, model)
        self.prendido = True
        self.mueblesPos = mueblesPos
        self.sig_pos = None
        self.movimientos = list()
        self.movimientos_recorrido = list()
        self.carga = 100
        self.recorrido = recorrido
        self.estacion_de_carga = None
        RobotLimpieza.funcionando.append(self.unique_id)


    
    def get_agente_from_pos(self, pos):
        agente = self.model.grid.get_cell_list_contents([pos])[0]
        return agente

    def limpiar_una_celda(self, lista_de_celdas_sucias):
        
        celda_a_limpiar = self.random.choice(lista_de_celdas_sucias)
        celda_a_limpiar.sucia = False
        self.sig_pos = celda_a_limpiar.pos
        RobotLimpieza.celdas_limpias.append(celda_a_limpiar.pos)

    def cargar(self):

        if self.estacion_de_carga == None:
            distancias_carga = []
            posiciones = Habitacion.pos_estaciones_carga(self.model)
            for i in posiciones:
                estacion_carga = self.model.grid.get_cell_list_contents([i])[0]
                if estacion_carga.ocupada == True:
                    distancias_carga.append(1000)
                else:
                    distancias_carga.append(self.get_distance(self.pos, i))
            if len(distancias_carga) == 0:
                return 0
            else:
                min_index = distancias_carga.index(min(distancias_carga))
                self.estacion_de_carga = posiciones[min_index]
        elif self.pos == self.estacion_de_carga:
            estacion_carga = self.model.grid.get_cell_list_contents([self.estacion_de_carga])[0]
            estacion_carga.ocupada = True
            self.carga += 25
            if self.carga < 100:
                self.sig_pos = self.pos
            if self.carga > 100:
                self.carga = 100
                self.estacion_de_carga = None
                Habitacion.agregar_recarga(self.model)
        elif self.pos == self.estacion_de_carga and self.carga > 90:
            self.estacion_de_carga = None
            estacion_carga = self.model.grid.get_cell_list_contents([self.estacion_de_carga])[0]
            estacion_carga.ocupada = False
        elif self.estacion_de_carga:
            lista_de_vecinos = self.model.grid.get_neighbors(
                self.pos, moore=True, include_center=False)
            self.dirigirse(self.estacion_de_carga, lista_de_vecinos)

    def dirigirse(self, pos_final, lista_de_vecinos):
        distancias_vecinos = []
        for i in range(len(lista_de_vecinos)):
            # Que no sea mueble ni este en celdas limpias
            agente_vecino = self.get_agente_from_pos(lista_de_vecinos[i].pos)

            if isinstance(agente_vecino, Mueble) != True and lista_de_vecinos[i].pos in self.movimientos_recorrido:
                distancia = round(self.get_distance(agente_vecino.pos, pos_final), 3)
                distancias_vecinos.append((i, distancia))

        for i in range(len(lista_de_vecinos)):
            # Que no sea mueble ni este en celdas limpias
            agente_vecino = self.get_agente_from_pos(lista_de_vecinos[i].pos)
            if isinstance(agente_vecino, Mueble) != True:
                distancia = round(self.get_distance(agente_vecino.pos, pos_final), 3)
                distancias_vecinos.append((i, distancia))


        index_min_distancia = min(distancias_vecinos, key=lambda x: x[1])[0]
        min_distancia = lista_de_vecinos[index_min_distancia]
        self.sig_pos = min_distancia.pos
        
        if len(self.movimientos) > 1:
            if self.sig_pos == self.movimientos[-1]:
                print("sacamos una pos")
                if len(self.recorrido) > 0:
                    self.recorrido.pop(0)
                
    def seleccionar_nueva_pos(self, lista_de_vecinos):
        # print(self.recorrido)
        if len(self.recorrido) > 0:
            if self.pos == self.recorrido[0]:
                self.movimientos_recorrido.append(self.recorrido[0])
                self.recorrido.pop(0)
                for i in lista_de_vecinos:
                    if isinstance(i, EstacionDeCarga) and self.carga < 80:
                        self.cargar()
                if len(self.recorrido) > 0:
                    while(isinstance(self.get_agente_from_pos(self.recorrido[0]), Mueble)):
                        self.recorrido.pop(0)
                        if(len(self.recorrido) == 0):
                            break
                
            if len(self.recorrido) > 0:
                self.dirigirse(self.recorrido[0], lista_de_vecinos)

    def get_distance(self, p1, p2):
        term_x = (p2[0] - p1[0])**2
        term_y = (p2[1] - p1[1])**2
        distance = math.sqrt(term_x + term_y)
        return distance



    # Hace el metodo publico
    @staticmethod
    def buscar_celdas_sucia(lista_de_vecinos):
        # #Opción 1
        # return [vecino for vecino in lista_de_vecinos
        #                 if isinstance(vecino, Celda) and vecino.sucia]
        # #Opción 2
        celdas_sucias = list()
        for vecino in lista_de_vecinos:
            if isinstance(vecino, Celda) and vecino.sucia:
                celdas_sucias.append(vecino)
                
        return celdas_sucias
    

    def step(self):

        lista_de_vecinos = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False)
    
        

        if RobotLimpieza.busca_contrato == self.unique_id:
            # Encontrar el mayor contrato
            if len(RobotLimpieza.contratos) > 0:
                lens_contratos = [len(i[1]) for i in RobotLimpieza.contratos]
                index_mayor_contrato = lens_contratos.index(max(lens_contratos))
                contrato = RobotLimpieza.contratos[index_mayor_contrato][1]
                RobotLimpieza.contratado = RobotLimpieza.contratos[index_mayor_contrato][0]

                #Agregar mitad de contrato a tu recorrido
                self.recorrido.extend(contrato[len(contrato)//2::])
                print(f'Contrato: {contrato}\nExtend part: {self.recorrido}')
                RobotLimpieza.contratos = []
                RobotLimpieza.busca_contrato = -1
            
            #Quitar esa parte del otro agente
        elif RobotLimpieza.busca_contrato == self.unique_id and len(RobotLimpieza.contratos) == 0:
            self.prendido = False
        elif RobotLimpieza.contratado == self.unique_id:
            if len(self.recorrido) > 2:
                self.recorrido = self.recorrido[::len(self.recorrido)//2]
            RobotLimpieza.contratado = -1
        # Subir contrato
        elif RobotLimpieza.busca_contrato != -1 and len(self.recorrido) > 2:
            RobotLimpieza.contratos.append((self.unique_id, self.recorrido))

            print("RobotLimpieza")
            print(RobotLimpieza.contratos)

        elif len(self.recorrido) == 0 and RobotLimpieza.busca_contrato == -1:
            print("Busca contrato")
            RobotLimpieza.busca_contrato = self.unique_id
        elif self.estacion_de_carga != None:
            self.cargar()
        else: 

            celdas_sucias = self.buscar_celdas_sucia(lista_de_vecinos)

            if len(celdas_sucias) == 0:
                if self.carga < 30:
                    self.cargar()
                else:

                    self.seleccionar_nueva_pos(lista_de_vecinos)
            else:
                self.limpiar_una_celda(celdas_sucias)
                # self.seleccionar_nueva_pos(lista_de_vecinos)



    def advance(self):
        if self.pos != self.sig_pos:    
            self.movimientos.append(self.sig_pos)
            Habitacion.agregar_movimiento(self.model)
        if self.prendido == False:
            self.sig_pos = self.pos
            self.model.grid.move_agent(self, self.sig_pos)
        elif self.carga > 0:
            self.carga -= 1
            self.model.grid.move_agent(self, self.sig_pos)

class Habitacion(Model):
    def __init__(self, M: int, N: int,
                 num_agentes: int = 5,
                 porc_celdas_sucias: float = 0.6,
                 porc_muebles: float = 0.1,
                 modo_pos_inicial: str = 'Fija',
                 ):

        self.tiempo_inicial = time.time()
        self.tiempo_final = 0
        self.num_agentes = num_agentes
        self.porc_celdas_sucias = porc_celdas_sucias
        self.porc_muebles = porc_muebles
        self.recargas = 0
        self.movimientos_totales = 0

        self.grid = MultiGrid(M, N, False)
        self.schedule = SimultaneousActivation(self)

        posiciones_disponibles = [pos for _, pos in self.grid.coord_iter()]
        
        # Posicionamiento de estaciones de carga
        #Cuadrnte 1
        # (M//4, N//4) = (5, 5)

        #Cuadrante 2
        # (M//4, N*3//4) = (5, 15)

        #Cuadrante 3
        # (M*3//4, N//4) = (15, 5)

        #Cuadrante 4
        # (M*3//4, N*3//4) = (15, 15)


        self.posiciones_estaciones_carga = [(M//4, N//4),
                                       (M//4, N*3//4),
                                       (M*3//4, N//4),
                                       (M*3//4, N*3//4)]
        for id, pos in enumerate(self.posiciones_estaciones_carga):
            estacion = EstacionDeCarga(id+1, self)
            self.grid.place_agent(estacion, pos)
            posiciones_disponibles.remove(pos)
            #Fix the problem
            # 

        # Posicionamiento de muebles
        num_muebles = int(M * N * porc_muebles)
        posiciones_muebles = self.random.sample(posiciones_disponibles, k=num_muebles)
        
        mueblesPos = posiciones_muebles.copy()

        # self.datacollector = DataCollector(
        #       model_reporters={"Tiempo_limpieza": self.get_grid,
        #                       "Movimientos_realizados": self.get_on_fire,
        #                        "Recargas": self.get_burned}
        #   )

        for id, pos in enumerate(posiciones_muebles):
            # print(pos)
            mueble = Mueble(int(f"{num_agentes}0{id}") + 1, self)
            self.grid.place_agent(mueble, pos)
            posiciones_disponibles.remove(pos)

        # Posicionamiento de celdas sucias
        self.num_celdas_sucias = int(M * N * porc_celdas_sucias)
        posiciones_celdas_sucias = self.random.sample(
            posiciones_disponibles, k=self.num_celdas_sucias)

        for id, pos in enumerate(posiciones_disponibles):
            suciedad = pos in posiciones_celdas_sucias
            celda = Celda(int(f"{num_agentes}{id}") + 1, self, suciedad)
            self.grid.place_agent(celda, pos)

        # Posicionamiento de agentes robot
        if modo_pos_inicial == 'Aleatoria':
            pos_inicial_robots = self.random.sample(posiciones_disponibles, k=num_agentes)
        else:  # 'Fija'
            pos_inicial_robots = [(1, 1)] * num_agentes


        for id in range(num_agentes):
            reverse_count = False
            recorrido = []
            # Crear un recorrido para cada robot dividido en zonas en base al numero de robots
            for i in range(M//num_agentes*id, M//num_agentes*id + M//num_agentes):
                if reverse_count == False:
                    for n in range(M):
                        recorrido.append((i, n))
                    reverse_count = True
                else:
                    for n in reversed(range(M)):
                        recorrido.append((i, n))
                    reverse_count = False
            robot = RobotLimpieza(id, self, mueblesPos, recorrido)
            self.grid.place_agent(robot, pos_inicial_robots[id])
            self.schedule.add(robot)

        self.datacollector = DataCollector(
            model_reporters={"Grid": get_grid, "Cargas": get_cargas,
                             "CeldasSucias": get_sucias},
        )

    def step(self):
        if get_sucias(self) != 0:
            self.datacollector.collect(self)

            self.schedule.step()
        else:
            if self.tiempo_final == 0:
                self.tiempo_final = time.time()
            self.resultados()

    def agregar_recarga(self):
        self.recargas += 1
    
    def agregar_movimiento(self):
        self.movimientos_totales += 1

    def pos_estaciones_carga(self):
        return self.posiciones_estaciones_carga
    
    def resultados(self):
        tiempo_limpieza = str(datetime.timedelta(seconds=(self.tiempo_final - self.tiempo_inicial)))
        movimientos_realizados = self.movimientos_totales
        recargas = self.recargas
        print(f'''
    Tiempo de lipieza: {tiempo_limpieza}
    Movimientos_realizados: {movimientos_realizados}
    Cantidad de recargas: {recargas}
    ''')

    def todoLimpio(self):
        for (content, x, y) in self.grid.coord_iter():
            for obj in content:
                if isinstance(obj, Celda) and obj.sucia:
                    return False
        return True
    


 


def get_grid(model: Model) -> np.ndarray:
    """
    Método para la obtención de la grid y representarla en un notebook
    :param model: Modelo (entorno)
    :return: grid
    """
    grid = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, pos = cell
        x, y = pos
        for obj in cell_content:
            if isinstance(obj, RobotLimpieza):
                grid[x][y] = 2
            elif isinstance(obj, Celda):
                grid[x][y] = int(obj.sucia)
    return grid


def get_cargas(model: Model):
    return [(agent.unique_id, agent.carga) for agent in model.schedule.agents]


def get_sucias(model: Model) -> int:
    """
    Método para determinar el número total de celdas sucias
    :param model: Modelo Mesa
    :return: número de celdas sucias
    """
    sum_sucias = 0
    for cell in model.grid.coord_iter():
        cell_content, pos = cell
        for obj in cell_content:
            if isinstance(obj, Celda) and obj.sucia:
                sum_sucias += 1
    return sum_sucias / model.num_celdas_sucias


def get_movimientos(agent: Agent) -> dict:
    if isinstance(agent, RobotLimpieza):
        return {agent.unique_id: agent.movimientos}
    # else:
    #    return 0(self, unique_id, model, suciedad: bool = False):
        # super().__init__(unique_id, model)
        # self.sucia = suciedad

# def graficar():
#     pass


