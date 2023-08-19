from mesa.model import Model
from mesa.agent import Agent
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector

import numpy as np
import math


class Celda(Agent):
    def __init__(self, unique_id, model, suciedad: bool = False):
        super().__init__(unique_id, model)
        self.sucia = suciedad

class Mueble(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class EstacionDeCarga(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.ocupada = False

        #AGB sin probar
    def recargar(self):
        self.ocupada = True
        self.enCarga = self.model.grid.get_cell_list_contents([self.pos])
        for agent in self.enCarga:
            
            if isinstance(agent, RobotLimpieza):
                agent.carga = 100

class RobotLimpieza(Agent):
    celdas_limpias = []
    def __init__(self, unique_id, model, mueblesPos, recorrido):
        super().__init__(unique_id, model)
        self.mueblesPos = mueblesPos
        self.sig_pos = None
        self.movimientos = list()
        self.carga = 100
        self.recorrido = recorrido
        # print(recorrido)
        
    #AGB sin probar
    def find_nearest(self, agent_type):
        agent = self.model.grid.get_cell_list_contents([self.pos])
        for agent in agent:
            if isinstance(agent, agent_type):
                return agent
        return None    

    def limpiar_una_celda(self, lista_de_celdas_sucias):
        celda_a_limpiar = self.random.choice(lista_de_celdas_sucias)
        celda_a_limpiar.sucia = False
        self.sig_pos = celda_a_limpiar.pos
        RobotLimpieza.celdas_limpias.append(celda_a_limpiar.pos)
        # print(RobotLimpieza.celdas_limpias)

    def seleccionar_nueva_pos(self, lista_de_vecinos):
        if self.pos == self.recorrido[0]:
            self.recorrido.pop(0)
            self.sig_pos = self.recorrido[0]
        else:
            distancias_vecinos = []
            for i in lista_de_vecinos:
                distancias_vecinos.append(self.get_distance(self.recorrido[0], i.pos))
                
            index_min_distancia = distancias_vecinos.index(min(distancias_vecinos))
            min_distancia = lista_de_vecinos[index_min_distancia]
            self.sig_pos = min_distancia.pos

        # while True:
        #     #Checa si la siguiente posicion es un mueble para poderse mover
        #     #sin checar
        #     self.sig_pos = self.random.choice(lista_de_vecinos).pos
        #     if self.sig_pos not in self.mueblesPos:
        #         break

    def get_distance(self, p1, p2):
        term_x = (p2[0] - p1[0])**2
        term_y = (p2[1] - p1[1])**2
        distance = math.sqrt(term_x + term_y)
        return distance

    # def get_nearest_station(self, pos):
    #     posiciones_estaciones_carga = [(1, 1), (1, self.model.M-2), (N-2, 1), (M-2, N-2)]
    #     distances = []
    #     for 




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

        ontops = self.model.grid.get_cell_list_contents([self.pos])

        vecinos = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False)

        for vecino in vecinos:
            if isinstance(vecino, (Mueble, RobotLimpieza, EstacionDeCarga)) and vecino in RobotLimpieza.celdas_limpias:
                vecinos.remove(vecino)
        
        find_nearest = self.find_nearest(Celda)

        celdas_sucias = self.buscar_celdas_sucia(vecinos)

        if len(celdas_sucias) == 0:
            self.seleccionar_nueva_pos(vecinos)
        else:
            self.limpiar_una_celda(celdas_sucias)
        #AGB Se carga cada step 25 y no se mueve hasta llegar a 100 de carga y no se pasa de 100
        if isinstance(ontops[0], EstacionDeCarga):
            self.carga += 25
            if self.carga < 100:
                self.sig_pos = self.pos
            if self.carga > 100:
                self.carga = 100


    def advance(self):
        if self.pos != self.sig_pos:
            self.movimientos.append(self.sig_pos)
            
        if self.carga > 0:
            self.carga -= 1
            self.model.grid.move_agent(self, self.sig_pos)


class Habitacion(Model):
    def __init__(self, M: int, N: int,
                 num_agentes: int = 5,
                 porc_celdas_sucias: float = 0.6,
                 porc_muebles: float = 0.1,
                 modo_pos_inicial: str = 'Fija',
                 ):

        self.num_agentes = num_agentes
        self.porc_celdas_sucias = porc_celdas_sucias
        self.porc_muebles = porc_muebles

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


        posiciones_estaciones_carga = [(M//4, N//4),
                                       (M//4, N*3//4),
                                       (M*3//4, N//4),
                                       (M*3//4, N*3//4)]
        for id, pos in enumerate(posiciones_estaciones_carga):
            estacion = EstacionDeCarga(id+1, self)
            self.grid.place_agent(estacion, pos)
            posiciones_disponibles.remove(pos)
            #Fix the problem
            # 

        # Posicionamiento de muebles
        num_muebles = int(M * N * porc_muebles)
        posiciones_muebles = self.random.sample(posiciones_disponibles, k=num_muebles)
        
        mueblesPos = posiciones_muebles.copy()

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
            recorrido = []
            for i in range(M//num_agentes*id, M//num_agentes*id + M//num_agentes):
                for n in range(M):
                    recorrido.append((i, n))
            robot = RobotLimpieza(id, self, mueblesPos, recorrido)
            self.grid.place_agent(robot, pos_inicial_robots[id])
            self.schedule.add(robot)

        self.datacollector = DataCollector(
            model_reporters={"Grid": get_grid, "Cargas": get_cargas,
                             "CeldasSucias": get_sucias},
        )

    def step(self):
        self.datacollector.collect(self)

        self.schedule.step()

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


