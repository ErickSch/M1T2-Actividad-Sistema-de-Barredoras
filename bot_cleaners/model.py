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
    posiciones = []
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.ocupada = False
        if len(EstacionDeCarga.posiciones) <= unique_id:
            EstacionDeCarga.posiciones.append(self.pos)
        # print(EstacionDeCarga.posiciones)

        #AGB sin probar
    def recargar(self):
        self.ocupada = True
        self.enCarga = self.model.grid.get_cell_list_contents([self.pos])
            
        if isinstance(self.enCarga, RobotLimpieza):
            self.enCarga.carga = 100
            self.ocupada = False
# Que los robots que pasen muy cerca de estaciones se pongan a cargar sin importar su carga
# Si se está cargando, agregar su recorrido a la lista de otro robot cercano (contrato)
    # y si se termina de cargar antes, seguir con su recorrido e ir borrando las celdas
    # del recorrido del otro
# Funcionalidad de sacarle la vuelta a muebles.
class RobotLimpieza(Agent):

    celdas_limpias = []
    busca_contrato = -1
    contratos = []
    contratado = -1

    def __init__(self, unique_id, model, mueblesPos, recorrido):
        super().__init__(unique_id, model)
        self.mueblesPos = mueblesPos
        self.sig_pos = None
        self.movimientos = list()
        self.carga = 100
        self.recorrido = recorrido
        self.estacion_de_carga = None

        # print(recorrido)
        
    #AGB sin probar
    def find_nearest(self, agent_type):
        agent = self.model.grid.get_cell_list_contents([self.pos])
        for agent in agent:
            if isinstance(agent, agent_type):
                return agent
        return None 

    # def carga_cercana(self):

        # Tu -> Subir posicion
        # Otros -> Aceptar los que esten cerca
        # Tu -> Agarrar al mas cercano
        # 
    
    def get_agente_from_pos(self, pos):
        agente = self.model.grid.get_cell_list_contents([pos])[0]
        return agente

    def limpiar_una_celda(self, lista_de_celdas_sucias):
        
        celda_a_limpiar = self.random.choice(lista_de_celdas_sucias)
        celda_a_limpiar.sucia = False
        self.sig_pos = celda_a_limpiar.pos
        RobotLimpieza.celdas_limpias.append(celda_a_limpiar.pos)
        # print(RobotLimpieza.celdas_limpias)

    # tiene menos de 30 cargar
    # si no tiene coordenada en cargar: calcularla y ponerla
    # Si tiene coordenada en cargar: dirigirse a esa coordenada
    # si la coordenada actual es la misma que la estacion y la pila es 100 quitar el valor de estacion de carga

    def cargar(self):
        print(f'CARGAR:  {self.unique_id}-------')
        # siguiente_celda = self.model.grid.get_cell_list_contents([self.recorrido[0]])
        if self.estacion_de_carga == None:
            distancias_carga = []
            posiciones = Habitacion.pos_estaciones_carga(self.model)
            for i in posiciones:
                distancias_carga.append(self.get_distance(self.pos, i))
            min_index = distancias_carga.index(min(distancias_carga))
            self.estacion_de_carga = posiciones[min_index]
        elif self.pos == self.estacion_de_carga:
            self.carga += 25
            if self.carga < 100:
                self.sig_pos = self.pos
            if self.carga > 100:
                self.carga = 100
        elif self.pos == self.estacion_de_carga and self.carga == 100:
            self.estacion_de_carga = None
        elif self.estacion_de_carga:
            lista_de_vecinos = self.model.grid.get_neighbors(
                self.pos, moore=True, include_center=False)
            self.dirigirse(self.estacion_de_carga, lista_de_vecinos)

    def dirigirse(self, pos_final, lista_de_vecinos):
        # print(f'El robot {self.unique_id} esta en {self.pos} y se dirige a {pos_final}')
        distancias_vecinos = []
        for i in range(len(lista_de_vecinos)):
            # Que no sea mueble ni este en celdas limpias
            agente_vecino = self.get_agente_from_pos(lista_de_vecinos[i].pos)
            if isinstance(agente_vecino, Mueble) != True and lista_de_vecinos[i].pos in self.movimientos:
                distancia = round(self.get_distance(agente_vecino.pos, pos_final), 3)
                distancias_vecinos.append((i, distancia))
        
        for i in range(len(lista_de_vecinos)):
            # Que no sea mueble ni este en celdas limpias
            agente_vecino = self.get_agente_from_pos(lista_de_vecinos[i].pos)
            if isinstance(agente_vecino, Mueble) != True:
                distancia = round(self.get_distance(agente_vecino.pos, pos_final), 3)
                distancias_vecinos.append((i, distancia))

        # if isinstance(agente_vecino, Mueble) != True:
        #     distancia = round(self.get_distance(agente_vecino.pos, pos_final), 3)
        #     distancias_vecinos.append((i, distancia))

        index_min_distancia = min(distancias_vecinos, key=lambda x: x[1])[0]
        min_distancia = lista_de_vecinos[index_min_distancia]
        self.sig_pos = min_distancia.pos
        if self.unique_id == 1:

            print(f'\nEl robot {self.unique_id} esta en {self.pos} y se mueve a {self.sig_pos} porque es la mas cercana a {pos_final} con las distancias {distancias_vecinos} de los puntos {[i.pos for i in lista_de_vecinos]}')
 

    def seleccionar_nueva_pos(self, lista_de_vecinos):
        # print("Seleccionando nueva pos")
        # print(f'Posicion: {self.pos}')
        if len(self.recorrido) == 0:
            self.cargar()
        elif self.pos == self.recorrido[0]:
            self.recorrido.pop(0)
            
            while(isinstance(self.get_agente_from_pos(self.recorrido[0]), Mueble)):
                print("Esta es mueble")
                self.recorrido.pop(0)

        self.dirigirse(self.recorrido[0], lista_de_vecinos)

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

        # celda_actual = self.model.grid.get_cell_list_contents([self.pos])

        lista_de_vecinos = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False)
        
        #AGB Se carga cada step 25 y no se mueve hasta llegar a 100 de carga y no se pasa de 100

        # elif RobotLimpieza.busca_contrato == self.unique_id:
        #     # Encontrar el mayor contrato
        #     lens_contratos = [len(i) for i in RobotLimpieza.contratos]
        #     index_mayor_contrato = max(lens_contratos)
        #     contrato = RobotLimpieza.contratos[index_mayor_contrato]

        #     #Agregar mitad de contrato a tu recorrido
        #     self.recorrido.extend(contrato[len(contrato)/2::])
        #     #Quitar esa parte del otro agente

        #     # print(f'Contrato: {contrato}\nExtend part: {self.recorrido}')
        
        # elif RobotLimpieza.busca_contrato != -1:
        #     RobotLimpieza.contratos.append(self.recorrido)

        # elif len(self.recorrido) == 0 and RobotLimpieza.busca_contrato == -1:
        #     RobotLimpieza.busca_contrato = self.unique_id
        if self.estacion_de_carga:
            self.cargar()
        else: 
            # print(self.recorrido)

            # for vecino in lista_de_vecinos:
            #     if isinstance(vecino, (Mueble, RobotLimpieza, EstacionDeCarga)) and vecino in RobotLimpieza.celdas_limpias:
            #         lista_de_vecinos.pop(lista_de_vecinos.index(vecino))
            
            # find_nearest = self.find_nearest(Celda)

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
        # print(f'Id: {self.unique_id}\nAct_pos: {self.pos}\nSig_pos: {self.sig_pos}\nRecorrido[0]: {self.recorrido[0]}')
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
            for i in range(M//num_agentes*id, M//num_agentes*id + M//num_agentes-2):
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
        self.datacollector.collect(self)

        self.schedule.step()

    def pos_estaciones_carga(self):
        return self.posiciones_estaciones_carga

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


