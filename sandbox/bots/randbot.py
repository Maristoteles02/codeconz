import random
import json
import heapq
from bots import bot

class RandBot(bot.Bot):
    NAME = "RandBot"

    def __init__(self):
        super().__init__()
        self.player_num = None
        self.lighthouses = []
        self.position = None
        self.map = []

    def initialize(self, init_data):
        self.player_num = init_data["player_num"]
        self.position = tuple(init_data["position"])
        self.map = init_data["map"]
        self.lighthouses = [tuple(pos) for pos in init_data["lighthouses"]]
        print(json.dumps({"name": self.NAME}), flush=True)

    def is_valid_move(self, x, y):
        """Verifica si un movimiento es válido."""
        if 0 <= x < len(self.map[0]) and 0 <= y < len(self.map):
            return self.map[y][x] != 0
        return False

    def heuristic(self, a, b):
        """Calcula la distancia Manhattan entre dos puntos."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star_pathfinding(self, start, goal):
        """Algoritmo A* para encontrar la ruta más corta."""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self.reconstruct_path(came_from, current)

            cx, cy = current
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            for dx, dy in neighbors:
                neighbor = (cx + dx, cy + dy)
                if not self.is_valid_move(neighbor[0], neighbor[1]):
                    continue

                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def reconstruct_path(self, came_from, current):
        """Reconstruye la ruta encontrada."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def get_next_move(self, path, current_position):
        """Obtiene el siguiente movimiento basado en la ruta calculada."""
        if len(path) < 2:
            return None
        next_position = path[1]
        dx = next_position[0] - current_position[0]
        dy = next_position[1] - current_position[1]
        return {"x": dx, "y": dy}

    def find_closest_lighthouse(self, cx, cy):
        """Encuentra el faro más cercano al jugador."""
        min_distance = float('inf')
        closest_lh = None

        for lh in self.lighthouses:
            dist = abs(lh[0] - cx) + abs(lh[1] - cy)
            if dist < min_distance:
                min_distance = dist
                closest_lh = lh

        return closest_lh

    def valid_lighthouse_connections(self, state):
        """Encuentra faros válidos para conectar."""
        cx, cy = state['position']
        lighthouses = {tuple(lh["position"]): lh for lh in state["lighthouses"]}
        player_keys = state.get("keys", [])
        connections = []

        if (cx, cy) in lighthouses and lighthouses[(cx, cy)]["owner"] == self.player_num:
            for dest, lh in lighthouses.items():
                if dest != (cx, cy) and lh["have_key"] and [cx, cy] not in lh["connections"]:
                    connections.append(dest)
        return connections

    def find_random_valid_move(self, cx, cy):
        """Encuentra un movimiento aleatorio válido si no hay otras opciones."""
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(moves)

        for dx, dy in moves:
            nx, ny = cx + dx, cy + dy
            if self.is_valid_move(nx, ny):
                return {"x": dx, "y": dy}
        return None

    def play(self, states):
        actions_list = []

        for state in states:
            cx, cy = state["position"]
            lighthouses = {tuple(lh["position"]): lh for lh in state["lighthouses"]}
            
            # 1. Intentar conectar faros
            possible_connections = self.valid_lighthouse_connections(state)
            if possible_connections:
                target = random.choice(possible_connections)
                action = {"command": "connect", "destination": target}
                print(f"[DEBUG] Conectando faro en {cx}, {cy} con {target}")
                actions_list.append(action)
                continue
            
            # 2. Buscar y moverse hacia el faro más cercano
            closest_lighthouse = self.find_closest_lighthouse(cx, cy)
            if closest_lighthouse:
                path = self.a_star_pathfinding((cx, cy), closest_lighthouse)
                if path:
                    move = self.get_next_move(path, (cx, cy))
                    if move:
                        action = {"command": "move", "x": move["x"], "y": move["y"]}
                        print(f"[DEBUG] Moviéndose hacia faro más cercano en {closest_lighthouse}")
                        actions_list.append(action)
                        continue

            # 3. Intentar con faros alternativos
            for other_lh in self.lighthouses:
                if other_lh != closest_lighthouse:
                    path = self.a_star_pathfinding((cx, cy), other_lh)
                    if path:
                        move = self.get_next_move(path, (cx, cy))
                        if move:
                            action = {"command": "move", "x": move["x"], "y": move["y"]}
                            print(f"[DEBUG] Moviéndose hacia faro alternativo en {other_lh}")
                            actions_list.append(action)
                            break

            # 4. Intentar movimientos aleatorios como último recurso
            random_move = self.find_random_valid_move(cx, cy)
            if random_move:
                action = {"command": "move", "x": random_move["x"], "y": random_move["y"]}
                print(f"[DEBUG] Moviéndose aleatoriamente hacia {random_move}")
                actions_list.append(action)
                continue

            # No debería llegar aquí
            print("[ERROR] No se encontró ninguna acción válida. Esto no debería suceder.")
            actions_list.append({"command": "move", "x": 0, "y": 0})

        for action in actions_list:
            print(json.dumps(action), flush=True)

        return actions_list
