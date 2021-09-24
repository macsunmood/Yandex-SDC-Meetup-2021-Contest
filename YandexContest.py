import numpy as np
import heapq

# Коды возможных действий ровера
MOVE  = 0
STAY  = 1
TAKE  = 2
PUT   = 3

# Коды возможных перемещений ровера
UP    = 10
DOWN  = 11
LEFT  = 12
RIGHT = 13


# Словарь действий ровера
ACTIONS_DICT = {
    UP   : {'sign': 'U', 'move': [-1,  0]},  # движение на одну клетку вверх (уменьшить номер строки)
    DOWN : {'sign': 'D', 'move': [ 1,  0]},  # движение на одну клетку вниз (увеличить номер строки) 
    LEFT : {'sign': 'L', 'move': [ 0, -1]},  # движение на одну клетку влево (уменьшить номер столбца)
    RIGHT: {'sign': 'R', 'move': [ 0,  1]},  # движение на одну клетку вправо (увеличить номер столбца)
    STAY : {'sign': 'S'},  # остаться на месте и ничего не делать
    TAKE : {'sign': 'T'},  # остаться на месте и забрать самый старый заказ в текущей клетке
    PUT  : {'sign': 'P'},  # остаться на месте и выдать заказ в текущей клетке
}

# Коды поисковых задач ровера
SEARCH_ORDER = 100
SEARCH_PATH = 101

# Коды возможных статусов заказа
PENDING     = 200
IN_PROGRESS = 201
DONE        = 202

ITER_DURATION = 60  # итерация состоит из 60 секунд


# https://gist.github.com/ryancollingwood/32446307e976a11a1185a5394d6657bc
class Node:
    '''A node class for A* Pathfinding'''
    def __init__(self, parent=None, pos=None):
        self.parent = parent
        self.pos = pos
        self.g = self.h = self.f = 0

    def __eq__(self, other):
        return self.pos == other.pos
    
    def __repr__(self):
        return f'{self.pos} - g: {self.g} h: {self.h} f: {self.f}'

    # defining less than for purposes of heap queue
    def __lt__(self, other):
        return self.f < other.f
    
    # defining greater than for purposes of heap queue
    def __gt__(self, other):
        return self.f > other.f

def return_path(current_node):
    path = []
    current = current_node
    while current is not None:
        path.append(current.pos)
        current = current.parent
        
    path = path[::-1]
        
    moves = [np.array(path[i + 1]) - path[i]
             for i in range(len(path) - 1)]

    get_next_move = lambda x, y: {
        x == -1: UP, 
        x ==  1: DOWN, 
        y == -1: LEFT,
        y ==  1: RIGHT, 
    }[True]

    return [get_next_move(*m) for m in moves]

def find_shortest_path(grid, begin, end):
    N = len(grid)
    start_node = Node(None, begin)
    end_node = Node(None, end)
    open_list = []
    closed_list = []
    
    heapq.heapify(open_list) 
    heapq.heappush(open_list, start_node)
    
    iterations = 0
    while len(open_list) > 0:
        iterations += 1
        if iterations > (N * N // 2):
            return return_path(current_node)       
        
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)
        
        if current_node == end_node:
            return return_path(current_node)
        children = []
        
        # Проверяем соседние точки по всем четырем направлениям:
        for shift in ((0, -1), (0, 1), (-1, 0), (1, 0)): # adjacent squares            
            xc, yc = np.add(current_node.pos, shift)  # get node pos
            if not 0 <= xc < N or not 0 <= yc < N:  # make sure within range
                continue
            if grid[xc, yc] != 0:  # make sure walkable terrain
                continue
            # create new node and append to children dict
            children.append(Node(current_node, [xc, yc]))
            
        for ch in children:
            if len([closed_ch for closed_ch in closed_list 
                    if closed_ch == ch]) > 0:
                continue
            ch.g = current_node.g + 1
            ch.h = sum([(ch.pos[0] - end_node.pos[0]) ** 2,
                        (ch.pos[1] - end_node.pos[1]) ** 2])
            ch.f = ch.g + ch.h
            if len([open_node for open_node in open_list 
                    if ch.pos == open_node.pos and ch.g > open_node.g]) > 0:
                continue
            heapq.heappush(open_list, ch)
    return None


class Rover:
    def __init__(self, r_id, x, y, search_task=SEARCH_ORDER):
        self.r_id = r_id
        self.x = x
        self.y = y
        self.search_task = search_task
        self.o_id = None
        self.loaded = False
        self.earned = 0
        self.route = None
        self.next_action = None
        self.actions_str = ''
    
    def find_order(self, orders_pool, grid):
        orders_list = []
        for o_id, order in orders_pool.items():
            if order.status == PENDING:
                order.min_path = find_shortest_path(
                    grid, 
                    begin=[self.x, self.y], 
                    end=[order.x_from, order.y_from]
                )
                orders_list.append(o_id)

        if len(orders_list) == 0:
            self.next_action = STAY
            return None
        
        min_time = min([orders_pool[o_id].time_passed 
                        for o_id in orders_list])
        min_time_orders = [o_id for o_id in orders_list 
                           if orders_pool[o_id].time_passed == min_time]
        self.o_id = sorted(min_time_orders, 
                           key=lambda o_id: len(orders_pool[o_id].min_path), 
                           reverse=True)[0]
        self.route = orders_pool[self.o_id].min_path
        self.next_action = MOVE
        return orders_pool[self.o_id]
    
    def find_path(self, orders_pool, grid):
        x_end, y_end = 0, 0
        order = orders_pool[self.o_id]
        x_end, y_end = order.x_to, order.y_to
        self.route = find_shortest_path(
            grid, 
            begin=[self.x, self.y], 
            end=[x_end, y_end]
        )
        self.next_action = MOVE
    
    def move(self):
        current_move = ACTIONS_DICT[self.route.pop(0)]
        self.x, self.y = np.add([self.x, self.y], current_move['move'])
        self.actions_str += current_move['sign']
        if len(self.route) == 0:
            self.next_action = PUT if self.loaded else TAKE
            
    def stay(self):
        self.actions_str += ACTIONS_DICT[STAY]['sign']

    def take(self):
        self.loaded = True
        self.search_task = SEARCH_PATH
        self.actions_str += ACTIONS_DICT[TAKE]['sign']
        
    def put(self, earned):
        self.earned += earned
        self.o_id = None
        self.loaded = False        
        self.route = None
        self.search_task = SEARCH_ORDER
        self.actions_str += ACTIONS_DICT[PUT]['sign']


class Order:
    def __init__(self, x_from, y_from, x_to, y_to, status):
        self.x_from = x_from
        self.y_from = y_from
        self.x_to = x_to
        self.y_to = y_to
        self.status = status
        self.time_passed = 0


def input_error(condition):
    return f'INPUT ERROR: Argument out of range ({condition})'

def process_input_start(input_hook=None):
    '''В этой функции парсим поступающий стартовый ввод и извлекаем из него полезные данные'''
    if input_hook:  # определяем источник: std.input или список в памяти
        pixie = lambda: input_hook.pop(0)
    else:
        pixie = lambda: input()
    # Парсим pixie 
    (
        N,        # карта города имеет размер N * N
        MaxTips,  # максимальное количество чаевых для одного заказа
        Cost_c    # стоимость постройки каждого робота
    ) = [int(n) for n in pixie().split()]  # парсим начало input
    
    # Представляем карту города в виде матрицы { 0 - пусто; 1 - препятствие }
    CityMap = np.zeros((N, N), dtype=int)
    for i in range(N):
        row = [1 if ch == '#' else 0 for ch in pixie()]
        CityMap[i, :] = row
    
    (
        T,  # количество итераций взаимодействия
        D   # суммарное количество заказов
    ) = [int(n) for n in pixie().split()]  # парсим окончание input
    
    # Проверяем все переменные на соответствие диапазонам, заданным в условиях задачи
    for condition in ('N <= 2000', 'MaxTips <= 50000', 'Cost_c <= 10**9', 
                      'T <= 100000', 'D <= 10000000'):
        assert eval(condition), input_error(condition)
    return N, CityMap, MaxTips, Cost_c, T, D

def process_input_orders(N, input_hook=None):
    '''Функция для парсинга информации о новых поступающих заказах'''
    if input_hook:  # определяем источник: std.input или список в памяти
        pixie = lambda: input_hook.pop(0)
    else:
        pixie = lambda: input()
    # Парсим pixie
    k = int(pixie())  # количество новых курьерских заказов
    
    new_orders = []    
    for i in range(k):
        # координаты начальной и конечной точки заказа
        S_row, S_col, F_row, F_col = [int(c) for c in pixie().split()]

        # проверяем координаты на соответствие: (1 ≤ S_row, S_col, F_row, F_col ≤ N)
        for condition in [f'1 <= {coord} <= N' 
                          for coord in ('S_row', 'S_col', 'F_row', 'F_col')]:
            assert eval(condition), input_error(condition)

        # адаптируем к внутренней системе координат и добавляем в список новых заказов
        new_orders.append(list(map(lambda x: x - 1, 
                                   (S_row, S_col, F_row, F_col))))
    return new_orders

def random_orders(N, grid, kmax=5):
    k = np.random.randint(kmax)
    orders = []
    obstacles = [(x, y) for x in range(N) for y in range(N) if grid[x, y] == 1]
    for i in range(k):
        ok = False
        while not ok:
            start_coords  = (np.random.randint(1, N + 1), 
                             np.random.randint(1, N + 1))
            finish_coords = (np.random.randint(1, N + 1), 
                             np.random.randint(1, N + 1))
            if (start_coords not in obstacles and 
                finish_coords not in obstacles and
                start_coords != finish_coords):
                ok = True
        S_row, S_col, F_row, F_col = *start_coords, *finish_coords
        orders += [f'{S_row} {S_col} {F_row} {F_col}']
    return [str(k)] + orders


class Simulation:
    def run(self, input_hook=None):
        (self.N, 
         self.grid, 
         self.MaxTips, 
         self.Cost_c, 
         self.T, 
         self.D) = process_input_start(input_hook)
        (self.R, 
         self.rovers_pool) = self.decide_start()
        
        return_tips = True if input_hook else False
        total_tips = self.run_iterations(input_hook, return_tips)
        if total_tips:
            self.results(total_tips)
        
    def decide_start(self):
        free_cells = np.argwhere(self.grid == 0)  # координаты всех пустых клеток
        # Для обеспечения доставки всех заказов минимальное число роверов 
        # считаем как: D {общее количество заказов} / T {количество итераций}:
        R = max(1, min(round(self.D / self.T), 100))  # количество размещаемых роверов (1 ≤ R ≤ 100)
        # Определим начальное расположение всех роверов
        rovers_pool = []  # пул роверов
        for r_id in range(R):
            # располагаем радномно в свободных клетках:
            x, y = free_cells[np.random.choice(len(free_cells))]
            rovers_pool.append(
                Rover(r_id, x, y, search_task=SEARCH_ORDER)  # спауним новый ровер
            )
        # Передадим службе ответную информацию
        print(R, flush=True)  # количество роверов
        for r in rovers_pool:
            print(r.x + 1, r.y + 1, flush=True)  # координаты - числа от 1 до N
        return R, rovers_pool

    def run_iterations(self, input_hook=None, return_tips=False):
        self.orders_pool = {}
        last_o_id = 0
        # Производим T итераций
        for it in range(self.T):
            # На каждой итерации получаем информацию о новых размещенных заказах
            new_orders = process_input_orders(self.N, input_hook)  # считываем новые заказы
            if len(new_orders) > 0:
                for o_id, o_coords in enumerate(new_orders):
                    self.orders_pool[last_o_id + o_id] = Order(
                        *o_coords, status=PENDING,  # спауним новый заказ
                    )
                last_o_id += len(new_orders)

            response = self.iteration(self.R, self.rovers_pool, self.orders_pool)

            # На каждой итерации в ответ возвращаем по 60 действий для каждого из роверов
            for actions in response:
                print(actions, flush=True)
        if return_tips:
            return sum([rover.earned for rover in self.rovers_pool])
        
    def iteration(self, R, rovers_pool, orders_pool):
        '''Механика итерации'''
        iter_actions = []
        for iter_time in range(ITER_DURATION):
            # Сначала обрабатываем все поисковые задачи роверов (поиск заказов, поиск маршрута)
            for r_id in range(R):
                self.search_logic(rovers_pool[r_id])
            # Далее обрабатываем действия роверов (движение, получение и отгрузка заказа)
            for r_id in range(R):
                self.actions_logic(rovers_pool[r_id])
            # Добавляем секунду к времени существования всех заказов
            for order in orders_pool.values():
                order.time_passed += 1
        # Формируем и возвращаем список действий всех роверов в текущей итерации
        for r_id in range(R):
            iter_actions.append(rovers_pool[r_id].actions_str)
            rovers_pool[r_id].actions_str = ''
        return iter_actions

    def search_logic(self, rover):
        '''Поисковая логика роверов'''
        # Подключаем свободные роверы к обработке ближайших ожидающих заказов
        if rover.search_task == SEARCH_ORDER:
            order = rover.find_order(self.orders_pool, self.grid)
            if order is not None:
                order.status = IN_PROGRESS
                rover.search_task = None
        # Прокладываем кратчайший маршрут
        if rover.search_task == SEARCH_PATH:
            rover.find_path(self.orders_pool, self.grid)
            rover.search_task = None

    def actions_logic(self, rover):
        '''Логика действий роверов'''
        if rover.next_action == MOVE:
            rover.move()
        elif rover.next_action == STAY:
            rover.stay()
        elif rover.next_action == TAKE:
            rover.take()
        elif rover.next_action == PUT:
            order = self.orders_pool[rover.o_id]
            order.status = DONE
            earned = max(0, self.MaxTips - order.time_passed)
            rover.put(earned)
            
    def results(self, total_tips):
        # Итоговое количество очков, заработанное за один тест
        result = total_tips - self.R * self.Cost_c
        print(f'Total tips: {total_tips}; Total build cost: {self.R * self.Cost_c}')
        print(f'Result: {result}')


def read_test(file='01', test_dir='examples\\'):
    with open(test_dir + file) as f:
        return [row.strip() for row in f]

# input_hook = read_test('02')
input_hook = None


if __name__ == '__main__':
	sim = Simulation()
	sim.run(input_hook)