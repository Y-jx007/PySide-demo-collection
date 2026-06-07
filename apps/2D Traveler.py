import sys
import numpy as np
import random
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QSpinBox, 
                               QDoubleSpinBox, QProgressBar, QTextEdit, QGroupBox,
                               QCheckBox)
from PySide6.QtCore import QTimer, Qt, QThread, Signal
from PySide6.QtGui import QPainter, QPen, QColor, QFont

# 尝试导入numba进行加速
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # 定义一个空的装饰器，如果numba不可用则不会加速
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# ------------------- 距离计算 (优化版本) -------------------

@jit(nopython=True, parallel=True)
def calculate_distance_matrix_parallel(cities):
    num_cities = len(cities)
    matrix = np.zeros((num_cities, num_cities))
    for i in prange(num_cities):
        for j in range(i+1, num_cities):
            dx = cities[i, 0] - cities[j, 0]
            dy = cities[i, 1] - cities[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            matrix[i, j] = dist
            matrix[j, i] = dist
    return matrix

@jit(nopython=True)
def calculate_distance_fast(distance_matrix, individual):
    total = 0.0
    num_cities = len(individual)
    for i in range(num_cities):
        start_idx = individual[i]
        end_idx = individual[(i + 1) % num_cities]
        total += distance_matrix[start_idx, end_idx]
    return total

@jit(nopython=True, parallel=True)
def calculate_fitness_parallel(distance_matrix, population):
    num_individuals = len(population)
    fitnesses = np.zeros(num_individuals)
    for i in prange(num_individuals):
        individual = population[i]
        total_distance = 0.0
        num_cities = len(individual)
        for j in range(num_cities):
            start_idx = individual[j]
            end_idx = individual[(j + 1) % num_cities]
            total_distance += distance_matrix[start_idx, end_idx]
        fitnesses[i] = 1.0 / total_distance if total_distance > 0 else 1e10
    return fitnesses

# ------------------- 增量2-opt优化 -------------------

@jit(nopython=True)
def incremental_two_opt_numba(distance_matrix, individual, max_iterations=1000):
    n = len(individual)
    improved = True
    iterations = 0
    
    # 预计算当前路径的总距离
    current_distance = 0.0
    for i in range(n):
        current_distance += distance_matrix[individual[i], individual[(i + 1) % n]]
    
    while improved and iterations < max_iterations:
        improved = False
        
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j - i == 1:
                    continue
                
                # 增量计算：只计算变化的部分
                # 要移除的边: (i, i+1) 和 (j, j+1)
                # 要添加的边: (i, j) 和 (i+1, j+1)
                remove_dist = (distance_matrix[individual[i], individual[(i + 1) % n]] +
                             distance_matrix[individual[j], individual[(j + 1) % n]])
                
                add_dist = (distance_matrix[individual[i], individual[j]] +
                          distance_matrix[individual[(i + 1) % n], individual[(j + 1) % n]])
                
                delta = add_dist - remove_dist
                
                if delta < -1e-10:  # 有改进
                    # 反转 i+1 到 j 的段
                    individual[(i + 1):(j + 1)] = individual[(i + 1):(j + 1)][::-1]
                    current_distance += delta
                    improved = True
                    break  # 找到改进就跳出内层循环
            
            if improved:
                break
        
        iterations += 1
    
    return individual, current_distance

@jit(nopython=True)
def fast_two_opt_numba(distance_matrix, individual, max_iterations=50):
    """更快的2-opt实现，限制迭代次数"""
    n = len(individual)
    iterations = 0
    
    while iterations < max_iterations:
        best_delta = 0.0
        best_i = -1
        best_j = -1
        
        # 寻找最佳交换
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j - i == 1:
                    continue
                
                # 增量计算
                remove_dist = (distance_matrix[individual[i], individual[i + 1]] +
                             distance_matrix[individual[j], individual[(j + 1) % n]])
                
                add_dist = (distance_matrix[individual[i], individual[j]] +
                          distance_matrix[individual[i + 1], individual[(j + 1) % n]])
                
                delta = add_dist - remove_dist
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        # 如果找到改进就应用
        if best_delta < -1e-10:
            individual[best_i + 1:best_j + 1] = individual[best_i + 1:best_j + 1][::-1]
            iterations += 1
        else:
            break
    
    return individual

# ------------------- 遗传算法操作 (优化版本) -------------------

@jit(nopython=True)
def tournament_selection_numba(fitnesses, tournament_size, num_selections):
    population_size = len(fitnesses)
    selected_indices = np.zeros(num_selections, dtype=np.int32)
    
    for i in range(num_selections):
        contestants = np.random.choice(population_size, tournament_size, replace=False)
        best_idx = contestants[0]
        for j in range(1, tournament_size):
            if fitnesses[contestants[j]] > fitnesses[best_idx]:
                best_idx = contestants[j]
        selected_indices[i] = best_idx
    
    return selected_indices

@jit(nopython=True)
def ordered_crossover_numba(parent1, parent2):
    size = len(parent1)
    child = np.full(size, -1, dtype=np.int32)
    
    start, end = np.sort(np.random.choice(size, 2, replace=False))
    
    child[start:end+1] = parent1[start:end+1]
    
    current_pos = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[current_pos] in child:
                current_pos += 1
            child[i] = parent2[current_pos]
            current_pos += 1
    
    return child

@jit(nopython=True)
def mutate_swap_numba(individual):
    i, j = np.random.choice(len(individual), 2, replace=False)
    individual[i], individual[j] = individual[j], individual[i]
    return individual

@jit(nopython=True)
def mutate_inversion_numba(individual):
    i, j = np.sort(np.random.choice(len(individual), 2, replace=False))
    individual[i:j+1] = individual[i:j+1][::-1]
    return individual

@jit(nopython=True)
def mutate_scramble_numba(individual):
    i, j = np.sort(np.random.choice(len(individual), 2, replace=False))
    segment = individual[i:j+1].copy()
    np.random.shuffle(segment)
    individual[i:j+1] = segment
    return individual

# ------------------- 遗传算法核心类 (优化版本) -------------------

class TSPGeneticAlgorithmOptimized:
    def __init__(self, cities, population_size=100, mutation_rate=0.01, 
                 generations=1000, elite_size=5, use_2opt=False, two_opt_intensity=1):
        self.cities = cities
        self.num_cities = len(cities)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.elite_size = elite_size
        self.use_2opt = use_2opt
        self.two_opt_intensity = two_opt_intensity  # 控制2-opt的强度
        
        # 使用并行计算距离矩阵
        self.distance_matrix = calculate_distance_matrix_parallel(cities)
        
        self.current_generation = 0
        self.best_individual = None
        self.best_fitness = 0
        self.fitness_history = []
        
        # 预分配数组
        self.population_array = np.zeros((population_size, self.num_cities), dtype=np.int32)

    def create_individual(self):
        individual = np.arange(self.num_cities)
        np.random.shuffle(individual)
        return individual
    
    def create_population(self):
        population = []
        for i in range(self.population_size):
            individual = self.create_individual()
            population.append(individual)
        return population
    
    def selection(self, population, fitnesses):
        selected_indices = tournament_selection_numba(
            np.array(fitnesses), 
            3,
            self.population_size
        )
        return [population[i] for i in selected_indices]
    
    def crossover_batch(self, parents):
        children = []
        num_parents = len(parents)
        for i in range(0, num_parents - 1, 2):
            child1 = ordered_crossover_numba(parents[i], parents[i+1])
            child2 = ordered_crossover_numba(parents[i+1], parents[i])
            children.extend([child1, child2])
        return children
    
    def mutate_batch(self, individuals):
        mutated = []
        for individual in individuals:
            if np.random.random() < self.mutation_rate:
                mutation_type = np.random.randint(0, 3)
                individual_copy = individual.copy()
                if mutation_type == 0:
                    mutated.append(mutate_swap_numba(individual_copy))
                elif mutation_type == 1:
                    mutated.append(mutate_inversion_numba(individual_copy))
                else:
                    mutated.append(mutate_scramble_numba(individual_copy))
            else:
                mutated.append(individual)
        return mutated
    
    def two_opt(self, individual):
        # 使用增量2-opt优化
        optimized_individual = fast_two_opt_numba(
            self.distance_matrix, 
            np.array(individual),
            max_iterations=self.two_opt_intensity * 10  # 根据强度调整迭代次数
        )
        return optimized_individual.tolist()
    
    def run_generation_optimized(self, population):
        if self.current_generation >= self.generations:
            return population, True
            
        # 转换为numpy数组进行批量计算
        pop_array = np.array(population)
        fitnesses = calculate_fitness_parallel(self.distance_matrix, pop_array)
        
        current_best_fitness = np.max(fitnesses)
        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_individual = population[np.argmax(fitnesses)].copy()
        
        self.fitness_history.append(1/self.best_fitness)
        
        # 精英选择
        elite_indices = np.argsort(fitnesses)[-self.elite_size:]
        elites = [population[i] for i in elite_indices]
        
        # 选择
        selected = self.selection(population, fitnesses)
        
        # 交叉和变异
        children = self.crossover_batch(selected[:len(selected)-self.elite_size])
        mutated_children = self.mutate_batch(children)
        
        # 创建新种群
        new_population = mutated_children + elites
        new_population = new_population[:self.population_size]
        
        # 2-opt优化 - 只在部分个体上应用，并且减少频率
        if self.use_2opt and self.current_generation % max(1, 20 - self.two_opt_intensity) == 0:
            # 根据强度调整优化的个体数
            num_to_optimize = min(3 + self.two_opt_intensity // 3, len(new_population))
            for i in range(num_to_optimize):
                new_population[i] = self.two_opt(new_population[i])
        
        self.current_generation += 1
        return new_population, self.current_generation >= self.generations

# ------------------- 可视化和UI -------------------

class TSPCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.cities = np.array([])
        self.solution = []
        self.best_distance = 0
        self.generation = 0
        self.show_city_numbers = True
        self.point_size_base = 8
        self.line_width_base = 2
        self.setFixedSize(640, 640)
        
    def update_visual_settings(self, show_numbers, point_size, line_width):
        self.show_city_numbers = show_numbers
        self.point_size_base = point_size
        self.line_width_base = line_width
        self.update()
        
    def update_solution(self, cities, solution, best_distance, generation):
        self.cities = cities
        self.solution = solution
        self.best_distance = best_distance
        self.generation = generation
        self.update()
        
    def paintEvent(self, event):
        if len(self.cities) == 0:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.white)
        
        # 根据城市数量动态调整大小
        num_cities = len(self.cities)
        point_size = max(3, self.point_size_base - num_cities // 50)
        line_width = max(1, self.line_width_base - num_cities // 100)
        
        painter.setPen(QPen(Qt.gray, 1))
        painter.drawRect(0, 0, self.width()-1, self.height()-1)
        
        x_coords = self.cities[:,0]
        y_coords = self.cities[:,1]
        margin = 40
        content_size = self.width() - 2*margin
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        data_width = x_max - x_min
        data_height = y_max - y_min
        scale_x = content_size / data_width if data_width > 0 else 1
        scale_y = content_size / data_height if data_height > 0 else 1
        scale = min(scale_x, scale_y)
        
        offset_x = margin + (content_size - data_width*scale)/2
        offset_y = margin + (content_size - data_height*scale)/2
        
        # 使用numpy向量化计算坐标变换
        def transform_all_points(cities):
            tx = offset_x + (cities[:,0] - x_min) * scale
            ty = self.height() - offset_y - (cities[:,1] - y_min) * scale
            return np.column_stack((tx, ty))
        
        transformed_cities = transform_all_points(self.cities)
        
        # 绘制路径
        if len(self.solution) > 1:
            pen = QPen(QColor(0, 100, 200), line_width)
            painter.setPen(pen)
            
            for i in range(len(self.solution)):
                start_idx = self.solution[i]
                end_idx = self.solution[(i+1) % len(self.solution)]
                x1, y1 = transformed_cities[start_idx]
                x2, y2 = transformed_cities[end_idx]
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # 绘制城市点
        painter.setPen(QPen(Qt.black, 1))
        painter.setBrush(QColor(255, 100, 100))
        
        for i, (x, y) in enumerate(transformed_cities):
            painter.drawEllipse(int(x-point_size/2), int(y-point_size/2), 
                              point_size, point_size)
            
            if self.show_city_numbers:
                painter.setPen(QPen(Qt.black, 1))
                painter.drawText(int(x+point_size/2+2), int(y+point_size/2), str(i))
                painter.setPen(QPen(Qt.black, 1))
        
        # 显示信息
        painter.setPen(QPen(Qt.black, 1))
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        info_text = f"代数: {self.generation}  最短距离: {self.best_distance:.2f}"
        painter.drawText(10, 20, info_text)

# ------------------- Worker -------------------

class TSPWorker(QThread):
    update_signal = Signal(object, object, float, int)
    finished_signal = Signal()
    
    def __init__(self, cities, population_size, mutation_rate, generations, elite_size, use_2opt, refresh_interval, two_opt_intensity):
        super().__init__()
        self.cities = cities
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.elite_size = elite_size
        self.use_2opt = use_2opt
        self.refresh_interval = refresh_interval
        self.two_opt_intensity = two_opt_intensity
        self.ga = None
        self.population = None
        self.is_running = True
        
    def run(self):
        self.ga = TSPGeneticAlgorithmOptimized(
            self.cities, 
            self.population_size, 
            self.mutation_rate, 
            self.generations,
            self.elite_size,
            self.use_2opt,
            self.two_opt_intensity
        )
        self.population = self.ga.create_population()
        while self.is_running and self.ga.current_generation < self.ga.generations:
            self.population, done = self.ga.run_generation_optimized(self.population)
            # 使用用户设置的刷新间隔
            if self.ga.current_generation % self.refresh_interval == 0 or done:
                self.update_signal.emit(
                    self.cities,
                    self.ga.best_individual,
                    1/self.ga.best_fitness if self.ga.best_fitness > 0 else float('inf'),
                    self.ga.current_generation
                )
            if done:
                break
            self.msleep(10)
        # 最终更新一次
        self.update_signal.emit(
            self.cities,
            self.ga.best_individual,
            1/self.ga.best_fitness if self.ga.best_fitness > 0 else float('inf'),
            self.ga.current_generation
        )
        self.finished_signal.emit()
        
    def stop(self):
        self.is_running = False

# ------------------- 主窗口 -------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("遗传算法解决旅行商问题 - PySide6动态演示 (优化版)")
        self.setGeometry(100, 100, 600, 600)
        self.cities = np.array([])
        self.worker = None
        self.refresh_interval = 5
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        control_panel = QGroupBox("控制面板")
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(300)
        
        # 算法参数组
        param_group = QGroupBox("算法参数")
        param_layout = QVBoxLayout()
        param_group.setLayout(param_layout)
        
        city_layout = QHBoxLayout()
        city_layout.addWidget(QLabel("城市数量(4-1000):"))
        self.city_spin = QSpinBox()
        self.city_spin.setRange(4, 1000)
        self.city_spin.setValue(200)
        city_layout.addWidget(self.city_spin)
        param_layout.addLayout(city_layout)
        
        pop_layout = QHBoxLayout()
        pop_layout.addWidget(QLabel("种群大小(10-500):"))
        self.pop_spin = QSpinBox()
        self.pop_spin.setRange(10, 500)
        self.pop_spin.setValue(100)
        pop_layout.addWidget(self.pop_spin)
        param_layout.addLayout(pop_layout)
        
        mut_layout = QHBoxLayout()
        mut_layout.addWidget(QLabel("变异率(0.001-0.2):"))
        self.mut_spin = QDoubleSpinBox()
        self.mut_spin.setDecimals(3)
        self.mut_spin.setRange(0.001, 0.2)
        self.mut_spin.setValue(0.03)
        self.mut_spin.setSingleStep(0.005)
        mut_layout.addWidget(self.mut_spin)
        param_layout.addLayout(mut_layout)
        
        gen_layout = QHBoxLayout()
        gen_layout.addWidget(QLabel("迭代次数(100-10000):"))
        self.gen_spin = QSpinBox()
        self.gen_spin.setRange(100, 10000)
        self.gen_spin.setValue(1500)
        gen_layout.addWidget(self.gen_spin)
        param_layout.addLayout(gen_layout)
        
        elite_layout = QHBoxLayout()
        elite_layout.addWidget(QLabel("精英保留(0-20):"))
        self.elite_spin = QSpinBox()
        self.elite_spin.setRange(0,20)
        self.elite_spin.setValue(5)
        elite_layout.addWidget(self.elite_spin)
        param_layout.addLayout(elite_layout)
        
        self.use_2opt_check = QCheckBox("使用2-opt局部优化")
        self.use_2opt_check.setChecked(True)
        param_layout.addWidget(self.use_2opt_check)
        
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QLabel("2-opt强度(1-10):"))
        self.intensity_spin = QSpinBox()
        self.intensity_spin.setRange(1, 10)  # 修改为1-10
        self.intensity_spin.setValue(7)
        intensity_layout.addWidget(self.intensity_spin)
        param_layout.addLayout(intensity_layout)
        
        control_layout.addWidget(param_group)
        
        # 可视化设置组
        visual_group = QGroupBox("可视化设置")
        visual_layout = QVBoxLayout()
        visual_group.setLayout(visual_layout)
        
        self.show_numbers_check = QCheckBox("显示城市数字")
        self.show_numbers_check.setChecked(True)
        self.show_numbers_check.toggled.connect(self.update_visual_settings)
        visual_layout.addWidget(self.show_numbers_check)
        
        refresh_layout = QHBoxLayout()
        refresh_layout.addWidget(QLabel("刷新间隔(代):"))
        self.refresh_spin = QSpinBox()
        self.refresh_spin.setRange(1, 100)
        self.refresh_spin.setValue(5)
        self.refresh_spin.valueChanged.connect(self.update_refresh_interval)
        refresh_layout.addWidget(self.refresh_spin)
        visual_layout.addLayout(refresh_layout)
        
        point_layout = QHBoxLayout()
        point_layout.addWidget(QLabel("基础点大小:"))
        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(3, 20)
        self.point_size_spin.setValue(8)
        self.point_size_spin.valueChanged.connect(self.update_visual_settings)
        point_layout.addWidget(self.point_size_spin)
        visual_layout.addLayout(point_layout)
        
        line_layout = QHBoxLayout()
        line_layout.addWidget(QLabel("基础线粗细:"))
        self.line_width_spin = QSpinBox()
        self.line_width_spin.setRange(1, 5)
        self.line_width_spin.setValue(2)
        self.line_width_spin.valueChanged.connect(self.update_visual_settings)
        line_layout.addWidget(self.line_width_spin)
        visual_layout.addLayout(line_layout)
        
        control_layout.addWidget(visual_group)
        
        # 功能按钮
        self.generate_btn = QPushButton("生成随机城市")
        self.generate_btn.clicked.connect(self.generate_cities)
        control_layout.addWidget(self.generate_btn)
        
        self.start_btn = QPushButton("开始求解")
        self.start_btn.clicked.connect(self.start_algorithm)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止求解")
        self.stop_btn.clicked.connect(self.stop_algorithm)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        control_layout.addWidget(QLabel("进度:"))
        control_layout.addWidget(self.progress_bar)
        
        # 日志
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(300)
        log_layout.addWidget(self.log_text)
        control_layout.addWidget(log_group)
        
        control_layout.addStretch()
        
        # 可视化区域
        vis_layout = QVBoxLayout()
        self.tsp_canvas = TSPCanvas()
        vis_layout.addWidget(self.tsp_canvas)
        
        main_layout.addWidget(control_panel)
        main_layout.addLayout(vis_layout)
        
        self.generate_cities()
        
    def update_refresh_interval(self, interval):
        self.refresh_interval = interval
        
    def update_visual_settings(self):
        show_numbers = self.show_numbers_check.isChecked()
        point_size = self.point_size_spin.value()
        line_width = self.line_width_spin.value()
        self.tsp_canvas.update_visual_settings(show_numbers, point_size, line_width)
        
    def generate_cities(self):
        num_cities = self.city_spin.value()
        self.cities = np.random.rand(num_cities, 2) * 100
        self.tsp_canvas.update_solution(self.cities, [], 0, 0)
        self.log_text.append(f"生成了 {num_cities} 个随机城市")
        
    def start_algorithm(self):
        if self.worker and self.worker.isRunning():
            return
            
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.generate_btn.setEnabled(False)
        
        population_size = self.pop_spin.value()
        mutation_rate = self.mut_spin.value()
        generations = self.gen_spin.value()
        elite_size = self.elite_spin.value()
        use_2opt = self.use_2opt_check.isChecked()
        two_opt_intensity = self.intensity_spin.value()
        
        self.progress_bar.setRange(0, generations)
        self.progress_bar.setValue(0)
        
        self.worker = TSPWorker(
            self.cities,
            population_size,
            mutation_rate,
            generations,
            elite_size,
            use_2opt,
            self.refresh_interval,
            two_opt_intensity
        )
        self.worker.update_signal.connect(self.on_algorithm_update)
        self.worker.finished_signal.connect(self.on_algorithm_finished)
        
        self.log_text.append("开始遗传算法求解...")
        if use_2opt:
            self.log_text.append(f"增量2-opt局部优化已启用 (强度: {two_opt_intensity})")
            if two_opt_intensity <= 3:
                self.log_text.append("强度等级: 轻度优化 (快速)")
            elif two_opt_intensity <= 6:
                self.log_text.append("强度等级: 中度优化 (平衡)")
            else:
                self.log_text.append("强度等级: 深度优化 (高质量)")
        self.log_text.append(f"精英保留数量: {elite_size}")
        self.log_text.append(f"刷新间隔: 每 {self.refresh_interval} 代")
        if NUMBA_AVAILABLE:
            self.log_text.append("Numba加速已启用")
        else:
            self.log_text.append("Numba不可用，使用纯Python版本")
            
        self.worker.start()
        
    def stop_algorithm(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            
    def on_algorithm_update(self, cities, solution, distance, generation):
        self.tsp_canvas.update_solution(cities, solution, distance, generation)
        self.progress_bar.setValue(generation)
        
    def on_algorithm_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.generate_btn.setEnabled(True)
        best_distance = self.tsp_canvas.best_distance
        self.log_text.append(f"算法完成! 最短距离: {best_distance:.2f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())