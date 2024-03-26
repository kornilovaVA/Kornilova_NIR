import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tkinter import PhotoImage
import sys
import os

# pyinstaller --onefile --icon=ico.ico --add-data="ico.png;." Kornilova_NIR.py

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")
    
window1_open = False
window2_open = False
additional_windows = []
file_path = None
    
def read_matrix_and_vector(file_path):
    with open(file_path, 'r') as file:
        lines = file.read().strip().split('\n\n')
        A = np.array([list(map(int, row.split())) for row in lines[0].split('\n')])
        y0 = np.array(list(map(int, lines[1].split())))
    return A, y0

def MSE(y, y0):
    return np.mean(np.square(y - y0))

def gradientMSE(AT, dy):
    return (AT @ dy) * (2. / dy.size)

def Algo1(A, y0, xmin=0, step=1, K=1000000):
    if A is None or y0 is None:
        raise ValueError("Неверный аргумент!")
    if A.shape[0] != y0.size:
        raise ValueError("Неверный размер вектора!")
    if np.any(A < 0) or np.any(A > 1) or np.any(y0 < 0):
        raise ValueError("Неверный диапазон значений!")
    if step <= 0:
        raise ValueError("Неверное значение шага!")
    if xmin < 0:
        raise ValueError("Неверное значение xmin!")

    n, m = A.shape
    x = np.maximum(np.max(A * y0[:, None], axis=0), xmin)
    AT = A.T
    error = []

    for k in range(K):
        y = A @ x
        curGrad = gradientMSE(AT, y-y0)
        error.append(MSE(y, y0))
        indW, maxDGrad = -1, -1
        for i in range(m):
            if x[i] - step >= xmin:
                x[i] -= step
                tmp = A @ x - y0
                if np.all(tmp >= 0):
                    tmpDGrad = np.linalg.norm(gradientMSE(AT, tmp) - curGrad, 1)
                    if tmpDGrad > maxDGrad:
                        indW, maxDGrad = i, tmpDGrad
                x[i] += step

        if indW == -1:
            return x, error, k
        x[indW] -= step

    return x, error, k

def Algo2(A, y0, xmin=0, K=1000000):
    if A is None or y0 is None:
        raise ValueError("Неверный аргумент!")
    if A.shape[0] != y0.size:
        raise ValueError("Неверный размер вектора!")
    if np.any(A < 0) or np.any(A > 1) or np.any(y0 < 0):
        raise ValueError("Неверный диапазон значений!")
    if xmin < 0:
        raise ValueError("Неверное значение xmin!")

    n, m = A.shape
    x = np.maximum(np.max(A * y0[:, None], axis=0), xmin)
    AT = A.T
    error = []

    for k in range(K):
        y = A @ x
        curGrad = gradientMSE(AT, (y - y0))
        error.append(MSE(y, y0))
        indW, maxDGrad, stepW = -1, -1, 0
        for i in range(m):
            yTmp = np.where(A[:, i] != 0, y, np.inf)
            step = min(x[i] - xmin, np.min(yTmp - y0))
            if step > 0:
                x[i] -= step
                tmp = A @ x - y0
                if np.all(tmp >= 0):
                    tmpDGrad = np.linalg.norm(gradientMSE(AT, tmp) - curGrad, 1)
                    if tmpDGrad > maxDGrad:
                        indW, maxDGrad, stepW = i, tmpDGrad, step
                x[i] += step
        if indW == -1:
            return x, error, k
        x[indW] -= stepW
    return x, error, k

def Algo3(A, y0, xmin=0, step=1, K=1000000):
    if A is None or y0 is None:
        raise ValueError("Неверный аргумент!")
    if A.shape[0] != y0.size:
        raise ValueError("Неверный размер вектора!")
    if np.any(A < 0) or np.any(A > 1) or np.any(y0 < 0):
        raise ValueError("Неверный диапазон значений!")
    if step <= 0:
        raise ValueError("Неверное значение шага!")
    if xmin < 0:
        raise ValueError("Неверное значение xmin!")

    n, m = A.shape
    x = np.maximum(np.max(A * y0[:, None], axis=0), xmin)
    error = []

    for k in range(K):
        y = A @ x
        error.append(MSE(y, y0))
        indW, sumW = -1, -1
        for i in range(m):
            if x[i] - step >= xmin:
                x[i] -= step
                tmp = A @ x - y0
                if np.all(tmp >= 0):
                    tmp1 = np.count_nonzero(A[:, i])
                    if tmp1>sumW:
                      indW, sumW = i, tmp1
                x[i] += step

        if indW == -1:
            return x, error, k
        x[indW] -= step

    return x, error, k

def plot_needs(A, x):
    global additional_windows
    plt.figure("Построение")
    k = 4
    b = sum(x)
    p1 = int(b * 0.3) + 1
    p2 = int(b * 0.1) + 1
    
    width_cells = b
    length_cells = A.shape[0] * p1 + (A.shape[0] + 1) * p2

    for i in range(0, (width_cells + 1) * k, k):
        plt.axvline(x=i, color='black', linewidth=0.5)
    for j in range(0, (length_cells + 1) * k, k):
        plt.axhline(y=j, color='black', linewidth=0.5)

    for i in range(A.shape[0]):
        tmp = A[i] * x
        ind = np.nonzero(tmp)[0][0]

        start_x = sum(x[:ind]) * k
        start_y = (i * p1 + (i + 1) * p2) * k
        width = sum(tmp) * k
        height = p1 * k

        block = patches.Rectangle((start_x, start_y), width, height, linewidth=1, edgecolor='blue', facecolor='blue')
        plt.gca().add_patch(block)
        
    plt.xlim(0, width_cells * 4 + 1)
    plt.ylim(0, length_cells * 4 + 1)
    plt.gca().set_aspect('equal')

    plt.axis('off')
    graph_window = plt.get_current_fig_manager().window
    additional_windows.append(graph_window)
    plt.show()

def plot_mse(error):
    global additional_windows
    iterations = range(1, len(error) + 1)
    plt.figure("График MSE")
    plt.plot(iterations, error, marker='o', color='black', label='Алгоритм 2', markersize=3)
    plt.xlabel('x - номер итерации [шт.]')
    plt.ylabel('y - среднеквадратичная ошибка [ед.]')
    plt.title('График зависимости mse от номера итерации')
    graph_window = plt.get_current_fig_manager().window
    additional_windows.append(graph_window)
    plt.show()
    
def load_data():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("TXT files", "*.txt")])
    if file_path:
        load_button.pack_forget()
        calculate_button.pack(pady=10)
        algo_combo.pack(pady=10)
        back_button.pack(pady=10)
        
def go_back():
    global additional_windows
    for window in additional_windows:  
        try:
            if window.winfo_exists():  
                window.destroy()  
        except tk.TclError:  
            pass  
    additional_windows.clear()  
    for widget in root.winfo_children():
        widget.pack_forget()
    load_button.pack(pady=10)

def algo1():
    global window1_open, window2_open
    
    if file_path:
        A, y0 = read_matrix_and_vector(file_path)
        x, error, _ = Algo1(A, y0)
        
        def task1():
            global window1_open
            if not window1_open:
                window1_open = True
                plot_needs(A, x)
                window1_open = False
        
        def task2():
            global window2_open
            if not window2_open:
                window2_open = True
                plot_mse(error)
                window2_open = False
        
        algo_label = ttk.Label(root, text=f"Выбранный алгоритм: {selected_algo.get()}")
        algo_label.pack(pady=5)
        
        btn_1 = ttk.Button(root, text="Показать построение", command=task1)
        btn_1.pack(pady=5)
        
        btn_2 = ttk.Button(root, text="Показать график MSE", command=task2)
        btn_2.pack(pady=5)

def algo2():
    global window1_open, window2_open
    if file_path:
        A, y0 = read_matrix_and_vector(file_path)
        x, error, _ = Algo2(A, y0)
        
        def task1():
            global window1_open
            if not window1_open:
                window1_open = True
                plot_needs(A, x)
                window1_open = False
        
        def task2():
            global window2_open
            if not window2_open:
                window2_open = True
                plot_mse(error)
                window2_open = False
        
        algo_label = ttk.Label(root, text=f"Выбранный алгоритм: {selected_algo.get()}")
        algo_label.pack(pady=5)
        
        btn_1 = ttk.Button(root, text="Показать построение", command=task1)
        btn_1.pack(pady=5)
        
        btn_2 = ttk.Button(root, text="Показать график MSE", command=task2)
        btn_2.pack(pady=5)
    
def algo3():
    global window1_open, window2_open
    
    if file_path:
        A, y0 = read_matrix_and_vector(file_path)
        x, error, _ = Algo3(A, y0)
        
        def task1():
            global window1_open
            if not window1_open:
                window1_open = True
                plot_needs(A, x)
                window1_open = False
        
        def task2():
            global window2_open
            if not window2_open:
                window2_open = True
                plot_mse(error)
                window2_open = False
        
        algo_label = ttk.Label(root, text=f"Выбранный алгоритм: {selected_algo.get()}")
        algo_label.pack(pady=5)
        
        btn_1 = ttk.Button(root, text="Показать построение", command=task1)
        btn_1.pack(pady=5)
        
        btn_2 = ttk.Button(root, text="Показать график MSE", command=task2)
        btn_2.pack(pady=5)

def calculate():
    for widget in root.winfo_children():
        if widget not in [algo_combo, calculate_button, back_button]:
            widget.pack_forget()
            
    selected_index = algo_combo.current()  # Получение индекса выбранного алгоритма
    if selected_index == 0:
        algo1()
    elif selected_index == 1:
        algo2()
    elif selected_index == 2:
        algo3()

root = tk.Tk()
root.title("Выбор алгоритма")
size = 200  
root.geometry(f'{size*2}x{size*2}')
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
center_x = int(screen_width / 2 - size)
center_y = int(screen_height / 2 - size)
root.geometry(f'+{center_x}+{center_y}')
root.resizable(False, False)
root.configure(bg='white')
icon_path = os.path.join(base_path, 'ico.png')
icon = PhotoImage(file=icon_path)
root.iconphoto(False, icon)

selected_algo = tk.StringVar()

algo_combo = ttk.Combobox(root, textvariable=selected_algo, state='readonly')
algo_combo['values'] = ('Алгоритм 1', 'Алгоритм 2', 'Алгоритм 3')
algo_combo.current(0) 
calculate_button = ttk.Button(root, text="Считать", command=calculate)
load_button = ttk.Button(root, text="Загрузить данные", command=load_data)
back_button = ttk.Button(root, text="Вернуться назад", command=go_back)

load_button.pack(pady=10)
root.mainloop()