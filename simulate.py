import numpy as np
from control import ss, step_response
import matplotlib.pyplot as plt
from system_matrices import A, B, C, D, Kh

def simulate_system():
    A_cl = A - B @ Kh  # Новая матрица состояния для замкнутой системы

    closed_loop_system = ss(A_cl, B, C, D)

    time = np.linspace(0, 10, 1000)
    t, y = step_response(closed_loop_system, time)

    return t, y

def plot_response(t, y):
    plt.figure(figsize=(10, 8))
    num_outputs = y.shape[0]
    num_inputs = y.shape[1]

    for i in range(num_outputs):
        for j in range(num_inputs):
            plt.plot(t, y[i, j, :], label=f'Output {i + 1}, Input {j + 1}')

    plt.xlabel('Time [s]')
    plt.ylabel('Response')
    plt.legend()
    plt.title('Step Response of the Closed-Loop System')
    plt.grid()
    plt.show()
