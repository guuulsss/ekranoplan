import sys
from PyQt5 import QtWidgets
from simulation import Simulation
from plotter import Plotter
from gui import GuiApp

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    simulation = Simulation(initial_state=[10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            d_func=lambda t: 0.1,
                            Fthrust_func=lambda t: 0,
                            H_func=lambda z: 100)
    plotter = Plotter(canvas=None, param_canvas=None)  # Инициализация с реальными объектами
    window = GuiApp(simulation, plotter)
    window.show()
    sys.exit(app.exec_())
