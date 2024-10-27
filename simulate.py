import numpy as np
from scipy.integrate import solve_ivp
from dynamics import dynamics

class Simulation:
    def __init__(self, initial_state, d_func, Fthrust_func, H_func, t_span=(0, 1e9)):
        self.initial_state = initial_state
        self.d_func = d_func
        self.Fthrust_func = Fthrust_func
        self.H_func = H_func
        self.t_span = t_span
        self.current_time = 0
        self.state = initial_state.copy()
        self.t_data = []
        self.state_data = []

    def update_state(self, dt):
        t_span = (self.current_time, self.current_time + dt)
        d = self.d_func(self.current_time)
        Fthrust = self.Fthrust_func(self.current_time)
        
        try:
            sol = solve_ivp(lambda t, y: dynamics(t, y, d, Fthrust, self.H_func),
                            t_span, self.state, method='Radau', max_step=dt)
            if not sol.success:
                raise RuntimeError('Integrator failed.')
            self.current_time += dt
            self.state = sol.y[:, -1]
            self.t_data.append(self.current_time)
            self.state_data.append(self.state.copy())
            return True
        except Exception as e:
            print(f"Simulation error at time {self.current_time:.2f}: {e}")
            return False
