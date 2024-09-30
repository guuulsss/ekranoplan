import numpy as np

def dynamics(input_vals):
    u, v, w, p, q, r, phi, psi, teta, d, Fthrust, H = input_vals
    vel2 = u ** 2 + v ** 2 + w ** 2
    m = 11
    m2 = 5

    Cl = 50 * teta + 0.4 + np.sqrt(H) * 0.001
    Cd = 7 * teta + 0.07

    g = 9.8
    Ixx = 8.93
    Iyy = 11.24
    Izz = 18.9

    D = Cd * vel2
    L = Cl * vel2

    Xg = m * g * np.sin(teta)
    Yg = m * g * np.sin(phi) * np.cos(teta)
    Zg = m * g * np.cos(phi) * np.cos(teta)

    X = Fthrust * np.cos(teta) - D * np.cos(teta) - Xg
    Y = Yg
    Z = -L * np.cos(teta) - Fthrust * np.sin(teta) + Zg

    du = X / m + r * v - q * w
    dv = Y / m - r * u + p * w
    dw = Z / m - p * v + q * u

    My = 0.71 * (Cl + 5 * d) * vel2 - m2 * g * 1.5
    Mx = 0
    Mz = 0

    dp = (q * r * (Iyy - Izz)) / Ixx + Mx / Ixx
    dq = (p * r * (Izz - Ixx)) / Iyy + My / Iyy
    dr = (p * q * (Ixx - Iyy)) / Izz + Mz / Izz

    return np.array([du, dv, dw, dp, dq, dr])

def kinematics(input_vals):
    u, v, w, p, q, r, teta, phi, psi = input_vals

    dteta = q * np.cos(phi) - r * np.sin(phi)
    dphi = p + q * np.sin(phi) * np.tan(teta) + r * np.cos(phi) * np.tan(teta)
    dpsi = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(teta)

    dxe = u * np.cos(psi) * np.cos(teta) + v * (np.cos(psi) * np.sin(teta) * np.sin(phi)
                                                - np.sin(psi) * np.cos(phi)) + w * (
                  np.cos(psi) * np.sin(teta) * np.cos(phi) + np.sin(psi) * np.sin(phi))

    dye = u * np.sin(psi) * np.cos(teta) + v * (np.sin(psi) * np.sin(teta) * np.sin(phi)) \
          + w * (np.cos(psi) * np.sin(teta) * np.cos(phi) + np.sin(psi) * np.sin(phi))

    dze = -u * np.sin(teta) + v * np.cos(teta) * np.sin(phi) + w * np.cos(teta) * np.cos(phi)

    return np.array([dteta, dphi, dpsi, dxe, dye, dze])
