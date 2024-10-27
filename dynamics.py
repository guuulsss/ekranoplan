import numpy as np

def dynamics(t, state, d, Fthrust, H_func):
    u, v, w, p, q, r, phi, psi, theta, x, y_pos, z = state

    theta = np.clip(theta, -np.pi/2 + 0.01, np.pi/2 - 0.01)
    phi = np.clip(phi, -np.pi/2 + 0.01, np.pi/2 - 0.01)

    vel2 = u ** 2 + v ** 2 + w ** 2
    m = 11
    m2 = 5

    H = H_func(z)

    epsilon = 0.1
    k = 1.0
    Cl = (0.4 + 50 * theta) * (1 + k / (H + epsilon))
    Cd = (0.07 + 7 * theta) * (1 + k / (H + epsilon))

    g = 9.8
    Ixx = 8.93
    Iyy = 11.24
    Izz = 18.9

    D = Cd * vel2
    L = Cl * vel2

    Xg = m * g * np.sin(theta)
    Yg = m * g * np.sin(phi) * np.cos(theta)
    Zg = m * g * np.cos(phi) * np.cos(theta)

    X = Fthrust * np.cos(theta) - D * np.cos(theta) - Xg
    Y = Yg
    Z = -L * np.cos(theta) - Fthrust * np.sin(theta) + Zg

    du = X / m + r * v - q * w
    dv = Y / m - r * u + p * w
    dw = Z / m - p * v + q * u

    My = 0.71 * (Cl + 5 * d) * vel2 - m2 * g * 1.5
    Mx = 0
    Mz = 0

    dp = (q * r * (Iyy - Izz)) / Ixx + Mx / Ixx
    dq = (p * r * (Izz - Ixx)) / Iyy + My / Iyy
    dr = (p * q * (Ixx - Iyy)) / Izz + Mz / Izz

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    tan_theta = np.tan(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    cos_theta = np.clip(cos_theta, 0.01, None)

    dphi = p + q * sin_phi * tan_theta + r * cos_phi * tan_theta
    dpsi = q * sin_phi / cos_theta + r * cos_phi / cos_theta
    dtheta = q * cos_phi - r * sin_phi

    dx = (u * cos_theta * np.cos(psi) +
          v * (sin_phi * sin_theta * np.cos(psi) - cos_phi * np.sin(psi)) +
          w * (cos_phi * sin_theta * np.cos(psi) + sin_phi * np.sin(psi)))
    dy = (u * cos_theta * np.sin(psi) +
          v * (sin_phi * sin_theta * np.sin(psi) + cos_phi * np.cos(psi)) +
          w * (cos_phi * sin_theta * np.sin(psi) - sin_phi * np.cos(psi)))
    dz = -u * sin_theta + v * sin_phi * cos_theta + w * cos_phi * cos_theta

    return [du, dv, dw, dp, dq, dr, dphi, dpsi, dtheta, dx, dy, dz]
