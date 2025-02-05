import numpy as np

class PlanarQuadrotor:
    def __init__(self, m=1, I=1, d=0.5, g=9.81):
        self.m = m  # Mass (kg)
        self.I = I  # Moment of inertia (kg·m²)
        self.d = d  # Rotor distance from CoM (m)
        self.g = g  # Gravity (m/s²)

    def dynamics(self, t, state, F1, F2):
        x, x_dot, z, z_dot, theta, theta_dot = state
        F_total = F1 + F2
        F_diff = F1 - F2

        # ODEs (nonlinear dynamics)
        x_ddot = (F_total * np.cos(theta)) / self.m
        z_ddot = (F_total * np.sin(theta)) / self.m - self.g
        theta_ddot = (F_diff * self.d) / self.I

        return [x_dot, x_ddot, z_dot, z_ddot, theta_dot, theta_ddot]

class PointMass:
    def __init__(self, m=1.0, g=9.81):
        self.m = m  # Mass (kg)
        self.g = g  # Gravity (m/s²)

    def dynamics(self, t, state, u1, u2):
        """
        Linear dynamics for point mass:
        - u1: Acceleration in x-direction
        - u2: Acceleration in z-direction

        Returns the derivatives of state variables.
        """
        x, x_dot, z, z_dot = state

        # Acceleration dynamics (linearized)
        x_ddot = u1 / self.m
        z_ddot = u2 / self.m - self.g

        return [x_dot, x_ddot, z_dot, z_ddot]

    def linear_dynamics(self):
        """
        Returns the linearized system matrices A and B for the point mass system.
        """
        A = np.array([[0, 1, 0, 0],  # State transition matrix (A)
                      [0, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])

        B = np.array([[0, 0],  # Control matrix (B)
                      [1 / self.m, 0],
                      [0, 0],
                      [0, 1 / self.m]])

        return A, B