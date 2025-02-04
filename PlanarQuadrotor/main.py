import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are, logm, pinv, block_diag
import control as ctrl
import matplotlib.animation as animation

MAX_THRUST = 30

# =============================================
# 1. Nonlinear Quadrotor Dynamics (Continuous)
# =============================================
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

# =============================================
# 2. Data Collection for DMDc
# =============================================
def collect_training_data(num_trajectories=50, steps_per_traj=100, dt=0.05):
    quad = PlanarQuadrotor()
    X = []  # States [x, x_dot, z, z_dot, theta, theta_dot]
    U = []  # Controls [F1, F2]
    X_prime = []  # Next states

    for _ in range(num_trajectories):
        # Random initial condition
        x0 = np.array([
            np.random.uniform(-2, 2),  # x
            np.random.uniform(-1, 1),  # x_dot
            np.random.uniform(0, 4),  # z
            np.random.uniform(-1, 1),  # z_dot
            np.random.uniform(0, np.pi),  # theta
            np.random.uniform(-0.5, 0.5)  # theta_dot
        ])

        # Random control sequence
        f1 = np.random.normal(0, 2, steps_per_traj)
        f2 = np.random.normal(0, 2, steps_per_traj)
        F1 = np.clip(f1 + quad.m * quad.g / 2, 0, MAX_THRUST)
        F2 = np.clip(f2 + quad.m * quad.g / 2, 0, MAX_THRUST)

        # Simulate trajectory
        current_state = x0.copy()
        for k in range(steps_per_traj):
            # Record data
            X.append(current_state)
            U.append([F1[k], F2[k]])

            # Integrate nonlinear dynamics
            sol = solve_ivp(quad.dynamics, [0, dt], current_state,
                            args=(F1[k], F2[k]), t_eval=[dt])
            next_state = sol.y[:, -1]
            X_prime.append(next_state)

            current_state = next_state.copy()

    return np.array(X).T, np.array(U).T, np.array(X_prime).T


# =============================================
# 3. Define Observables (Koopman Lifting)
# =============================================
def lift_state(state, scale=None):
    x, x_dot, z, z_dot, theta, theta_dot = state
    lifted_state = np.array([
        x,
        x_dot,
        z,
        z_dot,
        theta,
        theta_dot,
        np.sin(theta),
        np.cos(theta),
        x_dot ** 2,
        z_dot ** 2,
        theta_dot ** 2,
        x_dot * np.sin(theta),
        z_dot * np.cos(theta),
        theta * theta_dot,
        # 1.0  # Constant term for affine dynamics
    ])

    # Scale to normalize states that are too big or too small
    if scale is not None:
        lifted_state = np.multiply(lifted_state, scale)

    return lifted_state

def unlift_states(koopman_state, scale=None):
    if scale is not None:
        S = np.diag(1/scale)
        unscaled_state = S@koopman_state
        unlifted_state = unscaled_state[0:6]
        return unlifted_state
    else:
        return koopman_state[0:6]

def lift_controls(states, controls):
    x, x_dot, z, z_dot, theta, theta_dot = states
    F1, F2 = controls

    return np.array([
        F1,
        F2,
        # F1*np.cos(theta),
        # F1*np.sin(theta),
        # F2*np.cos(theta),
        # F2*np.sin(theta)
    ])


# =============================================
# 4. DMDc for Koopman Operator and Control Matrix
# =============================================
def dmdc(X, U, X_prime):
    # Lift states to observable space
    Psi = np.array([lift_state(x) for x in X.T]).T
    Psi_prime = np.array([lift_state(x) for x in X_prime.T]).T

    U_lifted = []
    for idx, u in enumerate(U.T):
        lifted_u = np.array(lift_controls(X[:, idx], u))
        U_lifted.append(lifted_u)

    U_lifted = np.array(U_lifted).T

    # Construct DMDc matrices
    Omega = np.vstack([Psi, U_lifted])

    # Solve for Koopman operator and control matrix
    AB = Psi_prime @ pinv(Omega)
    A = AB[:, :Psi.shape[0]]
    B = AB[:, Psi.shape[0]:]

    # Get the scale of the states
    # This can prevent ill conditioning.
    Psi_max = np.max(Psi, axis=1)
    Psi_min = np.min(Psi, axis=1)
    scale = 1 / (Psi_max - Psi_min)

    return A, B, scale

def scale_matrices(A, B, scale):
    S = np.diag(scale)
    S_inv = np.diag(1/scale)

    A_tilde = S@A@S_inv
    B_tilde = S@B

    return A_tilde, B_tilde

# =============================================
# 5. Compare Koopman and Nonlinear Dynamics
# =============================================
def compare_koopman_vs_nonlinear(x0, A, B, scale, dt=0.05, T=10):
    # Initialize random controls for simulation
    n_steps = int(T / dt)
    states_koopman = np.zeros((6, n_steps + 1))
    states_nonlinear = np.zeros((6, n_steps + 1))
    states_koopman[:, 0] = x0  # Non-lifted initial state for Koopman dynamics
    states_nonlinear[:, 0] = x0  # Non-lifted initial state for Nonlinear dynamics
    quad = PlanarQuadrotor()

    # Loop to simulate the systems independently
    for k in range(n_steps):
        current_state_koopman = states_koopman[:, k]
        current_state_nonlinear = states_nonlinear[:, k]

        # Random control input for both Koopman and Nonlinear dynamics
        F1 = np.clip(np.random.normal(5, 2), 0, MAX_THRUST)
        F2 = np.clip(np.random.normal(5, 2), 0, MAX_THRUST)

        lifted_states = lift_state(current_state_koopman, scale)
        controls = np.array([F1, F2])
        lifted_controls = lift_controls(current_state_koopman, controls)

        # Simulate Koopman dynamics (using random controls)
        next_states_koopman = A @ lifted_states + B @ lifted_controls
        states_koopman[:, k + 1] = unlift_states(next_states_koopman, scale)

        # Simulate Nonlinear dynamics (using random controls)
        sol_nonlinear = solve_ivp(lambda t, y: quad.dynamics(t, y, F1, F2),
                                  [0, dt], current_state_nonlinear, t_eval=[dt])
        states_nonlinear[:, k + 1] = sol_nonlinear.y[:, -1]

    # Compute RMS error between the two trajectories
    rms_error = np.sqrt(np.mean((states_koopman - states_nonlinear) ** 2, axis=1))

    return states_koopman, states_nonlinear, rms_error

# =============================================
# 6. LQR Controller Design in Observable Space
# =============================================
def design_lqr(A, B, dt, Q_diag, R_diag):
    # Convert discrete-time A to continuous-time L (assuming small dt)
    Ac = logm(A) / dt
    Bc = B / dt

    # Weights (adjust these based on your system)
    Q = np.diag(Q_diag)
    R = np.diag(R_diag)

    K, _, _ = ctrl.dlqr(A, B, Q, R)
    check_controllability(A, B)

    eigvals, eigvec = np.linalg.eig(A-B@K)
    print(eigvals)

    print(f"A condition number: {np.linalg.cond(A)}")
    print(f"B condition number: {np.linalg.cond(B)}")

    return K

def check_controllability(A, B):
    n = A.shape[0]
    C = B
    for i in range(1, n):
        C = np.hstack((C, np.linalg.matrix_power(A, i).dot(B)))
    rank = np.linalg.matrix_rank(C)
    if rank == n:
        print("System is controllable")
    else:
        print("System is not controllable")

# =============================================
# 7. Closed-Loop Simulation with Koopman LQR
# =============================================
def simulate_closed_loop(dynamics, x0, K, x_des, scale, lift_state, dt=0.05, T=10):
    psi_des = lift_state(x_des, scale)
    n_steps = int(T / dt)
    states = np.zeros((x0.shape[0], n_steps + 1))
    states[:, 0] = x0
    controls = np.zeros((2, n_steps))

    for k in range(n_steps):
        # Current state and observable
        states[4, k] = states[4, k] % (2 * np.pi)
        current_state = states[:, k]
        psi = lift_state(current_state, scale)
        psi_error = psi - psi_des

        # Compute control
        u_lifted = -K @ psi_error
        F1 = np.clip(u_lifted[0], 0, MAX_THRUST) + dynamics.m * 9.81 / 2
        F2 = np.clip(u_lifted[1], 0, MAX_THRUST) + dynamics.m * 9.81 / 2
        controls[:, k] = [F1, F2]

        # Simulate nonlinear dynamics
        sol = solve_ivp(dynamics.dynamics, [0, dt], current_state,
                        args=(F1, F2), t_eval=[dt])
        states[:, k + 1] = sol.y[:, -1]

    return states, controls


# Define a function to animate the planar quadrotor's position and orientation
def animate_quadrotor(states, dt=0.05):
    """
    Animates the planar quadrotor's trajectory and orientation as a rectangle.

    Args:
        states: The states of the quadrotor [x, x_dot, z, z_dot, theta, theta_dot].
        dt: Time step for the animation (default is 0.05 seconds).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Set up the plot limits
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_title('Planar Quadrotor Simulation')

    # Create a rectangle to represent the quadrotor
    quad_width = 0.7
    quad_height = 0.3
    quad = plt.Rectangle((-quad_height / 2, -quad_width / 2), quad_height, quad_width, color="b", angle=0)
    ax.add_patch(quad)

    # Update function for the animation
    def update(frame):
        # Get the current state at time `frame`
        x = states[0, frame]
        z = states[2, frame]
        theta = states[4, frame]

        # Update the position of the quadrotor (rectangle)
        quad.set_xy([x - quad_width / 2, z - quad_height / 2])
        quad.angle = np.degrees(theta)  # Update orientation (rotation) in degrees

        return quad,

    # Create an animation
    ani = animation.FuncAnimation(fig, update, frames=len(states[0]), interval=dt * 1000, blit=True)

    plt.show()

# =============================================
# Main Workflow (Extended)
# =============================================
if __name__ == "__main__":
    np.random.seed(1)
    # Data collection parameters
    DT = 0.05
    Q_diag = [1, 1, 1, 1, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1]  # Tune these!
    R_diag = [1, 1,
              # 0.1, 0.1, 0.1, 0.1
              ]

    # 1. Collect training data
    print("Collecting training data...")
    X, U, X_prime = collect_training_data(num_trajectories=50, steps_per_traj=100, dt=DT)

    # 2. Perform DMDc
    print("Performing DMDc...")
    A, B, scale = dmdc(X, U, X_prime)

    A, B = scale_matrices(A, B, scale)

    # 3. Compare Koopman vs Nonlinear dynamics (before controller design)
    print("Comparing Koopman and Nonlinear dynamics...")
    x0_test = np.array([-1.5, 0.0, 3.5, -0.5, np.pi/2 + 0.1, 0.1])  # Perturbed initial state
    states_koopman, states_nonlinear, rms_error = compare_koopman_vs_nonlinear(x0_test, A, B, scale, dt=DT, T=10)

    # Print RMS error
    print(f"RMS error between Koopman and Nonlinear dynamics: {rms_error}")

    # Plot comparison results
    t = np.arange(0, 10 + DT, DT)
    plt.figure(figsize=(12, 8))

    # Position plot
    plt.subplot(2, 2, 1)
    plt.plot(t, states_koopman[0], label='Koopman x')
    plt.plot(t, states_nonlinear[0], label='Nonlinear x')
    plt.plot(t, states_koopman[2], label='Koopman z')
    plt.plot(t, states_nonlinear[2], label='Nonlinear z')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()

    # Velocity plot
    plt.subplot(2, 2, 2)
    plt.plot(t, states_koopman[1], label='Koopman vx')
    plt.plot(t, states_nonlinear[1], label='Nonlinear vx')
    plt.plot(t, states_koopman[3], label='Koopman vz')
    plt.plot(t, states_nonlinear[3], label='Nonlinear vz')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()

    # Angle plot
    plt.subplot(2, 2, 3)
    plt.plot(t, np.rad2deg(states_koopman[4]), label='Koopman Theta')
    plt.plot(t, np.rad2deg(states_nonlinear[4]), label='Nonlinear Theta')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta (deg)')
    plt.legend()

    # Phase portrait
    plt.subplot(2, 2, 4)
    plt.plot(states_koopman[0], states_koopman[2], label='Koopman Trajectory')
    plt.plot(states_nonlinear[0], states_nonlinear[2], label='Nonlinear Trajectory')
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 4. Design LQR controller
    print("Designing LQR controller...")
    K = design_lqr(A, B, DT, Q_diag, R_diag)

    # 5. Simulate closed-loop system (initial state far from hover)
    z_des = 2.0  # Desired height
    x_des = np.array([0, 0, z_des, 0, np.pi/2, 0])

    print("Simulating closed-loop system with LQR controller...")
    x0 = np.array([-2, -0.2, 4, 0, np.pi/2-0.3, 0])
    quaddynamics = PlanarQuadrotor()
    states, controls = simulate_closed_loop(quaddynamics, x0, K, x_des, scale, lift_state, dt=DT, T=10)

    # pmdynamics = PointMass()
    # A, B = pmdynamics.linear_dynamics()
    # sys_cont = ctrl.ss(A, B, np.eye(A.shape[0]), np.zeros(B.shape))
    # sys_disc  = ctrl.c2d(sys_cont, DT)
    # Ad = sys_disc.A
    # Bd = sys_disc.B
    # K = design_lqr(Ad, Bd, DT, np.array([10,1,10,1]), np.array([1, 1]))
    # x0 = np.array([-1, 0.5, 1, 0.7])
    # z_des = 5
    # x_des = np.array([10, 0, z_des, 0])
    # def lift(state, scale):
    #     return state
    # states, controls = simulate_closed_loop(pmdynamics, x0, K, x_des, scale, lift, dt=DT, T=10)

    # 6. Plot closed-loop performance results
    t = np.arange(0, 10 + DT, DT)
    plt.figure(figsize=(12, 8))

    # Position plot
    plt.subplot(2, 3, 1)
    plt.plot(t, states[0], label='x')
    plt.plot(t, states[2], label='z')
    plt.plot(t, z_des * np.ones_like(t), 'k--', label='Desired z')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()

    # Angle plot
    plt.subplot(2, 3, 2)
    plt.plot(t, np.rad2deg(states[4]))
    plt.xlabel('Time (s)')
    plt.ylabel('Theta (deg)')

    # Angular velocity plot
    plt.subplot(2, 3, 3)
    plt.plot(t, np.rad2deg(states[5]))
    plt.xlabel('Time (s)')
    plt.ylabel('Theta dot (deg/s)')

    # Control inputs
    plt.subplot(2, 3, 4)
    plt.plot(t[:-1], controls[0], label='F1')
    plt.plot(t[:-1], controls[1], label='F2')
    plt.xlabel('Time (s)')
    plt.ylabel('Thrust (N)')
    plt.legend()

    # Phase portrait
    plt.subplot(2, 3, 5)
    plt.plot(states[0], states[2])
    plt.scatter(0, z_des, c='r', marker='*', s=200, label='Desired')
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.legend()

    plt.tight_layout()

    animate_quadrotor(states)
    plt.show()
