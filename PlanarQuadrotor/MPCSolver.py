import cvxpy as cp

class MPCSolver:
    def __init__(self, A, B, Q, R, horizon):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.horizon = horizon

    def solve_mpc(self, x0, x_des):
        """
        Solves an MPC problem for the linear system:
            x_{k+1} = A x_k + B u_k,
        tracking the desired state x_des over a prediction/control horizon.

        The cost is given by:
          J = sum_{k=0}^{horizon-1} [ (x_k - x_des)^T Q (x_k - x_des) + u_k^T R u_k ]
              + (x_horizon - x_des)^T Q (x_horizon - x_des)
        Parameters:
        -----------
        x0 : numpy array of shape (n,)
             The initial state.
        x_des : numpy array of shape (n,)
             The desired state (tracking target).
        Returns:
        --------
        x_traj : numpy array of shape (n, horizon+1)
             The optimal state trajectory.
        u_traj : numpy array of shape (m, horizon)
             The optimal control trajectory.
        """
        n = x0.shape[0]
        m = self.B.shape[1]

        # Define optimization variables:
        # x: state trajectory (n x (horizon+1)), u: control trajectory (m x horizon)
        x = cp.Variable((n, self.horizon + 1))
        u = cp.Variable((m, self.horizon))

        # Define the cost and constraints:
        cost = 0
        constraints = []

        # Initial condition constraint
        constraints += [x[:, 0] == x0]

        # Build the cost function and system dynamics constraints over the horizon
        for k in range(self.horizon):
            # Running cost: state tracking error and control effort
            cost += cp.quad_form(x[:, k] - x_des, self.Q) + cp.quad_form(u[:, k], self.R)
            # System dynamics: x_{k+1} = A*x_k + B*u_k
            constraints += [x[:, k + 1] == self.A @ x[:, k] + self.B @ u[:, k]]

        # Terminal cost for the state tracking error at the final time step
        cost += cp.quad_form(x[:, self.horizon] - x_des, self.Q)

        # Set up and solve the optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()

        # Extract the optimal trajectories
        x_traj = x.value
        u_traj = u.value

        return x_traj, u_traj