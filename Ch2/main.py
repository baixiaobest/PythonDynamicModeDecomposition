import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import svd, eig, inv

HEIGHT = 449
WIDTH = 199
NUM_SNAPSHOTS = 151
TRUNCATED_SVD_K = 10

def load_data(name='VORTALL'):
    file_path = '../../DATA/FLUIDS/CYLINDER_ALL.mat'

    # Load the .mat file
    vorticity_data = scipy.io.loadmat(file_path)[name]

    return vorticity_data

def animate_data():
    vorticity_data = load_data('UALL')
    vorticity_data = np.reshape(vorticity_data, (HEIGHT, WIDTH, NUM_SNAPSHOTS))

    ani_obj = AnimateVorticity(vorticity_data, 'Vorticity', interval=100)
    ani_obj.animate()

    # Show the animation
    plt.show()

def DMD(X, X1, k=TRUNCATED_SVD_K):
    U, S, Vh = svd(X, full_matrices=False)
    U = U[:, 0:TRUNCATED_SVD_K]
    S = S[0:TRUNCATED_SVD_K]
    Vh = Vh[0:TRUNCATED_SVD_K, :]
    V = Vh.T
    S_inv = 1.0 / S
    S_inv = np.diag(S_inv)

    AU = X1 @ V @ S_inv
    A_hat = U.T @ AU
    A_hat_eigvals, A_hat_eigvec = np.linalg.eig(A_hat)

    return U, np.diag(S), S_inv, V, A_hat, A_hat_eigvals, A_hat_eigvec

def plot_DMD():
    vorticity_data = load_data('VORTALL')
    X = vorticity_data[:, 0:NUM_SNAPSHOTS-1]
    X1 = vorticity_data[:, 1:NUM_SNAPSHOTS]
    U, S, S_inv, V, A_hat, A_hat_eigvals, A_hat_eigvec = DMD(X, X1, TRUNCATED_SVD_K)

    AU = X1 @ V @ S_inv

    Φ = AU @ A_hat_eigvec
    Φ_modes = np.reshape(Φ, (HEIGHT, WIDTH, TRUNCATED_SVD_K))

    for i in range(TRUNCATED_SVD_K):
        fig, ax = plt.subplots(2, 1, figsize=(4, 9))
        img1 = ax[0].imshow(np.real(Φ_modes[:, :, i]), cmap='bwr', aspect='auto')
        img2 = ax[1].imshow(np.imag(Φ_modes[:, :, i]), cmap='bwr', aspect='auto')
        colorbar = plt.colorbar(img1, ax=ax, label='Vorticity Mode (real)')
        colorbar2 = plt.colorbar(img2, ax=ax, label='Vorticity Mode (imaginary)')
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Y')

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(np.real(A_hat_eigvals), np.imag(A_hat_eigvals))
    plt.show()

def predict():
    vorticity_data = load_data('UALL')
    X = vorticity_data[:, 0:NUM_SNAPSHOTS - 1]
    X1 = vorticity_data[:, 1:NUM_SNAPSHOTS]
    U, S, S_inv, V, A_hat, A_hat_eigvals, A_hat_eigvec = DMD(X, X1, TRUNCATED_SVD_K)

    xk = X[:, 0]

    UW_eig = U@A_hat_eigvec@np.diag(A_hat_eigvals)
    UW_inv = inv(A_hat_eigvec)@U.T

    X_pred = np.zeros((xk.shape[0], NUM_SNAPSHOTS-1))
    for i in range(NUM_SNAPSHOTS-1):
        xk = UW_eig @ (UW_inv @ xk)
        X_pred[:, i] = xk

    X_diff = np.abs(X_pred - X1)
    vorticity_data = np.reshape(X1, (HEIGHT, WIDTH, NUM_SNAPSHOTS-1))
    predicted_vorticity_data = np.reshape(X_pred, (HEIGHT, WIDTH, NUM_SNAPSHOTS-1))
    X_diff = np.reshape(X_diff, (HEIGHT, WIDTH, NUM_SNAPSHOTS-1))

    ani_obj = AnimateVorticity(predicted_vorticity_data, 'Predicted Vorticity', interval=100)
    ani_obj.animate()

    ani_obj2 = AnimateVorticity(vorticity_data, 'Actual Vorticity', interval=100)
    ani_obj2.animate()

    ani_obj3 = AnimateVorticity(X_diff, 'Vorticity difference', interval=100)
    ani_obj3.animate()

    # Show the animation
    plt.show()

class AnimateVorticity:
    def __init__(self, vorticity_data, name, interval=100):
        self.vorticity_data = vorticity_data
        self.name = name
        self.interval= interval
        self.ani = None

    def animate(self):
        num_frame = self.vorticity_data.shape[2]
        # Create a figure and axis for plotting
        fig, ax = plt.subplots(figsize=(6, 5))

        # Set up the plot (initial snapshot to display)
        img = ax.imshow(self.vorticity_data[:, :, 0], cmap='bwr', aspect='auto')
        colorbar = plt.colorbar(img, ax=ax, label='Vorticity')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Set title for the initial snapshot
        title_text = ax.set_title(f'{self.name} 0')

        # Update function for the animation
        def update(frame):
            img.set_data(self.vorticity_data[:, :, frame])  # Update the image data
            title_text.set_text(f'{self.name} {frame + 1}')  # Update the title
            return [img, title_text]

        # Create the animation
        self.ani = FuncAnimation(fig, update, frames=range(0, num_frame), interval=self.interval, blit=False)

if __name__=='__main__':
    # plot_DMD()
    # animate_data()
    predict()