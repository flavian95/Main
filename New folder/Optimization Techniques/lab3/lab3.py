import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox

class SVDVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SVD Geometric Interpretation")
        
        # Control Panel
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        tk.Label(control_frame, text="Dimension (n):").pack(side=tk.LEFT)
        self.dim_var = tk.StringVar(value="2")
        tk.Radiobutton(control_frame, text="2D", variable=self.dim_var, value="2", command=self.update_inputs).pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="3D", variable=self.dim_var, value="3", command=self.update_inputs).pack(side=tk.LEFT)
        
        tk.Label(control_frame, text="  Matrix A (comma-separated rows, semicolon-separated cols):").pack(side=tk.LEFT)
        self.matrix_entry = tk.Entry(control_frame, width=30)
        self.matrix_entry.insert(0, "2, 1; -1, 1") # Default 2D matrix
        self.matrix_entry.pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="Visualize SVD", command=self.plot_svd).pack(side=tk.LEFT)

        # Plotting Area
        self.fig = plt.Figure(figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def update_inputs(self):
        if self.dim_var.get() == "2":
            self.matrix_entry.delete(0, tk.END)
            self.matrix_entry.insert(0, "2, 1; -1, 1")
        else:
            self.matrix_entry.delete(0, tk.END)
            self.matrix_entry.insert(0, "2, 0, 0; 0, 1.5, 0; 0, 0, 1")

    def parse_matrix(self):
        try:
            n = int(self.dim_var.get())
            matrix_str = self.matrix_entry.get()
            rows = matrix_str.split(';')
            A = np.array([[float(num) for num in row.split(',')] for row in rows])
            
            if A.shape != (n, n):
                raise ValueError(f"Matrix must be {n}x{n}")
            if np.linalg.matrix_rank(A) < n:
                messagebox.showwarning("Warning", "Matrix is not invertible. Ellipsoid will be flat.")
            return A
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid matrix format.\n{str(e)}")
            return None

    def plot_svd(self):
        A = self.parse_matrix()
        if A is None: return
        
        self.fig.clf()
        n = int(self.dim_var.get())
        U, S, Vt = np.linalg.svd(A)

        if n == 2:
            self.plot_2d(A, U, S, Vt)
        else:
            self.plot_3d(A, U, S, Vt)
            
        self.canvas.draw()

    def plot_2d(self, A, U, S, Vt):
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        
        # Generate unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle = np.array([np.cos(theta), np.sin(theta)])
        
        # Transform circle
        ellipse = A @ circle
        
        # Original Circle
        ax1.plot(circle[0, :], circle[1, :], 'b-')
        ax1.quiver(0, 0, Vt[0,0], Vt[0,1], color='r', scale=1, scale_units='xy', angles='xy', label='v1')
        ax1.quiver(0, 0, Vt[1,0], Vt[1,1], color='g', scale=1, scale_units='xy', angles='xy', label='v2')
        ax1.set_xlim(-2, 2); ax1.set_ylim(-2, 2)
        ax1.set_aspect('equal'); ax1.grid(True)
        ax1.set_title("Unit Ball (Domain)\nRight Singular Vectors (V)")
        ax1.legend()

        # Transformed Ellipse
        ax2.plot(ellipse[0, :], ellipse[1, :], 'b-')
        # Plot principal axes (Sigma * U)
        ax2.quiver(0, 0, S[0]*U[0,0], S[0]*U[1,0], color='r', scale=1, scale_units='xy', angles='xy', label='σ1 * u1')
        ax2.quiver(0, 0, S[1]*U[0,1], S[1]*U[1,1], color='g', scale=1, scale_units='xy', angles='xy', label='σ2 * u2')
        
        limit = np.max(S) * 1.5
        ax2.set_xlim(-limit, limit); ax2.set_ylim(-limit, limit)
        ax2.set_aspect('equal'); ax2.grid(True)
        ax2.set_title("Rotated Ellipse (Codomain)\nLeft Singular Vectors (U)")
        ax2.legend()

    def plot_3d(self, A, U, S, Vt):
        ax1 = self.fig.add_subplot(121, projection='3d')
        ax2 = self.fig.add_subplot(122, projection='3d')
        
        # Generate unit sphere
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 40)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Flat coordinates for transformation
        sphere_coords = np.vstack((x.flatten(), y.flatten(), z.flatten()))
        ellipsoid_coords = A @ sphere_coords
        
        xe = ellipsoid_coords[0, :].reshape(40, 40)
        ye = ellipsoid_coords[1, :].reshape(40, 40)
        ze = ellipsoid_coords[2, :].reshape(40, 40)
        
        ax1.plot_surface(x, y, z, color='b', alpha=0.3)
        ax2.plot_surface(xe, ye, ze, color='r', alpha=0.3)

        # Plot vectors (omitted for brevity, can add using ax.quiver)
        ax1.set_title("Unit Sphere")
        ax2.set_title("Transformed Ellipsoid")
        
        limit = np.max(S) * 1.5
        ax2.set_xlim([-limit, limit]); ax2.set_ylim([-limit, limit]); ax2.set_zlim([-limit, limit])

if __name__ == "__main__":
    root = tk.Tk()
    app = SVDVisualizerApp(root)
    root.mainloop()