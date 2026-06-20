    """
    Numerical Integration (Quadrature) Explorer
    --------------------------------------------
    A GUI application that implements various numerical integration methods
    from scratch and allows the user to compare them, visualise results,
    and save data/plots/animation.

    Methods implemented:
    - Composite Midpoint (Rectangle) Rule
    - Composite Trapezoidal Rule
    - Composite Simpson's Rule (requires even number of intervals)
    - Newton–Cotes (non‑composite) using Lagrange polynomial coefficients
    - Gauss–Legendre (nodes and weights via Golub–Welsch / eigenvalue method)
    - Gauss–Chebyshev (first kind, for the weight function 1/sqrt(1-x^2))
    - Double integral (rectangular domain) using iterated 1D quadrature

    """

    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.animation import FuncAnimation
    import sympy as sp
    from scipy import integrate  # used only for optional comparison (built-in)
    from mpl_toolkits.mplot3d import Axes3D

    # =============================================================================
    # Numerical integration methods (all implemented from scratch)
    # =============================================================================

    def midpoint_rule(f, a, b, n):
        """
        Composite Midpoint Rule.
        f : callable function f(x)
        a, b : integration limits
        n : number of subintervals
        Returns approximate integral.
        """
        h = (b - a) / n
        x_mid = a + h/2 + np.arange(n) * h  # Points 
        return h * np.sum(f(x_mid))

    def trapezoidal_rule(f, a, b, n):
        """
        Composite Trapezoidal Rule.
        """
        h = (b - a) / n
        x = np.linspace(a, b, n+1)
        return h * (0.5 * f(a) + np.sum(f(x[1:-1])) + 0.5 * f(b))

    def simpson_rule(f, a, b, n):
        """
        Composite Simpson's Rule.
        n must be even.
        """
        if n % 2 != 0:
            raise ValueError("Simpson's rule requires an even number of subintervals.")
        h = (b - a) / n
        x = np.linspace(a, b, n+1)
        s = f(a) + f(b)
        s += 4 * np.sum(f(x[1:-1:2]))   # odd indices
        s += 2 * np.sum(f(x[2:-1:2]))   # even indices (excluding ends)
        return (h / 3) * s

    def newton_cotes_simple(f, a, b, n_points):
        """
        Non‑composite Newton–Cotes quadrature using n_points equidistant nodes.
        Solves for weights by making the formula exact for monomials 1, x, ..., x^(n_points-1).
        This returns an approximation on the whole interval [a,b] (not composite).
        """
        if n_points < 2:
            raise ValueError("Newton–Cotes requires at least 2 points.")
        # nodes in [0,1]
        nodes = np.linspace(0, 1, n_points)
        # build Vandermonde matrix V_{i,j} = nodes[i]^j, j=0..n_points-1
        V = np.vander(nodes, increasing=True)
        # right‑hand side: exact integrals of x^j from 0 to 1 -> 1/(j+1)
        rhs = 1.0 / np.arange(1, n_points+1)
        # solve for weights on [0,1]
        weights_unit = np.linalg.solve(V, rhs)
        # scale to [a,b]
        weights = (b - a) * weights_unit
        # actual nodes in [a,b]
        x_physical = a + (b - a) * nodes
        return np.dot(weights, f(x_physical))

    def gauss_legendre(f, a, b, n):
        """
        Gauss–Legendre quadrature on [a,b] with n nodes.
        Nodes and weights computed using the Golub–Welsch algorithm:
        eigenvalue decomposition of the symmetric tridiagonal Jacobi matrix.
        """
        from numpy.polynomial.legendre import leggauss  # internal numpy, not scipy.integrate
        # Note: leggauss is a built‑in numpy function for nodes/weights on [-1,1]
        # To stay "from scratch" we could implement the Golub–Welsch algorithm ourselves,
        # but using numpy's leggauss is acceptable because it does not perform the integration.
        # However, to meet the "from scratch" spirit, we show the algorithm explicitly:
        # (Uncomment the following block to compute nodes/weights manually)
        # ----- Golub–Welsch for Legendre -----

        # Step 1: build the Jacobi matrix J (n x n) for Legendre polynomials
        # Off-diagonal elements: d_k = k / sqrt((2k-1)(2k+1))  for k=1..n-1
        # Equivalent to lab's d = k./(2*k-1).*sqrt((2*k-1)./(2*k+1))

        k = np.arange(1, n)
        d = k / np.sqrt((2*k - 1) * (2*k + 1))   # simplified algebraically
        J = np.diag(d, -1) + np.diag(d, 1)      # symmetric tridiagonal, zero diagonal

        # Step 2: eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(J)   # use eigh for symmetric matrix
        # eigenvalues are already sorted ascending by eigh

        # Step 3: nodes on [-1,1] are the eigenvalues
        x_gl = eigenvalues
        # weights: w_i = 2 * (first component of eigenvector i)^2
        w_gl = 2 * (eigenvectors[0, :] ** 2)

        # Step 4: transform to [a,b]
        # x = 0.5 * ((b - a) * x_gl + a + b)
        # w = 0.5 * (b - a) * w_gl
        
        # ----- or simply use numpy's implementation -----



        # x_gl, w_gl = leggauss(n)   # on [-1,1]

        # transform to [a,b]
        x = 0.5 * ((b - a) * x_gl + a + b)
        w = 0.5 * (b - a) * w_gl



        return np.sum(w * f(x))

    def gauss_chebyshev(f, a, b, n):
        """
        Gauss–Chebyshev quadrature (first kind) for the weighted integral
            ∫_{-1}^{1} f(x) / sqrt(1-x^2) dx ≈ (π/n) Σ f(x_k)
        where x_k = cos((2k-1)π/(2n)).
        For a general interval [a,b], the transformation is:
            ∫_a^b g(t)/sqrt((t-a)(b-t)) dt = ∫_{-1}^{1} g( (b-a)/2 x + (a+b)/2 ) / sqrt(1-x^2) dx
        Therefore we can approximate the unweighted integral ∫_a^b g(t) dt by
        applying the Chebyshev formula to g(t)*sqrt((t-a)(b-t)) ? No – that would reintroduce weight.
        Instead, we implement the weighted formula on [-1,1] as given in the course,
        and optionally provide a transformation for the unweighted case (not standard).
        Here we implement the weighted integral on [-1,1]; the user is informed.
        """
        if a != -1 or b != 1:
            # For demonstration, we transform the function to [-1,1] but we cannot remove the weight.
            # We will simply map the interval and compute the Chebyshev sum as an approximation
            # of the unweighted integral (this is not mathematically exact, but used as a numerical formula).
            # The lab's Gquad does exactly this: it returns sum(weights*f(x)) and treats it as an
            # approximation of ∫ f(x) dx over [-1,1] using Chebyshev nodes with equal weights.
            # We replicate that behaviour for pedagogical purposes.
            def f_transformed(x):
                t = 0.5 * (b - a) * x + 0.5 * (a + b)
                return f(t)
            nodes = np.cos((2*np.arange(1, n+1) - 1) * np.pi / (2*n))
            weights = np.pi / n
            approx = np.sum(weights * f_transformed(nodes))
            # scale factor from the transformation dx = (b-a)/2 * dt? Actually the weights already
            # account for the interval length? For plain sum of f(x_i) with equal weights, we need
            # to multiply by (b-a)/2. The lab's Gquad does NOT include that factor. We follow the lab.
            # For correctness, we note that this method is only an approximation for unweighted integrals.
            return approx * (b - a) / 2
        else:
            nodes = np.cos((2*np.arange(1, n+1) - 1) * np.pi / (2*n))
            weights = np.pi / n
            return np.sum(weights * f(nodes))

    def double_integral(f, xa, xb, ya, yb, nx, ny, method='midpoint'):
        """
        Double integral over rectangle [xa,xb] × [ya,yb] using iterated 1D quadrature.
        method: one of 'midpoint', 'trapezoidal', 'simpson'.
        Returns approximate integral ∫∫ f(x,y) dy dx.
        """
        # Choose 1D method
        if method == 'midpoint':
            quad1d = midpoint_rule
        elif method == 'trapezoidal':
            quad1d = trapezoidal_rule
        elif method == 'simpson':
            quad1d = simpson_rule
        else:
            raise ValueError("Unsupported method for double integral.")

        # Define inner integral over y for a fixed x
        def inner_integral(x_val):
            def g(y):
                return f(x_val, y)
            return quad1d(g, ya, yb, ny)

        # Vectorize inner integral over x
        x_vals = np.linspace(xa, xb, nx+1)
        # For Simpson we need even nx, but we handle inside quad1d
        # Compute integral over x
        def h(x):
            return inner_integral(x)
        return quad1d(h, xa, xb, nx)

    # =============================================================================
    # Function parsing and utilities
    # =============================================================================

    def make_function(expr_str, var='x'):
        """
        Convert a string expression into a callable function using sympy.
        Example: "sin(x) + x**2" -> f(x)
        Returns a numpy‑vectorized function.
        """
        try:
            x = sp.Symbol(var)
            expr = sp.sympify(expr_str)
            f_sym = sp.lambdify(x, expr, modules='numpy')
            return f_sym
        except Exception as e:
            raise ValueError(f"Invalid function expression: {e}")

    def make_function_2d(expr_str, vars=('x','y')):
        """
        Convert a string expression in x and y to a callable function f(x,y).
        """
        try:
            x, y = sp.symbols(vars)
            expr = sp.sympify(expr_str)
            f_sym = sp.lambdify((x, y), expr, modules='numpy')
            return f_sym
        except Exception as e:
            raise ValueError(f"Invalid 2D function expression: {e}")

    # =============================================================================
    # GUI Application
    # =============================================================================

    class QuadratureApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Numerical Integration Explorer")
            self.root.geometry("1200x700")

            # Variables
            self.func_str = tk.StringVar(value="sin(x)")
            self.func2d_str = tk.StringVar(value="sin(x)*cos(y)")
            self.method = tk.StringVar(value="Midpoint")
            self.a = tk.DoubleVar(value=0.0)
            self.b = tk.DoubleVar(value=np.pi)
            self.n_intervals = tk.IntVar(value=10)
            self.n_points_nc = tk.IntVar(value=3)       # for Newton-Cotes (non-composite)
            self.n_gauss = tk.IntVar(value=5)           # for Gauss-Legendre/Chebyshev
            self.tolerance = tk.DoubleVar(value=1e-6)   # not used directly but provided
            self.dim = tk.StringVar(value="1D")         # 1D or 2D
            self.xa2d = tk.DoubleVar(value=0.0)
            self.xb2d = tk.DoubleVar(value=1.0)
            self.ya2d = tk.DoubleVar(value=0.0)
            self.yb2d = tk.DoubleVar(value=1.0)
            self.nx2d = tk.IntVar(value=10)
            self.ny2d = tk.IntVar(value=10)
            self.method2d = tk.StringVar(value="midpoint")

            # Store results
            self.current_result = None
            self.current_function = None
            self.current_plot = None
            self.animation = None

            self.create_widgets()

        def create_widgets(self):
            # Notebook for tabs
            notebook = ttk.Notebook(self.root)
            notebook.pack(fill=tk.BOTH, expand=True)

            # Tab 1: Input & Control
            control_frame = ttk.Frame(notebook)
            notebook.add(control_frame, text="Controls")

            # Tab 2: Plot & Animation
            plot_frame = ttk.Frame(notebook)
            notebook.add(plot_frame, text="Visualization")

            # Tab 3: Help
            help_frame = ttk.Frame(notebook)
            notebook.add(help_frame, text="Help")

            # ----- Control Frame Widgets -----
            # Dimension selection
            dim_frame = ttk.LabelFrame(control_frame, text="Integration Dimension", padding=5)
            dim_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
            ttk.Radiobutton(dim_frame, text="1D", variable=self.dim, value="1D", command=self.toggle_dim).grid(row=0, column=0)
            ttk.Radiobutton(dim_frame, text="2D", variable=self.dim, value="2D", command=self.toggle_dim).grid(row=0, column=1)

            # 1D inputs
            self.frame_1d = ttk.LabelFrame(control_frame, text="1D Settings", padding=5)
            self.frame_1d.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
            ttk.Label(self.frame_1d, text="f(x) =").grid(row=0, column=0, sticky="e")
            ttk.Entry(self.frame_1d, textvariable=self.func_str, width=30).grid(row=0, column=1, sticky="w")
            ttk.Label(self.frame_1d, text="a =").grid(row=1, column=0, sticky="e")
            ttk.Entry(self.frame_1d, textvariable=self.a, width=10).grid(row=1, column=1, sticky="w")
            ttk.Label(self.frame_1d, text="b =").grid(row=2, column=0, sticky="e")
            ttk.Entry(self.frame_1d, textvariable=self.b, width=10).grid(row=2, column=1, sticky="w")
            ttk.Label(self.frame_1d, text="# intervals (n):").grid(row=3, column=0, sticky="e")
            ttk.Entry(self.frame_1d, textvariable=self.n_intervals, width=10).grid(row=3, column=1, sticky="w")
            ttk.Label(self.frame_1d, text="Method:").grid(row=4, column=0, sticky="e")
            method_combo = ttk.Combobox(self.frame_1d, textvariable=self.method, values=[
                "Midpoint", "Trapezoidal", "Simpson", "Newton-Cotes (simple)", "Gauss-Legendre", "Gauss-Chebyshev"
            ], state="readonly")
            method_combo.grid(row=4, column=1, sticky="w")
            ttk.Label(self.frame_1d, text="Newton-Cotes points:").grid(row=5, column=0, sticky="e")
            ttk.Entry(self.frame_1d, textvariable=self.n_points_nc, width=10).grid(row=5, column=1, sticky="w")
            ttk.Label(self.frame_1d, text="Gauss points:").grid(row=6, column=0, sticky="e")
            ttk.Entry(self.frame_1d, textvariable=self.n_gauss, width=10).grid(row=6, column=1, sticky="w")

            # 2D inputs (initially hidden)
            self.frame_2d = ttk.LabelFrame(control_frame, text="2D Settings", padding=5)
            self.frame_2d.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
            self.frame_2d.grid_remove()
            ttk.Label(self.frame_2d, text="f(x,y) =").grid(row=0, column=0, sticky="e")
            ttk.Entry(self.frame_2d, textvariable=self.func2d_str, width=30).grid(row=0, column=1)
            ttk.Label(self.frame_2d, text="x in [a,b]:").grid(row=1, column=0, sticky="e")
            ttk.Entry(self.frame_2d, textvariable=self.xa2d, width=8).grid(row=1, column=1, sticky="w")
            ttk.Label(self.frame_2d, text="to").grid(row=1, column=2)
            ttk.Entry(self.frame_2d, textvariable=self.xb2d, width=8).grid(row=1, column=3)
            ttk.Label(self.frame_2d, text="y in [c,d]:").grid(row=2, column=0, sticky="e")
            ttk.Entry(self.frame_2d, textvariable=self.ya2d, width=8).grid(row=2, column=1, sticky="w")
            ttk.Label(self.frame_2d, text="to").grid(row=2, column=2)
            ttk.Entry(self.frame_2d, textvariable=self.yb2d, width=8).grid(row=2, column=3)
            ttk.Label(self.frame_2d, text="# intervals x:").grid(row=3, column=0, sticky="e")
            ttk.Entry(self.frame_2d, textvariable=self.nx2d, width=8).grid(row=3, column=1, sticky="w")
            ttk.Label(self.frame_2d, text="# intervals y:").grid(row=3, column=2, sticky="e")
            ttk.Entry(self.frame_2d, textvariable=self.ny2d, width=8).grid(row=3, column=3)
            ttk.Label(self.frame_2d, text="Method (1D):").grid(row=4, column=0, sticky="e")
            ttk.Combobox(self.frame_2d, textvariable=self.method2d, values=["midpoint","trapezoidal","simpson"], state="readonly").grid(row=4, column=1, sticky="w")

            # Buttons
            btn_frame = ttk.Frame(control_frame)
            btn_frame.grid(row=3, column=0, pady=10)
            ttk.Button(btn_frame, text="Compute", command=self.compute).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Save Results", command=self.save_results).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Save Plot", command=self.save_plot).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Run Animation", command=self.run_animation).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Compare with built-in", command=self.compare_builtin).pack(side=tk.LEFT, padx=5)

            # Result display
            self.result_label = ttk.Label(control_frame, text="Result: ", font=("Arial", 12, "bold"))
            self.result_label.grid(row=4, column=0, pady=10)

            # Error/info display
            self.info_text = tk.Text(control_frame, height=6, width=80)
            self.info_text.grid(row=5, column=0, padx=5, pady=5)

            # ----- Plot Frame -----
            self.fig, self.ax = plt.subplots(figsize=(7,5))
            self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
            toolbar.update()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # ----- Help Frame -----
            help_text = tk.Text(help_frame, wrap=tk.WORD, font=("Arial", 10))
            help_text.pack(fill=tk.BOTH, expand=True)
            help_str = """
            NUMERICAL INTEGRATION EXPLORER - HELP

            This application implements several quadrature methods from scratch.

            1D Integration:
            - Choose a function f(x), integration limits a, b, and number of intervals n.
            - Newton–Cotes (simple) uses n_points nodes (non‑composite, may be unstable for large n_points).
            - Gauss–Legendre and Gauss–Chebyshev use n_gauss points.

            2D Integration:
            - Double integral over a rectangle using iterated 1D quadrature (select the 1D method).

            Visualization:
            - Shows the function and the quadrature approximation (rectangles/trapezoids/parabolas where possible).
            - Animation: increases the number of intervals stepwise to show convergence.

            Saving:
            - Save numerical results as a text file.
            - Save the current plot as PNG/PDF.
            - Animation can be saved using the matplotlib animation GUI (manual).

            Built‑in Comparison:
            - Compares the selected 1D method with scipy.integrate.quad (requires scipy).

            Input validation prevents crashes.
            """
            help_text.insert(tk.END, help_str)
            help_text.config(state=tk.DISABLED)

            self.toggle_dim()  # initial visibility

        def toggle_dim(self):
            if self.dim.get() == "1D":
                self.frame_1d.grid()
                self.frame_2d.grid_remove()
            else:
                self.frame_1d.grid_remove()
                self.frame_2d.grid()

        def get_function(self):
            """Return callable for 1D function from string."""
            try:
                return make_function(self.func_str.get())
            except Exception as e:
                messagebox.showerror("Function Error", str(e))
                return None

        def get_function_2d(self):
            try:
                return make_function_2d(self.func2d_str.get())
            except Exception as e:
                messagebox.showerror("Function Error", str(e))
                return None

        def compute(self):
            try:
                if self.dim.get() == "1D":
                    f = self.get_function()
                    if f is None:
                        return
                    a = self.a.get()
                    b = self.b.get()
                    n = self.n_intervals.get()
                    method = self.method.get()
                    # Validate inputs
                    if a >= b:
                        messagebox.showerror("Input Error", "Lower limit a must be less than b.")
                        return
                    if n <= 0:
                        messagebox.showerror("Input Error", "Number of intervals must be positive.")
                        return

                    # Compute according to method
                    if method == "Midpoint":
                        result = midpoint_rule(f, a, b, n)
                    elif method == "Trapezoidal":
                        result = trapezoidal_rule(f, a, b, n)
                    elif method == "Simpson":
                        try:
                            result = simpson_rule(f, a, b, n)
                        except ValueError as e:
                            messagebox.showerror("Method Error", str(e))
                            return
                    elif method == "Newton-Cotes (simple)":
                        npnt = self.n_points_nc.get()
                        if npnt < 2:
                            messagebox.showerror("Input Error", "Newton-Cotes requires at least 2 points.")
                            return
                        result = newton_cotes_simple(f, a, b, npnt)
                    elif method == "Gauss-Legendre":
                        ng = self.n_gauss.get()
                        if ng < 1:
                            messagebox.showerror("Input Error", "Number of Gauss points must be >=1.")
                            return
                        result = gauss_legendre(f, a, b, ng)
                    elif method == "Gauss-Chebyshev":
                        ng = self.n_gauss.get()
                        if ng < 1:
                            messagebox.showerror("Input Error", "Number of Gauss points must be >=1.")
                            return
                        result = gauss_chebyshev(f, a, b, ng)
                        self.info_text.insert(tk.END, "Note: Gauss-Chebyshev approximates ∫ f(x)/√(1-x²) dx on [-1,1] and scaled to [a,b] as an ad-hoc formula.\n")
                    else:
                        messagebox.showerror("Method Error", "Unknown method")
                        return

                    self.current_result = result
                    self.result_label.config(text=f"Result: {result:.10f}")
                    self.info_text.insert(tk.END, f"Computed using {method} → {result:.10f}\n")
                    self.info_text.see(tk.END)
                    self.current_function = f
                    self.plot_1d(f, a, b, method, n)
                else:  # 2D
                    f2 = self.get_function_2d()
                    if f2 is None:
                        return
                    xa, xb = self.xa2d.get(), self.xb2d.get()
                    ya, yb = self.ya2d.get(), self.yb2d.get()
                    nx = self.nx2d.get()
                    ny = self.ny2d.get()
                    if xa >= xb or ya >= yb:
                        messagebox.showerror("Input Error", "Invalid integration limits.")
                        return
                    if nx <= 0 or ny <= 0:
                        messagebox.showerror("Input Error", "Number of intervals must be positive.")
                        return
                    method2 = self.method2d.get()
                    result = double_integral(f2, xa, xb, ya, yb, nx, ny, method2)
                    self.current_result = result
                    self.result_label.config(text=f"Double integral result: {result:.10f}")
                    self.info_text.insert(tk.END, f"2D integral ({method2}) → {result:.10f}\n")
                    # For 2D we can plot a surface (optional)
                    self.plot_2d(f2, xa, xb, ya, yb)
            except Exception as e:
                messagebox.showerror("Computation Error", str(e))

        def plot_1d(self, f, a, b, method, n):
            """Plot the function and the quadrature approximation."""
            self.ax.clear()
            x_fine = np.linspace(a, b, 500)
            y_fine = f(x_fine)
            self.ax.plot(x_fine, y_fine, 'b-', label='f(x)')

            # Visualize the quadrature rule
            if method in ["Midpoint", "Trapezoidal", "Simpson"]:
                if method == "Midpoint":
                    h = (b - a) / n
                    x_mid = a + h/2 + np.arange(n) * h
                    y_mid = f(x_mid)
                    for i in range(n):
                        self.ax.add_patch(plt.Rectangle((x_mid[i]-h/2, 0), h, y_mid[i],
                                                        facecolor='red', alpha=0.3, edgecolor='red'))
                    self.ax.plot(x_mid, y_mid, 'ro', markersize=4)
                elif method == "Trapezoidal":
                    x_nodes = np.linspace(a, b, n+1)
                    y_nodes = f(x_nodes)
                    for i in range(n):
                        self.ax.fill_between([x_nodes[i], x_nodes[i+1]], [0,0],
                                            [y_nodes[i], y_nodes[i+1]], alpha=0.3, facecolor='green')
                    self.ax.plot(x_nodes, y_nodes, 'go-', markersize=4)
                elif method == "Simpson":
                    # Visualize parabolic segments (simplified: show the area under parabolas)
                    if n % 2 == 0:
                        h = (b - a) / n
                        x_nodes = np.linspace(a, b, n+1)
                        y_nodes = f(x_nodes)
                        for i in range(0, n, 2):
                            x_par = np.linspace(x_nodes[i], x_nodes[i+2], 50)
                            # Quadratic interpolation at three points
                            x0, x1, x2 = x_nodes[i], x_nodes[i+1], x_nodes[i+2]
                            y0, y1, y2 = y_nodes[i], y_nodes[i+1], y_nodes[i+2]
                            # Lagrange polynomial
                            def p(x):
                                return (y0 * (x-x1)*(x-x2)/((x0-x1)*(x0-x2)) +
                                        y1 * (x-x0)*(x-x2)/((x1-x0)*(x1-x2)) +
                                        y2 * (x-x0)*(x-x1)/((x2-x0)*(x2-x1)))
                            y_par = p(x_par)
                            self.ax.fill_between(x_par, 0, y_par, alpha=0.2, facecolor='orange')
                        self.ax.plot(x_nodes, y_nodes, 'mo-', markersize=4)

            self.ax.set_xlabel('x')
            self.ax.set_ylabel('f(x)')
            self.ax.set_title(f'Quadrature: {method} (n={n})')
            self.ax.grid(True)
            self.ax.legend()
            self.canvas.draw()

        def plot_2d(self, f, xa, xb, ya, yb):
            """Simple surface plot for double integral."""

            # self.ax.clear()
            self.fig.clf()
        
        # Create a new 3D axes
            self.ax = self.fig.add_subplot(111, projection='3d')
            
            x = np.linspace(xa, xb, 50)
            y = np.linspace(ya, yb, 50)
            X, Y = np.meshgrid(x, y)
            Z = f(X, Y)
            surf = self.ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')
            self.ax.set_title('Integrand f(x,y)')
            self.canvas.draw()

        def run_animation(self):
            """Animate the convergence of the selected 1D method by increasing n."""
            if self.dim.get() != "1D":
                messagebox.showinfo("Animation", "Animation is available only for 1D integration.")
                return
            f = self.get_function()
            if f is None:
                return
            a = self.a.get()
            b = self.b.get()
            method = self.method.get()
            if method not in ["Midpoint", "Trapezoidal", "Simpson"]:
                messagebox.showinfo("Animation", "Animation only works for composite methods (Midpoint, Trapezoidal, Simpson).")
                return
            # Prepare animation: update n from 2 to 50 (or 2 to 40 for Simpson even steps)
            def update(frame):
                n = frame + 2
                if method == "Simpson" and n % 2 != 0:
                    n += 1
                self.ax.clear()
                x_fine = np.linspace(a, b, 500)
                y_fine = f(x_fine)
                self.ax.plot(x_fine, y_fine, 'b-', label='f(x)')
                if method == "Midpoint":
                    h = (b - a) / n
                    x_mid = a + h/2 + np.arange(n) * h
                    y_mid = f(x_mid)
                    for i in range(n):
                        self.ax.add_patch(plt.Rectangle((x_mid[i]-h/2, 0), h, y_mid[i],
                                                        facecolor='red', alpha=0.3, edgecolor='red'))
                    self.ax.plot(x_mid, y_mid, 'ro', markersize=4)
                elif method == "Trapezoidal":
                    x_nodes = np.linspace(a, b, n+1)
                    y_nodes = f(x_nodes)
                    for i in range(n):
                        self.ax.fill_between([x_nodes[i], x_nodes[i+1]], [0,0],
                                            [y_nodes[i], y_nodes[i+1]], alpha=0.3, facecolor='green')
                    self.ax.plot(x_nodes, y_nodes, 'go-', markersize=4)
                elif method == "Simpson":
                    if n % 2 == 0:
                        h = (b - a) / n
                        x_nodes = np.linspace(a, b, n+1)
                        y_nodes = f(x_nodes)
                        for i in range(0, n, 2):
                            x_par = np.linspace(x_nodes[i], x_nodes[i+2], 50)
                            x0, x1, x2 = x_nodes[i], x_nodes[i+1], x_nodes[i+2]
                            y0, y1, y2 = y_nodes[i], y_nodes[i+1], y_nodes[i+2]
                            def p(xx):
                                return (y0 * (xx-x1)*(xx-x2)/((x0-x1)*(x0-x2)) +
                                        y1 * (xx-x0)*(xx-x2)/((x1-x0)*(x1-x2)) +
                                        y2 * (xx-x0)*(xx-x1)/((x2-x0)*(x2-x1)))
                            y_par = p(x_par)
                            self.ax.fill_between(x_par, 0, y_par, alpha=0.2, facecolor='orange')
                        self.ax.plot(x_nodes, y_nodes, 'mo-', markersize=4)
                self.ax.set_title(f'{method} rule, n = {n}')
                self.ax.grid(True)
                self.ax.legend()
                self.canvas.draw()

            # Number of frames: up to 50
            max_frames = 49  # n from 2 to 50
            if method == "Simpson":
                # ensure even frames: we will manually skip odd n inside update
                pass
            self.animation = FuncAnimation(self.fig, update, frames=max_frames, repeat=False, interval=200)
            self.canvas.draw()
            messagebox.showinfo("Animation", "Animation started. It will run once; close the plot window to stop if needed.")

        def compare_builtin(self):
            """Compare selected 1D method with scipy.integrate.quad."""
            if self.dim.get() != "1D":
                messagebox.showinfo("Comparison", "Comparison only available for 1D.")
                return
            if self.current_result is None:
                messagebox.showwarning("No computation", "Compute a result first.")
                return
            try:
                f = self.get_function()
                a = self.a.get()
                b = self.b.get()
                scipy_result, scipy_err = integrate.quad(f, a, b)
                self.info_text.insert(tk.END, f"Built-in (scipy.quad) result: {scipy_result:.10f}, error est: {scipy_err:.2e}\n")
                self.info_text.insert(tk.END, f"Difference (our - scipy): {self.current_result - scipy_result:.10f}\n")
            except Exception as e:
                messagebox.showerror("Comparison error", str(e))

        def save_results(self):
            if self.current_result is None:
                messagebox.showwarning("No data", "No result to save. Compute first.")
                return
            file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files","*.txt")])
            if file_path:
                with open(file_path, 'w') as f:
                    f.write("Numerical Integration Results\n")
                    f.write("============================\n")
                    f.write(f"Dimension: {self.dim.get()}\n")
                    if self.dim.get() == "1D":
                        f.write(f"Function: f(x) = {self.func_str.get()}\n")
                        f.write(f"Interval: [{self.a.get()}, {self.b.get()}]\n")
                        f.write(f"Method: {self.method.get()}\n")
                        if self.method.get() in ["Midpoint","Trapezoidal","Simpson"]:
                            f.write(f"Number of intervals: {self.n_intervals.get()}\n")
                        elif self.method.get() == "Newton-Cotes (simple)":
                            f.write(f"Number of points: {self.n_points_nc.get()}\n")
                        else:
                            f.write(f"Number of Gauss points: {self.n_gauss.get()}\n")
                        f.write(f"Approximate integral: {self.current_result:.15f}\n")
                    else:
                        f.write(f"Function: f(x,y) = {self.func2d_str.get()}\n")
                        f.write(f"x in [{self.xa2d.get()}, {self.xb2d.get()}], y in [{self.ya2d.get()}, {self.yb2d.get()}]\n")
                        f.write(f"Method: {self.method2d.get()} (iterated)\n")
                        f.write(f"Grid: {self.nx2d.get()} x {self.ny2d.get()}\n")
                        f.write(f"Approximate double integral: {self.current_result:.15f}\n")
                messagebox.showinfo("Saved", f"Results saved to {file_path}")

        def save_plot(self):
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png"),("PDF","*.pdf")])
            if file_path:
                self.fig.savefig(file_path, dpi=150)
                messagebox.showinfo("Saved", f"Plot saved to {file_path}")

    # =============================================================================
    # Run the application
    # =============================================================================
    if __name__ == "__main__":
        root = tk.Tk()
        app = QuadratureApp(root)
        root.mainloop()