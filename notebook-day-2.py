import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Redstart: A Lightweight Reusable Booster""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="public/images/redstart.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Project Redstart is an attempt to design the control systems of a reusable booster during landing.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In principle, it is similar to SpaceX's Falcon Heavy Booster.

    >The Falcon Heavy booster is the first stage of SpaceX's powerful Falcon Heavy rocket, which consists of three modified Falcon 9 boosters strapped together. These boosters provide the massive thrust needed to lift heavy payloads—like satellites or spacecraft—into orbit. After launch, the two side boosters separate and land back on Earth for reuse, while the center booster either lands on a droneship or is discarded in high-energy missions.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(
        mo.Html("""
    <iframe width="560" height="315" src="https://www.youtube.com/embed/RYUr-5PYA7s?si=EXPnjNVnqmJSsIjc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>""")
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell(hide_code=True)
def _():
    import scipy
    import scipy.integrate as sci

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    from tqdm import tqdm

    # The use of autograd is optional in this project, but it may come in handy!
    import autograd
    import autograd.numpy as np
    import autograd.numpy.linalg as la
    from autograd import isinstance, tuple
    from numpy.linalg import matrix_rank
    from scipy.integrate import solve_ivp
    from scipy.signal import place_poles
    return (
        FFMpegWriter,
        FuncAnimation,
        mpl,
        np,
        place_poles,
        plt,
        scipy,
        solve_ivp,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The Model

    The Redstart booster in model as a rigid tube of length $2 \ell$ and negligible diameter whose mass $M$ is uniformly spread along its length. It may be located in 2D space by the coordinates $(x, y)$ of its center of mass and the angle $\theta$ it makes with respect to the vertical (with the convention that $\theta > 0$ for a left tilt, i.e. the angle is measured counterclockwise)

    This booster has an orientable reactor at its base ; the force that it generates is of amplitude $f>0$ and the angle of the force with respect to the booster axis is $\phi$ (with a counterclockwise convention).

    We assume that the booster is subject to gravity, the reactor force and that the friction of the air is negligible.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image(src="public/images/geometry.svg"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Constants

    For the sake of simplicity (this is merely a toy model!) in the sequel we assume that: 

      - the total length $2 \ell$ of the booster is 2 meters,
      - its mass $M$ is 1 kg,
      - the gravity constant $g$ is 1 m/s^2.

    This set of values is not realistic, but will simplify our computations and do not impact the structure of the booster dynamics.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Helpers

    ### Rotation matrix

    $$ 
    \begin{bmatrix}
    \cos \alpha & - \sin \alpha \\
    \sin \alpha &  \cos \alpha  \\
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return (R,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Videos

    It will be very handy to make small videos to visualize the evolution of our booster!
    Here is an example of how such videos can be made with Matplotlib and displayed in marimo.
    """
    )
    return


@app.cell(hide_code=True)
def _(FFMpegWriter, FuncAnimation, mo, np, plt, tqdm):
    def make_video(output):
        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        num_frames = 100
        fps = 30 # Number of frames per second

        def animate(frame_index):    
            # Clear the canvas and redraw everything at each step
            plt.clf()
            plt.xlim(0, 2*np.pi)
            plt.ylim(-1.5, 1.5)
            plt.title(f"Sine Wave Animation - Frame {frame_index+1}/{num_frames}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)

            x = np.linspace(0, 2*np.pi, 100)
            phase = frame_index / 10
            y = np.sin(x + phase)
            plt.plot(x, y, "r-", lw=2, label=f"sin(x + {phase:.1f})")
            plt.legend()

            pbar.update(1)

        pbar = tqdm(total=num_frames, desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=num_frames)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")

    _filename = "wave_animation.mp4"
    make_video(_filename)
    mo.show_code(mo.video(src=_filename))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Getting Started""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Constants

    Define the Python constants `g`, `M` and `l` that correspond to the gravity constant, the mass and half-length of the booster.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    g = 1.0
    M = 1.0
    l = 1
    return M, g, l


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Forces

    Compute the force $(f_x, f_y) \in \mathbb{R}^2$ applied to the booster by the reactor.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    f_x & = -f \sin (\theta + \phi) \\
    f_y & = +f \cos(\theta +\phi)
    \end{align*}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Center of Mass

    Give the ordinary differential equation that governs $(x, y)$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    M \ddot{x} & = -f \sin (\theta + \phi) \\
    M \ddot{y} & = +f \cos(\theta +\phi) - Mg
    \end{align*}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Moment of inertia

    Compute the moment of inertia $J$ of the booster and define the corresponding Python variable `J`.
    """
    )
    return


@app.cell
def _(M, l):
    J = M * l * l / 3
    J
    return (J,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Tilt

    Give the ordinary differential equation that governs the tilt angle $\theta$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    J \ddot{\theta} = - \ell (\sin \phi)  f
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Simulation

    Define a function `redstart_solve` that, given as input parameters: 

      - `t_span`: a pair of initial time `t_0` and final time `t_f`,
      - `y0`: the value of the state `[x, dx, y, dy, theta, dtheta]` at `t_0`,
      - `f_phi`: a function that given the current time `t` and current state value `y`
         returns the values of the inputs `f` and `phi` in an array.

    returns:

      - `sol`: a function that given a time `t` returns the value of the state `[x, dx, y, dy, theta, dtheta]` at time `t` (and that also accepts 1d-arrays of times for multiple state evaluations).

    A typical usage would be:

    ```python
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    ```

    Test this typical example with your function `redstart_solve` and check that its graphical output makes sense.
    """
    )
    return


@app.cell(hide_code=True)
def _(J, M, g, l, np, scipy):
    def redstart_solve(t_span, y0, f_phi):
        def fun(t, state):
            x, dx, y, dy, theta, dtheta = state
            f, phi = f_phi(t, state)
            d2x = (-f * np.sin(theta + phi)) / M
            d2y = (+ f * np.cos(theta + phi)) / M - g
            d2theta = (- l * np.sin(phi)) * f / J
            return np.array([dx, d2x, dy, d2y, dtheta, d2theta])
        r = scipy.integrate.solve_ivp(fun, t_span, y0, dense_output=True)
        return r.sol
    return (redstart_solve,)


@app.cell(hide_code=True)
def _(l, np, plt, redstart_solve):
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controlled Landing

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0) = - 2*\ell$,  can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    % y(t)
    y(t)
    = \frac{2(5-\ell)}{125}\,t^3
      + \frac{3\ell-10}{25}\,t^2
      - 2\,t
      + 10
    $$

    $$
    % f(t)
    f(t)
    = M\!\Bigl[
        \frac{12(5-\ell)}{125}\,t
        + \frac{6\ell-20}{25}
        + g
      \Bigr].
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(M, g, l, np, plt, redstart_solve):

    def smooth_landing_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example()
    return


@app.cell
def _(M, g, l, np, plt, redstart_solve):
    def smooth_landing_example_force():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example_force()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Drawing

    Create a function that draws the body of the booster, the flame of its reactor as well as its target landing zone on the ground (of coordinates $(0, 0)$).

    The drawing can be very simple (a rectangle for the body and another one of a different color for the flame will do perfectly!).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image("public/images/booster_drawing.png"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Make sure that the orientation of the flame is correct and that its length is proportional to the force $f$ with the length equal to $\ell$ when $f=Mg$.

    The function shall accept the parameters `x`, `y`, `theta`, `f` and `phi`.
    """
    )
    return


@app.cell(hide_code=True)
def _(M, R, g, l, mo, mpl, np, plt):
    def draw_booster(x=0, y=l, theta=0.0, f=0.0, phi=0.0, axes=None, **options):
        L = 2 * l
        if axes is None:
            _fig, axes = plt.subplots()

        axes.set_facecolor('#F0F9FF') 

        ground = np.array([[-2*l, 0], [2*l, 0], [2*l, -l], [-2*l, -l], [-2*l, 0]]).T
        axes.fill(ground[0], ground[1], color="#E3A857", **options)

        b = np.array([
            [l/10, -l], 
            [l/10, l], 
            [0, l+l/10], 
            [-l/10, l], 
            [-l/10, -l], 
            [l/10, -l]
        ]).T
        b = R(theta) @ b
        axes.fill(b[0]+x, b[1]+y, color="black", **options)

        ratio = l / (M*g) # when f= +MG, the flame length is l 

        flame = np.array([
            [l/10, 0], 
            [l/10, - ratio * f], 
            [-l/10, - ratio * f], 
            [-l/10, 0], 
            [l/10, 0]
        ]).T
        flame = R(theta+phi) @ flame
        axes.fill(
            flame[0] + x + l * np.sin(theta), 
            flame[1] + y - l * np.cos(theta), 
            color="#FF4500", 
            **options
        )

        return axes

    _axes = draw_booster(x=0.0, y=20*l, theta=np.pi/8, f=M*g, phi=np.pi/8)
    _fig = _axes.figure
    _axes.set_xlim(-4*l, 4*l)
    _axes.set_ylim(-2*l, 24*l)
    _axes.set_aspect("equal")
    _axes.grid(True)
    _MaxNLocator = mpl.ticker.MaxNLocator
    _axes.xaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.yaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.set_axisbelow(True)
    mo.center(_fig)
    return (draw_booster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualisation

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin the with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


@app.cell(hide_code=True)
def _(draw_booster, l, mo, np, plt, redstart_solve):
    def sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_1()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_2():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_2()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_3()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_4():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_4()
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    draw_booster,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_1.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_1())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_2():
        L = 2*l

        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_2.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_2())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_3.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_3())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_4():
        L = 2*l
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_4.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_4())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Linearized Dynamics""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Equilibria

    We assume that $|\theta| < \pi/2$, $|\phi| < \pi/2$ and that $f > 0$. What are the possible equilibria of the system for constant inputs $f$ and $\phi$ and what are the corresponding values of these inputs?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ##### A l'équilibre, nous avons : 
    \[ 
    \ddot{x} = \ddot{y} = \ddot{\theta} = 0 
    \]
    ##### Ainsi : 
    \[\ddot{\theta} = 0 \Rightarrow \ell f \sin(\varphi) = 0 \quad \Rightarrow \quad \varphi = 0  \quad (\text{puisque } |\varphi| < \frac{\pi}{2})\]
    ##### La force doit donc être dirigée le long de l’axe du booster.
    ##### Et : 
    \[
    \ddot{x} =0 \Rightarrow -f \sin(\theta+\phi) = 0 \quad \text{Or}  \quad \phi = 0 \Rightarrow  -f \sin(\theta) = 0   \quad \Rightarrow \quad \theta = 0 \quad (\text{puisque } |\theta| < \frac{\pi}{2})
    \]
    ##### Le booster doit donc être parfaitement vertical (aucune force latérale).
    ##### Enfin : 
    \[
    \ddot{y} =0,\phi= 0, \theta = 0 \Rightarrow fcos⁡(0)=Mg \quad \Rightarrow \quad f=Mg
    \]

    ##### Pour les vitesses : 
    \[
    \dot{x} = 0 \Rightarrow x =x_{eq}\\ 
    \dot{y} = 0 \Rightarrow y =y_{eq}
    \]
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linearized Model

    Introduce the error variables $\Delta x$, $\Delta y$, $\Delta \theta$, and $\Delta f$ and $\Delta \phi$ of the state and input values with respect to the generic equilibrium configuration.
    What are the linear ordinary differential equations that govern (approximately) these variables in a neighbourhood of the equilibrium?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ###### Introduisons les erreurs : 
    \[ 	\Delta \theta = \theta - 0 = \theta, \quad \theta_{eq} =0 \text{ (voir question précédente)}\\ 
        \Delta f = f - Mg  \quad f_{eq} =Mg \text{ (voir question précédente)}\\
        \Delta \varphi = \varphi - 0 = \varphi \quad \varphi_{eq} =0 \text{ (voir question précédente)} \\	
        \Delta x = x - x_{eq}\\
    	\Delta y = y - y_{eq}\\
    \]
    ###### On a : 
    \[
    M \ddot{x} = -f \sin(\theta + \varphi) \\
    M\ddot{y} = f \cos(\theta + \varphi) - Mg \\
    \ddot{\theta} = -\ell f \sin(\varphi) 
    \]
    ###### En introduisant les erreurs et en négligeant les termes de second ordre : : 
    \[
    \theta + \varphi \approx \Delta \theta + \Delta \varphi \\
    \sin(\Delta \theta + \Delta \varphi) \approx \Delta \theta + \Delta \varphi \Rightarrow M \ddot{\Delta x} \approx -f (\Delta \theta + \Delta \varphi) \approx -Mg (\Delta \theta + \Delta \varphi) - \Delta f (\Delta \theta + \Delta \varphi) \Rightarrow M \ddot{\Delta x} = -Mg (\Delta \theta + \Delta \varphi)  \\
    \cos(\theta + \varphi) \approx 1 - \frac{1}{2}(\Delta \theta + \Delta \varphi)^2 \approx 1  \Rightarrow  M \ddot{\Delta y} = \Delta f\\
    \sin(\varphi) \approx \varphi = \Delta \varphi, \quad f \approx Mg \Rightarrow \ddot{\Delta \theta} = -\frac{3g}{\ell} \Delta \varphi 
    \]
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ##### Enfin : 
    \[
    M \ddot{\Delta x} = -Mg (\Delta \theta + \Delta \varphi)  \\
    M \ddot{\Delta y} = \Delta f\\
    \ddot{\Delta \theta} = -\frac{3g}{\ell} \Delta \varphi 
    \]
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Standard Form

    What are the matrices $A$ and $B$ associated to this linear model in standard form?
    Define the corresponding NumPy arrays `A` and `B`.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ##### Nous cherchons à exprimer la dynamique linéarisée sous la forme d’état linéaire standard :
    \[
    \dot{\mathbf{x}} = A \mathbf{x} + B \mathbf{u}
    \]
    ##### On a : 


    \[
    \mathbf{x} = \begin{bmatrix} 
    \Delta x \\
    \dot{\Delta x}\\
    \Delta y \\ 
    \dot{\Delta y}\\
    \Delta \theta \\ 
    \dot{\Delta \theta} \\
    \end{bmatrix}
    \]
    ##### Et : 
    \[
    \mathbf{u} = \begin{bmatrix} 
    \Delta f \\ 
    \Delta \varphi 
    \end{bmatrix}
    \]
    ##### Nous avons les équations linéarisées (voir question précédente) : 
    \[
    M \ddot{\Delta x} = -Mg (\Delta \theta + \Delta \varphi) \\
    M \ddot{\Delta y} = \Delta f \\
    \ddot{\Delta \theta} = -\frac{3g}{\ell} \Delta \varphi
    \]
    ##### En premier ordre : 
    \[
    \dot{\mathbf{x}} = \begin{bmatrix} 
    \dot{\Delta x} \\
    \ddot{\Delta x}\\
    \dot{\Delta y} \\ 
    \ddot{\Delta y} \\ 
    \dot{\Delta \theta} \\ 
    \ddot{\Delta \theta} 
    \end{bmatrix}
    \]
    ###### Ecrivons le système sous la forme matricielle :
    \[
    \dot{\mathbf{x}} = A \mathbf{x} + B \mathbf{u}
    \]
    ###### Matrices A et B :

    \[
    A = \begin{bmatrix} 
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}
    \]

    \[
    B = \begin{bmatrix} 
    0 & 0 \\
    0 & -g \\
    0 & 0 \\
    \frac{1}{M} & 0 \\
    0 & 0 \\
    0 & -\frac{3g}{\ell}
    \end{bmatrix}
    \]
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Stability

    Is the generic equilibrium asymptotically stable?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ##### Le système linéaire est asymptotiquement stable si toutes les valeurs propres de \( A \) ont une partie réelle strictement négative.
    ##### Calculons les valeurs propres de A :
    """
    )
    return


@app.cell
def _(M, g, l, np):
    A = np.zeros((6,6))
    A[0,1] = 1
    A[1,4] = -g
    A[2,3] = 1
    A[4,5] = 1

    B = np.zeros((6,2))
    B[1,1] = -g
    B[3,0] = 1.0/M
    B[5,1] = -3.0*g/l

    print("A =", A)
    print("B =", B)

    return A, B


@app.cell
def _(A, np):
    eigenvalues = np.linalg.eigvals(A)
    print("Valeurs propres de A :")
    print(eigenvalues)
    return


@app.cell
def _(mo):
    mo.md(r"""Ainsi, l'équilibre n'est pas asymptotiquement stable.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controllability

    Is the linearized model controllable?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ###### Le système est contrôlable si la matrice : 
    \[
    \mathcal{C} = [B \ \ AB \ \ A^2B \ \ A^3B \ \ A^4B \ \ A^5B] 
    \]
    ###### est de rang 6.
    """
    )
    return


@app.cell
def _(A, B, np):
    AB1  = A.dot(B)
    A2B1 = A.dot(AB1)
    A3B1 = A.dot(A2B1)
    A4B1 = A.dot(A3B1)
    A5B1 = A.dot(A4B1)

    C1   = np.hstack([B, AB1, A2B1, A3B1,A4B1,A5B1])
    print(C1)
    return (C1,)


@app.cell
def _(C1, np):
    r1=np.linalg.matrix_rank(C1)
    print(r1)
    return


@app.cell
def _(mo):
    mo.md(r"""Le rang de C est égale à 6, ainsi le système est contrôlable.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Lateral Dynamics

    We limit our interest in the lateral position $x$, the tilt $\theta$ and their derivatives (we are for the moment fine with letting $y$ and $\dot{y}$ be uncontrolled). We also set $f = M g$ and control the system only with $\phi$.

    What are the new (reduced) matrices $A$ and $B$ for this reduced system?
    Check the controllability of this new system.
    """
    )
    return


@app.cell
def _(g, l, np):
    A_red = np.array([
        [0, 1, 0, 0],
        [0, 0, -g, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])

    B_red = np.array([
        [0],
        [-g],
        [0],
        [-3*g/l]
    ])


    AB  = A_red.dot(B_red)
    A2B = A_red.dot(AB)
    A3B = A_red.dot(A2B)
    C   = np.hstack([B_red, AB, A2B, A3B])
    print(C)
    return


@app.cell
def _(mo):
    mo.md(r"""Le rang de C est 4 donc le système est contrôlable.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linear Model in Free Fall

    Make graphs of $y(t)$ and $\theta(t)$ for the linearized model when $\phi(t)=0$,
    $x(0)=0$, $\dot{x}(0)=0$, $\theta(0) = 45 / 180  \times \pi$  and $\dot{\theta}(0) =0$. What do you see? How do you explain it?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Dans le modèle linéarisé, on a 
    \[
    \ddot{x}(t) = -g \theta_0 \quad \text{(constante)}
    \]

    En intégrant : 

    Vitesse horizontale :

    \[
    \dot{x}(t) = \dot{x}_0 - g \theta_0 t
    \]

    Position horizontale :

    \[
    x(t) = x_0 + \dot{x}_0 t - \frac{g \theta_0}{2} t^2
    \]

    Quant à l’angle \( \theta(t) \), il n’est soumis à aucun couple, donc il reste constant :

    \[
    \theta(t) = \theta_0
    \]
    """
    )
    return


@app.cell
def _(g, np, plt):
    theta0_1 = np.pi / 4 
    x0_1 = 0
    dx0_1 = 0

    t_3 = np.linspace(0, 5, 500)

    theta_t = np.full_like(t_3, theta0_1)

    d2_x_1 = -g * theta0_1
    dx_t = dx0_1 + d2_x_1 * t_3
    x_t = x0_1 + dx0_1 * t_3 + 0.5 * d2_x_1 * t_3**2


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t_3, x_t, label=r'$x(t)$')
    plt.xlabel("Temps (s)")
    plt.ylabel("Position horizontale x(t)")
    plt.grid(True)
    plt.title("Évolution de x(t)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t_3, theta_t, label=r'$\theta(t)$', color='orange')
    plt.xlabel("Temps (s)")
    plt.ylabel("Angle θ(t) [rad]")
    plt.grid(True)
    plt.title("Évolution de θ(t)")
    plt.legend()

    plt.tight_layout()
    plt.show()
    return (theta0_1,)


app._unparsable_cell(
    r"""
    L’angle θ étant constant et non nul, la composante de la gravité selon x est constante.
    Cela crée une accélération constante dans la direction x, donc x(t) suit une parabole.
    Le corps tombe avec une inclinaison fixe, ce qui induit une translation horizontale.
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Manually Tuned Controller

    Try to find the two missing coefficients of the matrix 

    $$
    K =
    \begin{bmatrix}
    0 & 0 & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    manages  when
    $\Delta x(0)=0$, $\Delta \dot{x}(0)=0$, $\Delta \theta(0) = 45 / 180  \times \pi$  and $\Delta \dot{\theta}(0) =0$ to: 

      - make $\Delta \theta(t) \to 0$ in approximately $20$ sec (or less),
      - $|\Delta \theta(t)| < \pi/2$ and $|\Delta \phi(t)| < \pi/2$ at all times,
      - (but we don't care about a possible drift of $\Delta x(t)$).

    Explain your thought process, show your iterations!

    Is your closed-loop model asymptotically stable?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    On rappelle l'équation linéarisée du système :

    $$
    \Delta \ddot{\theta} = -\frac{3g}{\ell} \Delta \varphi
    $$

    On a $K = [0~0~k_3~k_4]^T$, donc :

    $$
    \Delta \varphi = -k_3 \Delta \theta - k_4 \Delta \dot{\theta}
    $$

    En remplaçant la loi de contrôle :

    $$
    \Delta \ddot{\theta} = -\frac{3g}{\ell} \left( -k_3 \Delta \theta - k_4 \Delta \dot{\theta} \right)
    = \frac{3g}{\ell} k_3 \Delta \theta + \frac{3g}{\ell} k_4 \Delta \dot{\theta}
    \Rightarrow \Delta \ddot{\theta} + \left( -\frac{3g}{\ell} k_4 \right) \Delta \dot{\theta}
    + \left( -\frac{3g}{\ell} k_3 \right) \Delta \theta = 0
    $$

    Il s'agit de l'équation d'un système de second ordre :

    $$
    \ddot{\theta} + 2\zeta \omega_n \dot{\theta} + \omega_n^2 \theta = 0
    $$

    Par identification :

    $$
    -\frac{3g}{\ell} k_4 = 2\zeta \omega_n \quad \text{et} \quad -\frac{3g}{\ell} k_3 = \omega_n^2
    $$

    D’où :

    $$
    k_4 = -2\zeta \omega_n \cdot \frac{\ell}{3g}
    \qquad
    k_3 = -\omega_n^2 \cdot \frac{\ell}{3g}
    $$

    Pour une stabilisation en 20 secondes :

    (Nous avons interprété la consigne « stabiliser en 20 secondes environ » comme l’atteinte d’un état où le système est pratiquement revenu à l’équilibre. Pour donner un sens quantitatif à cette exigence, nous avons adopté un critère usuel selon lequel un système est considéré comme stabilisé lorsqu’il atteint 98 % de sa valeur finale, ce qui correspond à une erreur résiduelle de 2 %.)

    Dans le cas d’un système du second ordre à amortissement critique($\zeta = 1$), la réponse temporelle est donnée par :

    $$
    y(t) = 1 - e^{-t/\tau}\left(1 + \frac{t}{\tau} \right)
    $$

    À $t = 4\tau$, cette expression donne approximativement $0{,}98$, soit 98 % de la valeur finale.

    En appliquant ce critère à notre exigence de stabilisation en 20 secondes, nous obtenons : $4\tau = 20 \Rightarrow \tau = 5$ s, ce qui conduit à une pulsation propre $\omega_n = \frac{1}{\tau} = 0{,}2$ rad/s.

    Le choix de $\zeta = 1$ (amortissement critique) s’explique par sa capacité à assurer une réponse rapide sans oscillations ni dépassement, ce qui en fait un bon compromis entre rapidité et stabilité.


    Donc :

    - $\zeta = 1$
    - $\omega_n = 0.2$

    Par la suite :

    $$
    k_4 = -2 \zeta \omega_n \cdot \frac{\ell}{3g}
    = \frac{-0.4}{3}
    $$

    $$
    k_3 = -\omega_n^2 \cdot \frac{\ell}{3g}
    = \frac{-0.04 }{3}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    $$
    s^2 + 2\zeta \omega_n s + \omega_n^2 = 0
    $$

    En remplaçant $ζ=1$ et $\omega_n=0,2$ on a : 

    $$
    s^2 + 0{,}4s + 0{,}04 = 0
    $$

    Les racines sont : 

    $$
    s = -\zeta \omega_n \pm \omega_n \sqrt{\zeta^2 - 1}
    $$

    $$
    s = -0{,}2 \pm 0 \Rightarrow s = -0{,}2
    $$
    """
    )
    return


app._unparsable_cell(
    r"""
    Un système est asymptotiquement stable si et seulement si tous ses pôles ont une partie réelle strictement négative. Comme notre système a un pôle double à $s = -0.2$ (qui est négatif), le système en boucle fermée est asymptotiquement stable.
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Pole Assignment

    Using pole assignement, find a matrix

    $$
    K_{pp} =
    \begin{bmatrix}
    ? & ? & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K_{pp} \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    satisfies the conditions defined for the manually tuned controller and additionally:

      - result in an asymptotically stable closed-loop dynamics,

      - make $\Delta x(t) \to 0$ in approximately $20$ sec (or less).

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Le système est modélisé par :

    \[
    A =
    \begin{bmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & -1 & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0
    \end{bmatrix}, \quad
    B =
    \begin{bmatrix}
    0 \\
    -1 \\
    0 \\
    -3
    \end{bmatrix}
    \]


    La  matrice de commandabilité :

    \[
    \mathcal{C} = [B, AB, A^2B, A^3B]
    \]

    Et on vérifie que \( \text{rang}(\mathcal{C}) = 4 \) (voir partie 'Lateral Dynamics')
    Le système est complètement commandable, on peut donc librement placer les pôles.

    On veut que \( \Delta x(t) \to 0 \) en 20s.  
    La constante de temps est liée à la partie réelle du pôle dominant :

    \[
    \tau \approx \frac{1}{|\text{Re}(\lambda)|}
    \quad \Rightarrow \quad
    |\text{Re}(\lambda)| \geq \frac{1}{20} = 0.05
    \]

    Pour une convergence plus rapide et de bonnes performances dynamiques, on choisit :

    \[
     [-0.3, -0.5, -0.8, -1.0]
    \]


    Tous les pôles sont réels, négatifs, distincts :
    -assurent une bonne vitesse de réponse,
    -évitent les oscillations,
    -garantissent la stabilité.
    """
    )
    return


@app.cell
def _(np):
    A_pp = np.array([
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    B_pp = np.array([[0], [-1], [0], [-3]])
    return A_pp, B_pp


@app.cell
def _(A_pp, B_pp, place_poles):
    desired_poles = [-0.3, -0.5, -0.8, -1.0]
    Kpp = place_poles(A_pp, B_pp, desired_poles).gain_matrix
    print("Kpp =", Kpp)
    return


@app.cell
def _(M, g, l, np, plt, solve_ivp, theta0_1):
    K = np.array([0.04,        0.30333333, -0.81,       -0.96777778])  #kpp

    def nonlinear_model(t, state):
        x, dx, theta, dtheta = state
        phi = -K @ state
        f = M * g  
        ddx = -f * np.sin(theta + phi) / M
        ddtheta = -3 * g / l * phi
        return [dx, ddx, dtheta, ddtheta]


    x00 = [0, 0, theta0_1, 0] 

    t_span = (0, 30)
    t_eval = np.linspace(*t_span, 1000)
    sol = solve_ivp(nonlinear_model, t_span, x00, t_eval=t_eval)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(sol.t, sol.y[0])
    plt.xlabel("Temps (s)")
    plt.ylabel("x(t) [m]")
    plt.title("Évolution de x(t)")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(sol.t, sol.y[2])
    plt.xlabel("Temps (s)")
    plt.ylabel("θ(t) [rad]")
    plt.title("Évolution de θ(t)")
    plt.grid()

    plt.tight_layout()
    plt.show()
    return t_eval, t_span, x00


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Optimal Control

    Using optimal, find a gain matrix $K_{oc}$ that satisfies the same set of requirements that the one defined using pole placement.

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    L'objectif est de déterminer la matrice de gain \(K_{oc}\) en minimisant le coût quadratique \(J=\int_{0}^{\infty}\Bigl(X^TQX+\Delta\varphi^T R\,\Delta\varphi\Bigr)\,dt\).

    On choisit, selon la règle de Bryson, les pénalités : \(Q=\operatorname{diag}(10,\;1,\;10,\;1)\) (correspondant à 1 m de position, 0.2 m/s de vitesse, 0.1 rad d’erreur angulaire, 0.02 rad/s de vitesse angulaire) et \(R=1\).

    Avec  
    \(\;A=\begin{pmatrix}0&1&0&0\\0&0&-1&0\\0&0&0&1\\0&0&0&0\end{pmatrix},\;B=\begin{pmatrix}0\\-1\\0\\-3\end{pmatrix}\) 

    on calcule  \(\bigl[K_{oc},S,E\bigr]=\mathrm{lqr}(A,B,Q,R)\)  et on obtient  \(\displaystyle K_{oc}\approx\begin{pmatrix} 3.16 & 6.98 & -7.55 & -4.6\end{pmatrix}\) (voir l'éxecution dessous), avec valeurs propres de boucle fermée \(\Re(E)\approx\{-0.2,-0.2,-0.2\pm0.1i\}\), ce qui garantit un amortissement critique et un temps de stabilisation d’environ 20 s.
    """
    )
    return


@app.cell
def _(np):
    from control import lqr


    A4 = np.array([[0, 1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0]])

    B4 = np.array([[0],
                   [-1],
                   [0],
                   [-3]])

    Q = np.diag([10, 10, 100, 2000])
    R1 = np.array([[1]])

    K_oc, S, E = lqr(A4, B4, Q, R1)

    print("K_oc =\n", K_oc)
    return


app._unparsable_cell(
    r"""
    Note : nous avons défini initialement les pénalités qui sont dans le markdown. Ce qui est défini dans le code est le choix final après plusieurs itérations.
    """,
    name="_"
)


@app.cell
def _(M, g, l, np, plt, solve_ivp, t_eval, t_span, x00):
    K2 = np.array([ 3.16227766,  6.98459202, -7.55539691, -4.661717])  #koc pour Q = np.diag([1, 10, 1, 10])

    def nonlinear_model_2(t, state):
        x, dx, theta, dtheta = state
        phi = -K2 @ state
        f = M * g  
        ddx = -f * np.sin(theta + phi) / M
        ddtheta = -3 * g / l * phi
        return [dx, ddx, dtheta, ddtheta]



    sol2 = solve_ivp(nonlinear_model_2, t_span, x00, t_eval=t_eval)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(sol2.t, sol2.y[0])
    plt.xlabel("Temps (s)")
    plt.ylabel("x(t) [m]")
    plt.title("Évolution de x(t)")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(sol2.t, sol2.y[2])
    plt.xlabel("Temps (s)")
    plt.ylabel("θ(t) [rad]")
    plt.title("Évolution de θ(t)")
    plt.grid()

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(M, g, l, np, plt, solve_ivp, t_eval, t_span, x00):
    K3 = np.array([3.16227766,  16.37310688, -40.80581412, -50.48671804])  #koc pour Q = np.diag([10, 10, 100, 2000]) 

    def nonlinear_model_3(t, state):
        x, dx, theta, dtheta = state
        phi = -K3 @ state
        f = M * g  
        ddx = -f * np.sin(theta + phi) / M
        ddtheta = -3 * g / l * phi
        return [dx, ddx, dtheta, ddtheta]



    sol3 = solve_ivp(nonlinear_model_3, t_span, x00, t_eval=t_eval)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(sol3.t, sol3.y[0])
    plt.xlabel("Temps (s)")
    plt.ylabel("x(t) [m]")
    plt.title("Évolution de x(t)")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(sol3.t, sol3.y[2])
    plt.xlabel("Temps (s)")
    plt.ylabel("θ(t) [rad]")
    plt.title("Évolution de θ(t)")
    plt.grid()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Validation

    Test the two control strategies (pole placement and optimal control) on the "true" (nonlinear) model and check that they achieve their goal. Otherwise, go back to the drawing board and tweak the design parameters until they do!
    """
    )
    return


if __name__ == "__main__":
    app.run()
