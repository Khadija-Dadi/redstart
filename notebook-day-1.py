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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell
def _():
    import scipy
    import scipy.integrate as sci
    from scipy.integrate import solve_ivp
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from tqdm import tqdm

    # The use of autograd is optional in this project, but it may come in handy!
    import autograd
    import autograd.numpy as np
    import autograd.numpy.linalg as la
    from autograd import isinstance, tuple
    return FFMpegWriter, FuncAnimation, np, plt, solve_ivp, tqdm


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


@app.cell
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return


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


@app.cell
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
    (mo.video(src=_filename))
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


@app.cell
def _():
    l = 1
    M = 1
    g = 1
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


@app.cell
def _(mo):
    mo.md(
        r"""
    \[
    \begin{pmatrix}
    f_x \\[6pt]
    f_y
    \end{pmatrix}
    = f
    \begin{pmatrix}
    -\sin\bigl(\theta+\Phi\bigr) \\[4pt]
    \cos\bigl(\theta+\Phi\bigr)
    \end{pmatrix}.
    \]
    """
    )
    return


@app.cell
def _(np):
    f = 0
    theta = 0
    phi = 0
    fx = -f * np.sin(theta + phi)
    fy =  f * np.cos(theta + phi) 
    return f, fx, fy, phi


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Center of Mass

    Give the ordinary differential equation that governs $(x, y)$.
    """
    )
    return


app._unparsable_cell(
    r"""
    \[
    \begin{cases}
    \ddot x = -\dfrac{f}{M}\,\sin(\theta+\varPhi),\\[8pt]
    \ddot y = \dfrac{f}{M}\,\cos(\theta+\varPhi)\;-\;g.
    \end{cases}
    \]
    """,
    name="_"
)


@app.cell
def _(M, fx, fy, g):
    d2_x = fx / M
    d2_y = fy / M - g
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
def _(mo):
    mo.md(
        r"""

    \[
    J = \frac{1}{12} M (2l) ^2 =  \frac{1}{3} M \ell^2
    \]
    """
    )
    return


@app.cell
def _(M, l):
    J = 1/3* M*l**2
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


app._unparsable_cell(
    r"""
    \[
    \ddot{\theta} = - \frac{3f}{M\ell} \, \sin(\Phi)
    \]
    """,
    name="_"
)


@app.cell
def _(f, l, np, phi):
    torque = - f * l * np.sin(phi) # moment de f
    return (torque,)


@app.cell
def _(J, torque):
    d2_theta =torque / J
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


@app.cell
def _(J, M, g, l, np):
    def f_fusee (t, y, f_phi):
            x, dx, y, dy, theta, dtheta = y
            f , phi = f_phi(t,y)

            fx = -f* np.sin(theta + phi)
            fy = f * np.cos(theta + phi)

            torque = -f*l*np.sin(phi)
        
            d2_x = fx/M
            d2_y = fy/M - g

            d2_theta = torque/J

            return [dx, d2_x, dy, d2_y, dtheta, d2_theta]
   
    return (f_fusee,)


@app.cell
def _(f_fusee, solve_ivp):
    def redstart_solve(t_span, y0, f_phi):
        solution = solve_ivp(
            fun = lambda t,y : f_fusee(t, y, f_phi),t_span = t_span, y0 =y0, dense_output=True)
        return solution.sol
    return (redstart_solve,)


@app.cell
def _(l, np, plt, redstart_solve):
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] 
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # 
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

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0)$, can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell
def _(np):
    def find_coefficients(y0, yp0, y5, yp5):
        A = np.array([
            [0, 0, 0, 1],    # y(0) = d = y0
            [0, 0, 1, 0],    # y'(0) = c = y'0
            [125, 25, 5, 1], # y(5) = 125a + 25b + 5c + d = y5
            [75, 10, 1, 0]   # y'(5) = 75a + 10b + c = y'5
        ])
        B = np.array([y0, yp0, y5, yp5])
        coeffs = np.linalg.solve(A, B)
        return coeffs
    return (find_coefficients,)


@app.cell
def _(find_coefficients, l):
    a, b, c, d = find_coefficients(10, 0, l, 0)

    return a, b, c, d


@app.cell
def _(M, a, b, c, d, g):
    def optimal_force_cubic(t):
        y_ = a*t*3 + b*t*2 + c*t + d
        dy = 3*a*t**2 + 2*b*t + c
        d2_y = 6*a*t + 2*b
    
        f = M * (d2_y + g)
        return f

    return (optimal_force_cubic,)


@app.cell
def _(f_fusee, solve_ivp):
    def redstart(t_span, y0, f_phi):
        solution = solve_ivp(
            fun=lambda t, y:f_fusee(t, y, f_phi),
            t_span=t_span,
            y0=y0,
            dense_output=True
        )
        return solution.sol
    return (redstart,)


@app.cell
def _(l, np, optimal_force_cubic, plt, redstart):
    def optimal_scenario_cubic():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]  
    
        def f_phi_control(t, y):
            f = optimal_force_cubic(t)
            return np.array([f, 0.0])  
    
        sol = redstart(t_span, y0, f_phi_control)
    
        t = np.linspace(t_span[0], t_span[1], 100)
        y_t = sol(t)
    
        plt.figure(figsize=(12, 6))
    
        plt.subplot(1, 2, 1)
        plt.plot(t, y_t[2], label="Hauteur y(t)")
        plt.axhline(y=l, color='r', linestyle='--', label='y = l')
        plt.xlabel('Temps (s)')
        plt.ylabel('Hauteur (m)')
        plt.title('Trajectoire optimale')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        forces = [optimal_force_cubic(ti) for ti in t]
        plt.plot(t, forces, color='orange', label="Force f(t)")
        plt.xlabel('Temps (s)')
        plt.ylabel('Force (N)')
        plt.title('Profil de poussée')
        plt.grid(True)
        plt.legend()
    
        plt.tight_layout()
        plt.show()

        # Vérification des conditions finales
        y_final = sol(5.0)
        print(f"Hauteur finale: {y_final[2]:.4f} m (cible: 1 m)")
        print(f"Vitesse finale: {y_final[3]:.4f} m/s (cible: 0 m/s)")

    # Exécution
    optimal_scenario_cubic()
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


@app.cell
def _(np, plt):
    def draw_booster(x, y, theta, title):
        fig, ax = plt.subplots(figsize=(8, 6))
    
        ax.scatter(0, 0, color='red', s=100, label='Target Landing Zone')
    
        body_length = 1.0
        body_height = 2.0
    
        corners = np.array([[-body_length/2, -body_length/2, body_length/2, body_length/2],
                            [0, body_height, body_height, 0]])
    
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    
        rotated = rot @ corners
        rotated[0, :] += x
        rotated[1, :] += y
    
        ax.fill(rotated[0, :], rotated[1, :], color='blue', label='Booster Body')
    
        flame_length = 0.6
        flame_height = 1.5
    
        flame = np.array([[0, -flame_length/2, flame_length/2],
                          [-flame_height, 0, 0]])
    
        flame_rotated = rot @ flame
        flame_rotated[0, :] += x
        flame_rotated[1, :] += y
    
        ax.fill(flame_rotated[0, :], flame_rotated[1, :], color='orange', label='Reactor Flame')
    
        ax.set_xlim(-5, 5)
        ax.set_ylim(-2, 12)
        ax.set_aspect('equal')
        ax.set_xlabel('Horizontal Position (m)')
        ax.set_ylabel('Vertical Position (m)')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
    
        plt.show()

    return (draw_booster,)


@app.cell
def _(draw_booster):
    x, y, theta_2, title = 0, 10, 0, "Booster Visualization"
    draw_booster(x, y, theta_2, title)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualization

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


@app.cell
def _(np, plt):
    from IPython.display import HTML
    def draw_booster2(x, y, theta, title, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
    
        ax.scatter(0, 0, color='red', s=100, label='Target')
    
        body_length = 1.0
        body_height = 2.0
        corners = np.array([[-body_length/2, -body_length/2, body_length/2, body_length/2],
                           [0, body_height, body_height, 0]])
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
        rotated = rot @ corners
        rotated[0, :] += x
        rotated[1, :] += y
        ax.fill(rotated[0, :], rotated[1, :], color='blue', label='Booster')
    
        flame_length = 0.6
        flame_height = 1.5
        flame = np.array([[0, -flame_length/2, flame_length/2],
                         [-flame_height, 0, 0]])
        flame_rotated = rot @ flame
        flame_rotated[0, :] += x
        flame_rotated[1, :] += y
        ax.fill(flame_rotated[0, :], flame_rotated[1, :], color='orange', label='Flame')
    
        ax.set_xlim(-5, 5)
        ax.set_ylim(-2, 12)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

    return HTML, draw_booster2


@app.cell
def _(
    FFMpegWriter,
    FuncAnimation,
    draw_booster2,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def create_single_video(scenario_dict, output_prefix="booster"):
        fig = plt.figure(figsize=(10, 6))
        num_frames = 150
        fps = 30
    
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        sol = redstart_solve((0, 5), y0, scenario_dict["f_phi"])
    
        def animate(frame):
            plt.clf()
            ax = fig.add_subplot(111)
            t = 5 * frame / num_frames
            y_t = sol(t)
        
            draw_booster2(y_t[0], y_t[2], y_t[4], 
                        f"{scenario_dict['name']}\nTime: {t:.2f}s", 
                        ax=ax)
        
            ax.text(0.02, 0.95, 
                   f"Position: ({y_t[0]:.2f}, {y_t[2]:.2f})\nAngle: {np.degrees(y_t[4]):.1f}°",
                   transform=ax.transAxes,
                   bbox=dict(facecolor='white', alpha=0.7))
    
        video_file = f"{output_prefix}_{scenario_dict['name'].split()[0]}.mp4"
    
        with tqdm(total=num_frames, desc=f"Creating {scenario_dict['name']}") as pbar:
            anim = FuncAnimation(fig, lambda x: (animate(x), pbar.update(1)), frames=num_frames)
            writer = FFMpegWriter(fps=fps)
            anim.save(video_file, writer=writer)
    
        plt.close()
        return video_file
    return (create_single_video,)


@app.cell
def _(HTML, create_single_video):
    # Scenario 1
    video1 = create_single_video(
        {"name": "1 Free fall", "f_phi": lambda t, y: (0, 0)}
    )
    HTML(f'<video controls width="600" src="{video1}"></video>')

    return


@app.cell
def _(HTML, M, create_single_video, g):
    # Scenario 2
    video2 = create_single_video(
        {"name": "2 Vertical thrust", "f_phi": lambda t, y: (M*g, 0)}
    )
    HTML(f'<video controls width="600" src="{video2}"></video>')
    return


@app.cell
def _(HTML, M, create_single_video, g, np):

    # Scenario 3
    video3 = create_single_video(
        {"name": "3 Angled thrust", "f_phi": lambda t, y: (M*g, np.pi/8)}
    )
    HTML(f'<video controls width="600" src="{video3}"></video>')
    return


if __name__ == "__main__":
    app.run()
