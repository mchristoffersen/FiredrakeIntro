from firedrake import *
import matplotlib.pyplot as plt

# Set up mesh
mesh = UnitDiskMesh(3)

# Plot mesh
if True:
    fig, ax = plt.subplots()
    triplot(mesh, axes=ax)
    ax.set(title="Mesh", xlabel="x coordinate", ylabel="y coordinate", aspect="equal")
    ax.legend()
    plt.show()

# Define function space and functions
H = FunctionSpace(mesh, "CG", 1)

u = Function(H)
v = TestFunction(H)

# Weak form and boundary condition
q = Constant(1.0)  # heat production rate
k = Constant(0.1)  # thermal diffusivity

F = (k * dot(grad(u), grad(v)) - q * v) * dx

bc = DirichletBC(H, Constant(0.0), 1)

# Hand off to Firedrake
solve(
    F == 0,
    u,
    bcs=[bc],
)

# Plot FEM solution
if True:
    fig, ax = plt.subplots()
    pclr = firedrake.tripcolor(u, axes=ax, cmap="magma")
    ax.set(title="u", xlabel="x coordinate", ylabel="y coordinate", aspect="equal")
    fig.colorbar(pclr, label="Temperature")
    plt.show()

# Exact solution
x, y = SpatialCoordinate(mesh)
w = Function(H).interpolate((q / k) * (1 - x**2 - y**2) / 4)
err = Function(H).interpolate(u - w)

# Exact solution comparison plot
if True:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    # FEM solution
    pclr = tripcolor(u, axes=ax[0], cmap="magma")
    vmin, vmax = pclr.get_clim()
    ax[0].set(
        title="FEM Solution",
        xlabel="x coordinate",
        ylabel="y coordinate",
        aspect="equal",
    )

    # Exact solution interpolated onto FEM mesh
    tripcolor(w, axes=ax[1], cmap="magma", vmin=vmin, vmax=vmax)
    ax[1].set(
        title="Exact Solution",
        xlabel="x coordinate",
        aspect="equal",
    )

    fig.colorbar(pclr, ax=ax[:2], label="Temperature")

    # Difference
    pclr = tripcolor(err, axes=ax[2], cmap="magma")
    fig.colorbar(pclr, ax=ax[2], label="FEM-Exact")
    ax[2].set(
        title="Error",
        xlabel="x coordinate",
        ylabel="y coordinate",
        aspect="equal",
    )
    plt.show()
