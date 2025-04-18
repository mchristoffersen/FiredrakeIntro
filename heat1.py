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
x, y = SpatialCoordinate(mesh)
q = Function(H).interpolate(
    exp(-10 * (x**2 + (y - 0.25) ** 2))
)  # heat production rate
k = Constant(0.1)  # thermal diffusivity

# Plot q
if True:
    fig, ax = plt.subplots()
    im = firedrake.tripcolor(q, axes=ax, cmap="magma")
    ax.set(title="q", xlabel="x coordinate", ylabel="y coordinate", aspect="equal")
    fig.colorbar(im, label="Heat production")
    plt.show()


F = (k * dot(grad(u), grad(v)) - q * v) * dx

bc = DirichletBC(H, Constant(0.0), 1)

# Hand off to Firedrake
solve(
    F == 0,
    u,
    bcs=[bc],
)

# Plot solution
if True:
    fig, ax = plt.subplots()
    im = firedrake.tripcolor(u, axes=ax, cmap="magma")
    ax.set(title="u", xlabel="x coordinate", ylabel="y coordinate", aspect="equal")
    fig.colorbar(im, label="Temperature")
    plt.show()
