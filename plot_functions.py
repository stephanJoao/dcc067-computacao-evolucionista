import os
import numpy as np
import matplotlib.pyplot as plt
from opfunu.cec_based.cec2014 import F12014, F112014


def plot_functions_2d(function):
    x = np.linspace(-100, 100, 500)
    y = np.linspace(-100, 100, 500)

    X, Y = np.meshgrid(x, y)
    Z = np.array(
        [
            function(np.array([xx, yy]))
            for xx, yy in zip(np.ravel(X), np.ravel(Y))
        ]
    ).reshape(X.shape)

    plt.figure()
    plt.contourf(
        X,
        Y,
        Z,
        cmap="viridis",
        levels=np.linspace(np.min(Z), np.max(Z), 100),
    )
    plt.title(
        f"Contour plot of {format(function.__self__.__class__.__name__)}"
    )

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(
        f"contour_plot_{format(function.__self__.__class__.__name__)}.png",
        dpi=300,
    )


def plot_functions_3d(function):
    x = np.linspace(-100, 100, 500)
    y = np.linspace(-100, 100, 500)

    X, Y = np.meshgrid(x, y)
    Z = np.array(
        [
            function(np.array([xx, yy]))
            for xx, yy in zip(np.ravel(X), np.ravel(Y))
        ]
    ).reshape(X.shape)

    plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    plt.title(
        f"Surface plot of {format(function.__self__.__class__.__name__)}"
    )

    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    ax.set_zlim(np.min(Z), np.max(Z))
    plt.tight_layout()
    plt.savefig(
        f"surface_plot_{format(function.__self__.__class__.__name__)}.png",
        dpi=300,
    )


if __name__ == "__main__":
    ndim = 2
    f1 = F12014(ndim=ndim).evaluate
    f11 = F112014(ndim=ndim).evaluate

    # if plots dir does not exist, create it
    if not os.path.exists("plots"):
        os.makedirs("plots")
    os.chdir("plots")

    print("Plotting 2D")
    plot_functions_2d(f1)
    plot_functions_2d(f11)
    print("Plotting 3D")
    plot_functions_3d(f1)
    plot_functions_3d(f11)
