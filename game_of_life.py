import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib import animation


def life_step_gen_expr(x):
    nbrs_count = sum(np.roll(np.roll(x, i, 0), j, 1)
                     for i in (-1, 0, 1)
                     for j in (-1, 0, 1)
                     if (i != 0 or j != 0))
    return (nbrs_count == 3) | (x & nbrs_count == 2)


def life_step_scipy(x):
    nbrs_count = convolve2d(x, np.ones((3, 3)), mode='same', boundary='wrap')
    return (nbrs_count == 3) | (x & nbrs_count == 2)


life_step = life_step_gen_expr


def life_animation(x, dpi=10, frames=10, interval=300):
    x = np.asarray(x)
    assert x.ndim == 2
    x = x.astype(bool)

    x_blank = np.zeros_like(x)
    figsize = (x.shape[1] * 1. / dpi, x.shape[0] * 1. / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)
    im = ax.imshow(x, cmap=plt.cm.binary, interpolation='nearest')
    im.set_clim(-0.05, 1)  # make background gray

    # initialization function: plot the background of each frame
    def init():
        im.set_data(x_blank)
        return im,

    # animation function: this is called sequentially
    def animate(i):
        im.set_data(animate.x)
        animate.x = life_step(animate.x)
        return im,

    animate.x = x
    return animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=interval, repeat=False)


if __name__ == '__main__':
    np.random.seed(0)
    x = np.zeros((300, 400), dtype=bool)
    r = np.random.random((300, 400))
    x = (r > 0.75)
    a = life_animation(x, dpi=10, frames=100)
    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()
