import numpy as np
from matplotlib import pyplot as plt

def rotation(theta=0.):
    _theta = np.array(theta)
    ones = np.ones_like(_theta)
    zeros = np.zeros_like(_theta)
    sin_2theta = np.sin(2.0 * np.deg2rad(_theta))
    cos_2theta = np.cos(2.0 * np.deg2rad(_theta))
    rotation = np.array([
        [ones, zeros, zeros, zeros],
        [zeros, cos_2theta, sin_2theta, zeros],
        [zeros, -sin_2theta, cos_2theta, zeros],
        [zeros, zeros, zeros, ones]
    ])
    permutation = np.append(np.arange(2, rotation.ndim), np.array([0, 1]))
    rotation = np.transpose(rotation, permutation)
    return rotation

def ideal_linear_polarizer(theta=0.):
    first_rotation = rotation(theta)
    second_rotation = rotation(-theta)
    linear_polarizer = np.array([
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    return second_rotation @ linear_polarizer @ first_rotation

def ideal_linear_retarder(delta=0., theta=0.):
    _delta = np.array(delta)
    first_rotation = rotation(theta)
    second_rotation = rotation(-theta)
    sin_delta = np.sin(np.deg2rad(_delta))
    cos_delta = np.cos(np.deg2rad(_delta))
    ones = np.ones_like(sin_delta)
    zeros = np.zeros_like(sin_delta)
    linear_retarder = np.array([
        [ones, zeros, zeros, zeros],
        [zeros, ones, zeros, zeros],
        [zeros, zeros, cos_delta, sin_delta],
        [zeros, zeros, -sin_delta, cos_delta]
    ])
    permutation = np.append(np.arange(2, linear_retarder.ndim), np.array([0, 1]))
    linear_retarder = np.transpose(linear_retarder, permutation)
    return second_rotation @ linear_retarder @ first_rotation

def ideal_ncs(psi=0., delta=0.):
    _psi = np.deg2rad(psi)
    _delta = np.deg2rad(delta)
    n = np.cos(2.0 * _psi) * np.ones_like(_delta)
    c = np.sin(2.0 * _psi) * np.cos(_delta) * np.ones_like(_psi)
    s = np.sin(2.0 * _psi) * np.sin(_delta) * np.ones_like(_psi)
    ones = np.ones_like(n)
    zeros = np.zeros_like(n)
    ncs = np.array([
        [ones, -n, zeros, zeros],
        [-n, ones, zeros, zeros],
        [zeros, zeros, c, s],
        [zeros, zeros, -s, c]
    ])
    permutation = np.append(np.arange(2, ncs.ndim), np.array([0, 1]))
    ncs = np.transpose(ncs, permutation)
    return ncs
    
def intensity_tick_kwargs():
    return {
        'direction': 'in',
        'left': True,
        'right': True,
        'bottom': True,
        'top': True,
        'labelleft': True,
        'labelbottom': True,
        'labelright': True,
        'labeltop': False
    }

def assemble_matrix_tick_kwargs(mat_shape, i, j):
    _rows, _cols = mat_shape
    tick_kwargs = {
        'direction': 'in',
        'left': True,
        'right': True,
        'bottom': True,
        'top': True
    }

    tick_kwargs_first_col = {'labelleft': True, 'labelright': False}
    tick_kwargs_last_col = {'labelleft': False, 'labelright': True}
    tick_kwargs_other_col = {'labelleft': False, 'labelright': False}
    tick_kwargs_other_row = {'labelbottom': False, 'labeltop': False}
    tick_kwargs_last_row = {'labelbottom': True, 'labeltop': False}
    
    kwargs = {}
    kwargs.update(tick_kwargs)
    if j == 0:
        kwargs.update(tick_kwargs_first_col)
    elif j == _cols - 1:
        kwargs.update(tick_kwargs_last_col)
    else:
        kwargs.update(tick_kwargs_other_col)
    if i == _rows - 1:
        kwargs.update(tick_kwargs_last_row)
    else:
        kwargs.update(tick_kwargs_other_row)
    return kwargs
    
def plot_mueller(mueller, x=None, figax=None, title=None, color=None):
    mm_shape = np.asarray(np.shape(mueller))
    _ncols = mm_shape[-1]
    _nrows = mm_shape[-2]
    if figax is None:
        fig, ax = plt.subplots(nrows=_nrows, ncols=_ncols)
    else:
        fig, ax = figax
    image_data = False
    if np.size(mm_shape) > 3:
        image_data = True
    for col in range(_ncols):
        for row in range(_nrows):
            if color is None:
                if x is None:
                    _ = ax[row, col].plot(mueller[..., row, col])
                else:
                    _ = ax[row, col].plot(x, mueller[..., row, col])
            else:
                if x is None:
                    _ = ax[row, col].plot(mueller[..., row, col], color=color)
                else:
                    _ = ax[row, col].plot(x, mueller[..., row, col], color=color)
                    
            _tick_kwargs = assemble_matrix_tick_kwargs((_nrows, _ncols), row, col)
            _ = ax[row, col].tick_params(**_tick_kwargs)
            if figax is None:
                _ = ax[row, col].annotate('$\mathrm{m}_{' + '{}{}}}$'.format(row+1, col+1), xy=(0.05, 0.05), xycoords='axes fraction')
    if not(title is None):
        _ = fig.suptitle(title)
    ax_rows, ax_cols = ax.shape
    line_vals = np.array(ax[0, 0].get_lines()[0].get_ydata())
    for axr in range(ax_rows):
        for axc in range(ax_cols):
            for l in ax[axr, axc].get_lines():
                y_data = l.get_ydata()
                line_vals = np.append(line_vals, y_data)
    y_min = line_vals.min()
    y_max = line_vals.max()
    y_diff = y_max - y_min
    y_min -= 0.15 * y_diff
    y_max += 0.05 * y_diff
    for axr in range(ax_rows):
        for axc in range(ax_cols):
            _ = ax[axr, axc].set_ylim(y_min, y_max)
    return fig, ax

def ideal_depolarizer():
    return np.eye(4)[0][..., None] @ np.eye(4)[0][None, ...]