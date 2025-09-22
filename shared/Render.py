import os, glob, subprocess
import matplotlib
import numpy as np
import matplotlib.pyplot as plt



def collect_saved_frames(folder: str, output: str, fps=23, pattern='*.png'):
    patern_path = os.path.join(folder, pattern)
    output_path = os.path.join(folder, output)
    subprocess.run([
            'ffmpeg', '-y', '-framerate', str(fps), '-pattern_type', 'glob', '-i', patern_path, '-filter_complex',
            "[0:v] split [a][b]; [a] palettegen=stats_mode=full [p]; [b][p] paletteuse=dither=bayer:bayer_scale=3", 
            output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    for file in glob.glob(patern_path):
        os.remove(file)


def render_function_point(
        evaluation_function,
        rnge = (-5, 5),
        point: tuple = None,
        iteration: int = None,
        resolution = 30,
        dpi = 100):
    x = np.linspace(*rnge, resolution)
    y = np.linspace(*rnge, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = evaluation_function(np.array([X[i, j], Y[i, j]]))
    fig_size_inches = 500 / dpi
    fig, ax = plt.subplots(figsize=(fig_size_inches, fig_size_inches), dpi=dpi, subplot_kw={'projection': '3d'})
    ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', alpha=0.3)
    if point is not None:
        px, py = point
        pz = evaluation_function(np.array([px, py]))
        ax.scatter(px, py, pz, color='black', s=60, depthshade=False, zorder=10)
    text = f"{str.capitalize(evaluation_function.__name__)} function"
    if iteration is not None:
        text += f": iteration {iteration}"
    ax.set_title(text)
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)
    return img


def render_TSP(points, title="Points with Arrows", arrow_scale=1.0):
    arr = np.array(points, dtype=float)
    xs, ys, ids = arr[:, 0], arr[:, 1], arr[:, 2].astype(int)
    cmap = matplotlib.colormaps.get_cmap('tab10')
    unique_ids = np.unique(ids)
    id_to_color = {uid: cmap(i % cmap.N) for i, uid in enumerate(unique_ids)}
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = [id_to_color[i] for i in ids]
    ax.scatter(xs, ys, c=colors, s=60, zorder=3, edgecolors='black', linewidths=0.6)
    for x, y, pid in zip(xs, ys, ids):
        ax.text(x + 0.07, y + 0.05, f"{pid}", fontsize=12, color=id_to_color[pid])
    for i in range(len(xs) - 1):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i + 1], ys[i + 1]
        dx, dy = x1 - x0, y1 - y0
        ax.arrow(
            x0, y0, dx, dy,
            length_includes_head = True,
            head_width = 0.05 * arrow_scale,
            head_length = 0.1 * arrow_scale,
            linewidth = 1.5,
            color = "black",
            alpha = 0.8,
            zorder = 2,
        )
    x0, y0 = xs[-1], ys[-1]
    x1, y1 = xs[0], ys[0]
    dx, dy = x1 - x0, y1 - y0
    ax.arrow(
        x0, y0, dx, dy,
        length_includes_head = True,
        head_width = 0.05 * arrow_scale,
        head_length = 0.1 * arrow_scale,
        linewidth = 1.5,
        color = "black",
        alpha = 0.8,
        zorder = 2,
    )
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)
    xpad = max(0.5, 0.1 * (xs.max() - xs.min() + 1e-9))
    ypad = max(0.5, 0.1 * (ys.max() - ys.min() + 1e-9))
    ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
    ax.set_ylim(ys.min() - ypad, ys.max() + ypad)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)
    return img
