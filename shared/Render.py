import os, glob, subprocess
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
    ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', alpha=0.5)
    if point is not None:
        px, py = point
        pz = evaluation_function(np.array([px, py]))
        ax.scatter(px, py, pz, color='black', s=60, depthshade=False, zorder=10)
    ax.set_title(f"{str.capitalize(evaluation_function.__name__)} function")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)
    return img
