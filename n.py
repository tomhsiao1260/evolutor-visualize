import sys
import cmap
import numpy as np
import matplotlib.pyplot as plt

def vec_to_grad(vecx, vecy):
    h0, w0 = vecx.shape
    vx, vy = vecx.copy(), vecy.copy()
    img = np.zeros((h0, w0), dtype=np.float32)

    for i in range(1, w0):
        xi, yi = i, 0

        while xi >= 0:
            if yi != 0:
                img[yi, xi] = img[yi-1, xi] + vy[yi-1, xi]
            else:
                img[yi, xi] = img[yi, xi-1] + 0.01

            if xi != 0:
                vx[yi, xi-1] = img[yi, xi] - img[yi, xi-1]
                vy[yi, xi-1] = vecy[yi, xi-1] * vx[yi, xi-1] / vecx[yi, xi-1]

            xi -= 1
            yi += 1
        else:
            continue

    return img

def main():
    h0, w0 = 100, 100
    imageA = np.zeros((h0, 2*w0), dtype=np.float32)
    imageB = np.zeros((h0, 2*w0), dtype=np.float32)
    imageC = np.zeros((h0, 2*w0), dtype=np.float32)
    imageD = np.zeros((h0, 2*w0), dtype=np.float32)

    xs, ys = np.meshgrid(np.linspace(0, 1, w0), np.linspace(0, 1, h0))

    x, y, w, h = 0, 0, w0, h0
    # img = vec_to_grad(xs, ys)
    img = vec_to_grad(xs + 1, ys + 1)
    # img -= np.min(img)
    # img /= np.max(img) + 1e-5
    imageA[y:y+h, x:x+w] = img

    row_num, col_num = 2, 2
    fig, axes = plt.subplots(row_num, col_num, figsize=(9, 5))
    colormap = cmap.Colormap("tab20", interpolation="nearest")
    for ax in axes.flat: ax.axis('off')

    axes[0, 0].imshow(colormap(imageA), aspect='equal')
    axes[0, 0].set_title('A')

    axes[0, 1].imshow(colormap(imageB), aspect='equal')
    axes[0, 1].set_title('B')

    axes[1, 0].imshow(colormap(imageC), aspect='equal')
    axes[1, 0].set_title('C')

    plt.show()

if __name__ == '__main__':
    sys.exit(main())
