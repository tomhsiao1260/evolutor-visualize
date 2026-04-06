import cv2
import cmap
import numpy as np
import matplotlib.pyplot as plt

s, f = 100, 10
h0, w0 = s, s
h1, w1 = s//f, s//f

xs, ys = np.meshgrid(np.linspace(0, 1, w0), np.linspace(0, 1, h0))
xs -= 0.5
ys -= 0.5

imageA = np.sqrt(xs*xs + ys*ys)
imageA -= np.min(imageA)
imageA /= np.max(imageA)

padA = np.pad(imageA, pad_width=1, mode='edge')
dx = padA[1:-1,2:] - padA[1:-1,1:-1]
dy = padA[2:,1:-1] - padA[1:-1,1:-1]

dx = cv2.resize(dx, (h1, w1), interpolation=cv2.INTER_LINEAR)
dy = cv2.resize(dy, (h1, w1), interpolation=cv2.INTER_LINEAR)
x1, y1 = np.meshgrid(np.linspace(0, 1, w1), np.linspace(0, 1, h1))

row_num, col_num = 2, 2
fig, axes = plt.subplots(row_num, col_num, figsize=(5, 5))
colormap = cmap.Colormap("tab20", interpolation="nearest")
for ax in axes.flat: ax.axis('off')

axes[0, 0].imshow(colormap(imageA), aspect='equal')
axes[0, 0].set_title('F')

axes[0, 1].quiver(x1*w1, y1*h1, dx, dy, angles='xy')
axes[0, 1].set_xlim(0, w1)
axes[0, 1].set_ylim(h1, 0)
axes[0, 1].grid()
axes[0, 1].set_title('dF')

plt.show()
