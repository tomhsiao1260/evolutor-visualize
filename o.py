import cv2
import cmap
import numpy as np
import matplotlib.pyplot as plt

s = 100
h0, w0 = s, s
xs, ys = np.meshgrid(np.linspace(0, 1, w0), np.linspace(0, 1, h0))

imageA = np.sqrt(xs*xs + ys*ys)
imageA -= np.min(imageA)
imageA /= np.max(imageA)

A_edge = np.pad(imageA, pad_width=1, mode='edge')
dx = A_edge[1:-1,2:] - A_edge[1:-1,1:-1]
dy = A_edge[2:,1:-1] - A_edge[1:-1,1:-1]
dy *= -1
dx = cv2.resize(dx, (h0//10, w0//10), interpolation=cv2.INTER_LINEAR)
dy = cv2.resize(dy, (h0//10, w0//10), interpolation=cv2.INTER_LINEAR)
xss = cv2.resize(xs, (h0//10, w0//10), interpolation=cv2.INTER_LINEAR)
yss = cv2.resize(ys, (h0//10, w0//10), interpolation=cv2.INTER_LINEAR)

row_num, col_num = 2, 2
fig, axes = plt.subplots(row_num, col_num, figsize=(5, 5))
colormap = cmap.Colormap("tab20", interpolation="nearest")
for ax in axes.flat: ax.axis('off')

axes[0, 0].imshow(colormap(imageA), aspect='equal')
axes[0, 0].set_title('F')

axes[0, 1].quiver(xss*w0//10, yss*h0//10, dx, dy, angles='xy')
axes[0, 1].set_xlim(0, w0//10)
axes[0, 1].set_ylim(0, h0//10)
axes[0, 1].grid()
axes[0, 1].set_title('dF')

plt.show()
