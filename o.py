import cv2
import cmap
import numpy as np
import matplotlib.pyplot as plt

h0, w0 = 100, 100
x0, y0 = np.meshgrid(np.linspace(0, 1, w0), np.linspace(0, 1, h0))

h1, w1 = h0//10, w0//10
x1, y1 = np.meshgrid(np.arange(w1), np.arange(h1))

row_num, col_num = 2, 6
fig, axes = plt.subplots(row_num, col_num, figsize=(15, 5))
colormap = cmap.Colormap("tab20", interpolation="nearest")
for ax in axes.flat: ax.axis('off')

# A = np.sqrt(x0*x0 + y0*y0)
A = x0 - y0
A -= np.min(A)
A /= np.max(A)

dx = A[1:-1,2:] - A[1:-1,1:-1]
dy = A[2:,1:-1] - A[1:-1,1:-1]
dx = cv2.resize(dx, (h1, w1), interpolation=cv2.INTER_LINEAR)
dy = cv2.resize(dy, (h1, w1), interpolation=cv2.INTER_LINEAR)

i, j = 0, 0
axes[i, j].imshow(colormap(A), aspect='equal')
axes[i, j].set_title('F')

i, j = 0, 1
axes[i, j].quiver(x1, y1, dx, dy, angles='xy')
axes[i, j].set_aspect('equal')
axes[i, j].set_xlim(0, w1)
axes[i, j].set_ylim(h1, 0)
axes[i, j].grid()
axes[i, j].set_title('dF')

B = y0 * x0
B -= np.min(B)
B /= np.max(B)

dx = B[1:-1,2:] - B[1:-1,1:-1]
dy = B[2:,1:-1] - B[1:-1,1:-1]
dx = cv2.resize(dx, (h1, w1), interpolation=cv2.INTER_LINEAR)
dy = cv2.resize(dy, (h1, w1), interpolation=cv2.INTER_LINEAR)

i, j = 0, 2
axes[i, j].imshow(colormap(B), aspect='equal')
axes[i, j].set_title('F')

i, j = 0, 3
axes[i, j].quiver(x1, y1, dx, dy, angles='xy')
axes[i, j].set_aspect('equal')
axes[i, j].set_xlim(0, w1)
axes[i, j].set_ylim(h1, 0)
axes[i, j].grid()
axes[i, j].set_title('dF')

C = x0-y0 + y0*x0
C -= np.min(C)
C /= np.max(C)

dx = C[1:-1,2:] - C[1:-1,1:-1]
dy = C[2:,1:-1] - C[1:-1,1:-1]
dx = cv2.resize(dx, (h1, w1), interpolation=cv2.INTER_LINEAR)
dy = cv2.resize(dy, (h1, w1), interpolation=cv2.INTER_LINEAR)

i, j = 0, 4
axes[i, j].imshow(colormap(C), aspect='equal')
axes[i, j].set_title('F')

i, j = 0, 5
axes[i, j].quiver(x1, y1, dx, dy, angles='xy')
axes[i, j].set_aspect('equal')
axes[i, j].set_xlim(0, w1)
axes[i, j].set_ylim(h1, 0)
axes[i, j].grid()
axes[i, j].set_title('dF')

plt.show()
