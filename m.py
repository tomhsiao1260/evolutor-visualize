import sys
import cmap
import numpy as np
import matplotlib.pyplot as plt
from test import merge_rectangle
from test import merge_bridge
from test import merge_chunk
from test import normalize

def merge(chunk_left, chunk_right):
    l = chunk_left.copy()
    r = chunk_right.copy()
    d = l[:,-1:] - r[:,:1]

    shift = .5
    h0, w0 = chunk_left.shape
    chunk = np.zeros((h0, 2*w0), dtype=l.dtype)

    x, y, w, h = 0, 0, w0, h0
    chunk[y:y+h, x:x+w]  = normalize(-2*l + d)
    chunk[y:y+h, x:x+w] *= normalize(-1*l + 0) + shift
    chunk[y:y+h, x:x+w] /= normalize(-1*l + d) + shift
    chunk[y:y+h, x:x+w]  = normalize(-chunk[y:y+h, x:x+w])
    # chunk[y:y+h, x:x+w] = l
    x, y, w, h = w0, 0, w0, h0
    chunk[y:y+h, x:x+w]  = normalize(2*r + d)
    chunk[y:y+h, x:x+w] *= normalize(1*r + 0) + shift
    chunk[y:y+h, x:x+w] /= normalize(1*r + d) + shift
    chunk[y:y+h, x:x+w]  = normalize(chunk[y:y+h, x:x+w])
    # chunk[y:y+h, x:x+w] = r
    return align(chunk)

def align(chunk):
    h0, w0 = chunk.shape
    s = w0//2
    dl = chunk[:, s-2:s-1] - chunk[:, s-1:s+0]
    dm = chunk[:, s-1:s+0] - chunk[:, s+0:s+1]
    dr = chunk[:, s+0:s+1] - chunk[:, s+1:s+2]
    x, y, w, h = 0, 0, w0//2, h0
    chunk[y:y+h, x:x+w] -= (dm - dl) / 2
    x, y, w, h = w0//2, 0, w0//2, h0
    chunk[y:y+h, x:x+w] += (dm - dr) / 2
    return normalize(chunk)

def main():
    h0, w0 = 100, 100
    imageA = np.zeros((h0, 2*w0), dtype=np.float32)
    imageB = np.zeros((h0, 2*w0), dtype=np.float32)
    imageC = np.zeros((h0, 2*w0), dtype=np.float32)
    imageD = np.zeros((h0, 2*w0), dtype=np.float32)

    xs, ys = np.meshgrid(np.linspace(0, 1, w0), np.linspace(0, 1, h0))

    x, y, w, h = 0, 0, w0, h0
    angle_array = xs - ys
    # angle_array = xs + ys
    angle_array -= np.min(angle_array)
    angle_array /= np.max(angle_array)
    imageA[y:y+h, x:x+w] = angle_array

    x, y, w, h = w0, 0, w0, h0
    angle_array = xs + ys
    angle_array -= np.min(angle_array)
    angle_array /= np.max(angle_array)
    imageA[y:y+h, x:x+w] = angle_array

    x, y, w, h = w0//2, 0, w0, h0
    imageB[y:y+h, x:x+w] = np.linspace(0, 1, w)

    x, y, w, h = 0, 0, w0, h0
    chunk_left = imageA[y:y+h, x:x+w].copy()
    x, y, w, h = w0, 0, w0, h0
    chunk_right = imageA[y:y+h, x:x+w].copy()
    x, y, w, h = 0, 0, 2*w0, h0
    imageC[y:y+h, x:x+w] = merge(chunk_left, chunk_right)

    x, y, w, h = w0//2, 0, w0//2, h0
    chunk_left = imageA[y:y+h, x:x+w].copy()
    x, y, w, h = w0, 0, w0//2, h0
    chunk_right = imageA[y:y+h, x:x+w].copy()
    x, y, w, h = w0//2, 0, w0, h0
    imageD[y:y+h, x:x+w] = merge(chunk_left, chunk_right)

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

    axes[1, 1].imshow(colormap(imageD), aspect='equal')
    axes[1, 1].set_title('D')

    plt.show()

if __name__ == '__main__':
    sys.exit(main())
