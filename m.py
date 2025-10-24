import sys
import cmap
import numpy as np
import matplotlib.pyplot as plt
from test import create_angle_array
from test import merge_rectangle
from test import merge_bridge
from test import merge_chunk
from test import normalize

def merge(chunk_left, chunk_right, chunk_avg):
    l = chunk_left.copy()
    r = chunk_right.copy()
    a = chunk_avg.copy()

    h0, w0 = chunk_left.shape
    chunk = np.zeros_like(chunk_avg)
    d = l[:,-1:] - r[:,:1]

    x, y, w, h = 0, 0, w0, h0
    # chunk[y:y+h, x:x+w] = l
    chunk[y:y+h, x:x+w] = normalize(-a[y:y+h, x:x+w].copy())
    chunk[y:y+h, x:x+w] *= normalize(-a[y:y+h, x:x+w].copy()) + 1.0
    chunk[y:y+h, x:x+w] /= normalize(-l.copy() + d) + 1.0
    chunk[y:y+h, x:x+w] = normalize(-chunk[y:y+h, x:x+w])
    x, y, w, h = w0, 0, w0, h0
    # chunk[y:y+h, x:x+w] = r
    chunk[y:y+h, x:x+w] = normalize(a[y:y+h, x:x+w].copy())
    chunk[y:y+h, x:x+w] *= normalize(a[y:y+h, x:x+w].copy()) + 1.0
    chunk[y:y+h, x:x+w] /= normalize(r.copy() + d) + 1.0
    chunk[y:y+h, x:x+w] = normalize(chunk[y:y+h, x:x+w])
    return align(chunk)

def align(chunk):
    h0, w0 = chunk.shape
    d = (chunk[:, w0//2-1:w0//2] - chunk[:, w0//2:w0//2+1]) / 2
    x, y, w, h = 0, 0, w0//2, h0
    chunk[y:y+h, x:x+w] -= d
    x, y, w, h = w0//2, 0, w0//2, h0
    chunk[y:y+h, x:x+w] += d
    return normalize(chunk)

def main():
    h0, w0 = 100, 100
    imageA = np.zeros((h0, 2*w0), dtype=np.float32)
    imageB = np.zeros((h0, 2*w0), dtype=np.float32)
    imageC = np.zeros((h0, 2*w0), dtype=np.float32)
    imageD = np.zeros((h0, 2*w0), dtype=np.float32)

    xs, ys = np.meshgrid(np.linspace(0, 1, w0), np.linspace(0, 1, h0))

    x, y, w, h = 0, 0, w0, h0
    # angle_array = create_angle_array(h0)[::-1,::-1]
    angle_array = xs - ys
    # angle_array = xs + ys
    angle_array -= np.min(angle_array)
    angle_array /= np.max(angle_array)
    imageA[y:y+h, x:x+w] = angle_array

    x, y, w, h = w0, 0, w0, h0
    # angle_array = create_angle_array(h0)[:,::-1].T
    angle_array = xs + ys
    angle_array -= np.min(angle_array)
    angle_array /= np.max(angle_array)
    imageA[y:y+h, x:x+w] = angle_array

    x, y, w, h = w0//2, 0, w0, h0
    # imageB[y:y+h, x:x+w] = angle_array
    imageB[y:y+h, x:x+w] = np.linspace(0, 1, w)

    x, y, w, h = 0, 0, w0, h0
    a = imageA[:, w0-1:w0].copy()
    b = imageA[:, w0:w0+1].copy()
    diff = a-b
    imageC[y:y+h, x:x+w] = normalize(-(2.0*imageA[y:y+h, x:x+w].copy() - diff))
    imageC[y:y+h, x:x+w] *= normalize(-(2.0*imageA[y:y+h, x:x+w].copy() - diff)) + 1.0
    imageC[y:y+h, x:x+w] /= normalize(-(1.0*imageA[y:y+h, x:x+w].copy() - diff)) + 1.0
    imageC[y:y+h, x:x+w] = normalize(-imageC[y:y+h, x:x+w])
    x, y, w, h = w0, 0, w0, h0
    imageC[y:y+h, x:x+w] = normalize(2.0*imageA[y:y+h, x:x+w].copy() + diff)
    imageC[y:y+h, x:x+w] *= normalize(2.0*imageA[y:y+h, x:x+w].copy() + diff) + 1.0
    imageC[y:y+h, x:x+w] /= normalize(1.0*imageA[y:y+h, x:x+w].copy() + diff) + 1.0
    imageC[y:y+h, x:x+w] = normalize(imageC[y:y+h, x:x+w])

    x, y, w, h = 0, 0, w0, h0
    a = imageC[:, w0-1:w0].copy()
    b = imageC[:, w0:w0+1].copy()
    diff = (a-b)/2
    imageC[y:y+h, x:x+w] = imageC[y:y+h, x:x+w].copy() - diff
    x, y, w, h = w0, 0, w0, h0
    imageC[y:y+h, x:x+w] = imageC[y:y+h, x:x+w].copy() + diff
    x, y, w, h = 0, 0, 2*w0, h0
    imageC[y:y+h, x:x+w] = normalize(imageC[y:y+h, x:x+w])

    x, y, w, h = w0//2, h0//2, w0//2, h0//2
    chunk_left = imageA[y:y+h, x:x+w].copy()
    x, y, w, h = w0, h0//2, w0//2, h0//2
    chunk_right = imageA[y:y+h, x:x+w].copy()
    x, y, w, h = w0//2, h0//2, w0, h0//2
    chunk_avg = imageB[y:y+h, x:x+w].copy()
    x, y, w, h = w0//2, h0//2, w0, h0//2
    imageD[y:y+h, x:x+w] = merge(chunk_left, chunk_right, chunk_avg)

    x, y, w, h = w0//2, 0, w0//2, h0//2
    chunk_left = imageA[y:y+h, x:x+w].copy()
    x, y, w, h = w0, 0, w0//2, h0//2
    chunk_right = imageA[y:y+h, x:x+w].copy()
    x, y, w, h = w0//2, 0, w0, h0//2
    chunk_avg = imageB[y:y+h, x:x+w].copy()
    x, y, w, h = w0//2, 0, w0, h0//2
    imageD[y:y+h, x:x+w] = merge(chunk_left, chunk_right, chunk_avg)

    x, y, w, h = w0//2, 0, w0, h0
    # imageD[y:y+h, x:x+w] = align(imageD[y:y+h, x:x+w].T).T

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
