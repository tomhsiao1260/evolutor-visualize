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

    h0, w0 = chunk_left.shape
    xs, ys = np.meshgrid(np.linspace(0, 1, w0), np.linspace(0, 1, h0))

    lx = np.zeros_like(l)
    lx[:, 1:] = l[:, 1:] - l[:, :-1]
    lx[:, 0] = lx[:, 1]

    ldiff = np.zeros_like(l)

    lc = np.zeros_like(l)
    lc[:, 0] = 0
    # lc[:, 0] = l[:, 0]

    h, w = l.shape
    for i in range(w-1):
        lc[:, i+1] = lc[:, i] + lx[:, i] + ldiff[:, i]

    h0, w0 = chunk_left.shape
    chunk = np.zeros((h0, 2*w0), dtype=l.dtype)
    xs, ys = np.meshgrid(np.linspace(0, 1, w0), np.linspace(0, 1, h0))

    x, y, w, h = 0, 0, w0, h0
    chunk[y:y+h, x:x+w]  = lc
    # chunk[y:y+h, x:x+w]  = normalize(-2*l + d)
    # chunk[y:y+h, x:x+w] *= normalize(-1*l + 0) + shift
    # chunk[y:y+h, x:x+w] /= normalize(-1*l + d) + shift
    # chunk[y:y+h, x:x+w]  = normalize(-chunk[y:y+h, x:x+w])
    # chunk[y:y+h, x:x+w]  = normalize(chunk[y:y+h, x:x+w])
    x, y, w, h = w0, 0, w0, h0
    chunk[y:y+h, x:x+w]  = r
    # chunk[y:y+h, x:x+w]  = normalize(2*r + d)
    # chunk[y:y+h, x:x+w] *= normalize(1*r + 0) + shift
    # chunk[y:y+h, x:x+w] /= normalize(1*r + d) + shift
    # chunk[y:y+h, x:x+w]  = normalize(chunk[y:y+h, x:x+w])
    # return chunk
    return normalize(chunk)
    # return align(chunk)

def sum(chunk):
    h0, w0 = chunk.shape
    xs, ys = np.meshgrid(np.linspace(0, 1, w0), np.linspace(0, 1, h0))

    V = np.zeros_like(chunk)
    X = 1 * ys.copy() + 0 * np.ones_like(chunk)
    Y = -1 * xs.copy() + 0 * np.ones_like(chunk)
    Yc = Y.copy()

    L = np.sqrt(X*X + Y*Y)
    X /= L + 1e-4
    Y /= L + 1e-4

    for i in range(h0-1):
        V[i+1, 0] = V[i, 0] + Y[i, 0]

    for i in range(w0-1):
        # Y[1:, i] = V[1:, i] - V[:-1, i]
        # Y[0, i] = Y[1, i]

        # X[:, i] *= (Y[:, i] + 1e-4) / (Yc[:, i] + 1e-4)

        V[:, i+1] = V[:, i] + X[:, i]

    return normalize(X)

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

def align__(chunk):
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

def merge_(chunk_left, chunk_right):
    l = chunk_left.copy()
    r = chunk_right.copy()
    d = l[:,-1:] - r[:,:1]

    # shift = .0
    shift = .5
    h0, w0 = chunk_left.shape
    chunk = np.zeros((h0, 2*w0), dtype=l.dtype)

    x, y, w, h = 0, 0, w0, h0
    chunk[y:y+h, x:x+w]  = normalize(-2*l + d)
    chunk[y:y+h, x:x+w] *= normalize(-1*l + 0) + shift
    chunk[y:y+h, x:x+w] /= normalize(-1*l + d) + shift
    chunk[y:y+h, x:x+w]  = normalize(-chunk[y:y+h, x:x+w])
    x, y, w, h = w0, 0, w0, h0
    chunk[y:y+h, x:x+w]  = normalize(2*r + d)
    chunk[y:y+h, x:x+w] *= normalize(1*r + 0) + shift
    chunk[y:y+h, x:x+w] /= normalize(1*r + d) + shift
    chunk[y:y+h, x:x+w]  = normalize(chunk[y:y+h, x:x+w])
    # return chunk
    return align_(chunk)

def align_(chunk):
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
    # angle_array = xs - 5*ys
    angle_array = xs - ys
    angle_array -= np.min(angle_array)
    angle_array /= np.max(angle_array)
    imageA[y:y+h, x:x+w] = angle_array

    x, y, w, h = w0, 0, w0, h0
    angle_array = xs + ys
    angle_array -= np.min(angle_array)
    angle_array /= np.max(angle_array)
    imageA[y:y+h, x:x+w] = angle_array

    x, y, w, h = 0, 0, w0, h0
    chunk_left = imageA[y:y+h, x:x+w].copy()
    x, y, w, h = w0, 0, w0, h0
    chunk_right = imageA[y:y+h, x:x+w].copy()
    x, y, w, h = 0, 0, 2*w0, h0
    imageB[y:y+h, x:x+w] = merge(chunk_left, chunk_right)

    x, y, w, h = 0, 0, w0, h0
    imageC[y:y+h, x:x+w] = sum(imageC[y:y+h, x:x+w])

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
