import sys
import zarr
import cmap
import cv2
import argparse
from st import ST
import numpy as np
from tqdm import tqdm
from scipy import sparse
from wind2d import ImageViewer
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from scipy.interpolate import griddata
from test import merge_split

def createRadiusArray(umb, x, y, w, h):
    iys, ixs = np.mgrid[y:y+h, x:x+w]
    # print("mg", ixs.shape, iys.shape)
    # iys gives row ids, ixs gives col ids
    radsq = (ixs-umb[0])*(ixs-umb[0])+(iys-umb[1])*(iys-umb[1])
    rad = np.sqrt(radsq)
    # print("rad", rad.shape)
    rad -= np.min(rad)
    rad /= np.max(rad)
    return rad

def createThetaArray(umb, x, y, w, h):
    # return np.zeros((h, w))

    umb[1] += .5
    iys, ixs = np.mgrid[y:y+h, x:x+w]
    # print("mg", ixs.shape, iys.shape)
    # iys gives row ids, ixs gives col ids
    theta = np.arctan2(iys-umb[1], ixs-umb[0])
    # print("theta", theta.shape, theta.min(), theta.max())
    theta -= np.min(theta)
    theta /= np.max(theta)
    return theta

def main():
    num_plot, n, chunk = 6, 4, 50
    x0, y0, w0, h0 = 0, 0, chunk*n, chunk*n
    umb = np.array([w0//2, h0//2])

    image_uo = createThetaArray(umb, x0, y0, w0, h0)
    image_up = createThetaArray(umb, x0, y0, w0, h0)
    image_vo = createRadiusArray(umb, x0, y0, w0, h0)
    image_vp = createRadiusArray(umb, x0, y0, w0, h0)

    plt.figure(figsize=(13, 4))
    colormap = cmap.Colormap("tab20", interpolation="nearest")

    plt.subplot(1, num_plot, 1)
    plt.imshow(colormap(image_vo), aspect='equal')
    plt.axis('off')

    plt.subplot(1, num_plot, 2)
    plt.imshow(colormap(image_uo), aspect='equal')
    plt.axis('off')

    split = n
    image_vo, _ = merge_split(image_vo, image_vp, split)

    split = n // 2
    x, y, w, h = w0//4, 0, w0//2, h0//2
    image_uo[y:y+h, x:x+w], _ = merge_split(image_uo[y:y+h, x:x+w], image_up[y:y+h, x:x+w], split)

    plt.subplot(1, num_plot, 3)
    plt.imshow(colormap(image_vo), aspect='equal')
    plt.axis('off')

    plt.subplot(1, num_plot, 4)
    plt.imshow(colormap(image_uo), aspect='equal')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    sys.exit(main())




















