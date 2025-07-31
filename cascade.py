import sys
import zarr
import cmap
import cv2
import argparse
from st import ST
import numpy as np
from scipy import sparse
from wind2d import ImageViewer
import matplotlib.pyplot as plt

images_u, index_u = [], 0
images_v, index_v = [], 0

def show_image():
    plt.subplot(1, 2, 1)
    colormap = cmap.Colormap("tab20", interpolation="nearest")
    image = colormap(images_u[index_u])
    plt.imshow(image, aspect='equal')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    colormap = cmap.Colormap("tab20", interpolation="nearest")
    image = colormap(images_v[index_v])
    plt.imshow(image, aspect='equal')
    plt.axis('off')

    plt.draw()

def on_key(event):
    global index_u, index_v
    if event.key == ' ':  # space press
        index_u += 1
        index_v += 1
        if index_u >= len(images_u):
            plt.close()
        else:
            plt.clf()
            show_image()

def solveAxEqb(A, b):
    print("solving Ax = b", A.shape, b.shape)
    At = A.transpose()
    AtA = At @ A
    # print("AtA", AtA.shape, sparse.issparse(AtA))
    # print("ata", ata.shape, ata.dtype, ata[ata!=0], np.argwhere(ata))
    asum = np.abs(AtA).sum(axis=0)
    # print("asum", np.argwhere(asum==0))
    Atb = At @ b
    # print("Atb", Atb.shape, sparse.issparse(Atb))

    lu = sparse.linalg.splu(AtA.tocsc())
    # print("lu created")
    x = lu.solve(Atb)
    print("x", x.shape, x.dtype, x.min(), x.max())
    return x

def createRadiusArray(umb, x, y, w, h):
    iys, ixs = np.mgrid[y:y+h, x:x+w]
    # print("mg", ixs.shape, iys.shape)
    # iys gives row ids, ixs gives col ids
    radsq = (ixs-umb[0])*(ixs-umb[0])+(iys-umb[1])*(iys-umb[1])
    rad = np.sqrt(radsq)
    # print("rad", rad.shape)
    return rad

def createThetaArray(umb, x, y, w, h):
    umb[1] += .5
    iys, ixs = np.mgrid[y:y+h, x:x+w]
    # print("mg", ixs.shape, iys.shape)
    # iys gives row ids, ixs gives col ids
    theta = np.arctan2(iys-umb[1], ixs-umb[0])
    # print("theta", theta.shape, theta.min(), theta.max())
    return theta

def solveUV(basew, st, smoothing_weight, axis='u'):
    vecu = st.vector_u
    coh = st.coherence[:,:,np.newaxis]
    wvecu = coh*vecu
    shape = wvecu.shape[:2]

    if axis=='u':
        sparse_uxg = ImageViewer.sparseVecOpGrad(wvecu, is_cross=False)
    else:
        sparse_uxg = ImageViewer.sparseVecOpGrad(wvecu, is_cross=True)

    sparse_grad = ImageViewer.sparseGrad(shape)
    sparse_u_cross_grad = sparse.vstack((sparse_uxg, smoothing_weight*sparse_grad))

    A = sparse_u_cross_grad
    # print("A", A.shape, A.dtype)

    b = -sparse_u_cross_grad @ basew.flatten()
    b[basew.size:] = 0.
    x = solveAxEqb(A, b)
    out = x.reshape(basew.shape)
    out += basew
    return out

def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="")
    # parser.add_argument("input_ome_zarr",
    #                     help="input ome zarr directory")

    args = parser.parse_args()

    # input_ome_zarr = args.input_ome_zarr

    top_level = 3
    chunk = 128
    umb = np.array([4008, 2304]) # x, y
    x0, y0, w0, h0 = 1000, 1000, chunk*(2**top_level), chunk*(2**top_level)

    z_scroll = zarr.open('./evol1/scroll.zarr/', mode='r')

    u_init = createThetaArray(umb, x0, y0, w0, h0)
    u_init -= np.min(u_init)
    u_init /= np.max(u_init)
    v_init = createRadiusArray(umb, x0, y0, w0, h0)
    v_init -= np.min(v_init)
    v_init /= np.max(v_init)

    for level in reversed(range(top_level+1)):
        decimation = 2**level

        col_u = []
        col_v = []
        for i in range(2**(top_level-level)):
            row_u = []
            row_v = []
            for j in range(2**(top_level-level)):
                w = chunk
                h = chunk
                x = x0 // decimation + w * i
                y = y0 // decimation + h * j
                data = z_scroll[level][0, y:y+h, x:x+w]
                data = data.astype(np.float32) / 65535.
                st = ST(data)
                st.computeEigens()
                if len(images_u) == 0:
                    basew_u = u_init[::decimation,::decimation][h*j:h*(j+1), w*i:w*(i+1)]
                    basew_v = v_init[::decimation,::decimation][h*j:h*(j+1), w*i:w*(i+1)]
                else:
                    basew_u = cv2.resize(images_u[0], (0, 0), fx=2, fy=2)[h*j:h*(j+1), w*i:w*(i+1)]
                    basew_v = cv2.resize(images_v[0], (0, 0), fx=2, fy=2)[h*j:h*(j+1), w*i:w*(i+1)]
                data_u = solveUV(basew_u, st, smoothing_weight=.5, axis='u')
                data_v = solveUV(basew_v, st, smoothing_weight=.5, axis='v')
                row_u.append(data_u)
                row_v.append(data_v)
            col_u.append(np.vstack(row_u))
            col_v.append(np.vstack(row_v))
        image_u = np.hstack(col_u)
        image_v = np.hstack(col_v)
        image_u -= np.min(image_u)
        image_u /= np.max(image_u)
        image_v -= np.min(image_v)
        image_v /= np.max(image_v)
        images_u.insert(0, image_u)
        images_v.insert(0, image_v)

    # for level in range(1, top_level+1):
    #     decimation = 2**level

    #     col_u = []
    #     col_v = []
    #     for i in range(top_level-level+1):
    #         row_u = []
    #         row_v = []
    #         for j in range(top_level-level+1):
    #             w = chunk
    #             h = chunk
    #             x = x0 // decimation + w * i
    #             y = y0 // decimation + h * j
    #             data = z_scroll[level][0, y:y+h, x:x+w]
    #             data = data.astype(np.float32) / 65535.
    #             st = ST(data)
    #             st.computeEigens()
    #             print(level, len(images_u), i, j)
    #             basew_u = images_u[level-1][::2,::2][h*j:h*(j+1), w*i:w*(i+1)]
    #             basew_v = images_v[level-1][::2,::2][h*j:h*(j+1), w*i:w*(i+1)]
    #             data_u = solveUV(basew_u, st, smoothing_weight=.5, axis='u')
    #             data_v = solveUV(basew_v, st, smoothing_weight=.5, axis='v')
    #             row_u.append(data_u)
    #             row_v.append(data_v)
    #         col_u.append(np.vstack(row_u))
    #         col_v.append(np.vstack(row_v))
    #     image_u = np.hstack(col_u)
    #     image_v = np.hstack(col_v)
    #     image_u -= np.min(image_u)
    #     image_u /= np.max(image_u)
    #     image_v -= np.min(image_v)
    #     image_v /= np.max(image_v)
    #     images_u.append(image_u)
    #     images_v.append(image_v)

    plt.figure()
    show_image()
    plt.connect('key_press_event', on_key)
    plt.show()

if __name__ == '__main__':
    sys.exit(main())
