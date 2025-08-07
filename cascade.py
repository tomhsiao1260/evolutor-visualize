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
from scipy.interpolate import interp1d

def solveAxEqb(A, b):
    # print("solving Ax = b", A.shape, b.shape)
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
    # print("x", x.shape, x.dtype, x.min(), x.max())
    return x

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

    # h, w = out.shape
    # y, x = np.mgrid[:h, :w]
    # y, x = y/(h-1), x/(w-1)
    # y, x = 2*y-1, 2*x-1
    # y, x = 1-y**2, 1-x**2 # center: 1, edge: 0

    # mask = np.minimum(y, x)
    # out *= mask
    out += basew
    return out

def divp1(s, c):
    n = s // c
    if s%c > 0:
        n += 1
    return n

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
    umb[1] += .5
    iys, ixs = np.mgrid[y:y+h, x:x+w]
    # print("mg", ixs.shape, iys.shape)
    # iys gives row ids, ixs gives col ids
    theta = np.arctan2(iys-umb[1], ixs-umb[0])
    # print("theta", theta.shape, theta.min(), theta.max())
    theta -= np.min(theta)
    theta /= np.max(theta)
    return theta

def createInitUV(zarr_store, umb, x0, y0, w0, h0):
    max_level = len(zarr_store) - 1
    images, images_u, images_v = [], [], []

    for level in range(max_level+1):
        x = divp1(x0, 2**level)
        y = divp1(y0, 2**level)
        w = divp1(w0, 2**level)
        h = divp1(h0, 2**level)

        image = zarr_store[level][0, y:y+h, x:x+w]

        if level == 0:
            image_u = createThetaArray(umb, x0, y0, w0, h0)
            image_v = createRadiusArray(umb, x0, y0, w0, h0)
        else:
            image_u = images_u[level-1][::2, ::2].copy()
            image_v = images_v[level-1][::2, ::2].copy()

        images.append(image)
        images_u.append(image_u)
        images_v.append(image_v)

    return images, images_u, images_v

# def applyInterp(data, x, b):
#     interp = np.interp(data.flatten(), x, b)
#     return interp.reshape(data.shape)

def applyInterp(data, x, b):
    f = interp1d(x, b, kind='linear', fill_value='extrapolate', assume_sorted=True)
    flat_data = data.flatten()
    interp_vals = f(flat_data)
    return interp_vals.reshape(data.shape)

def updateUV(args):
    image, image_u, image_v, image_u_ref, image_v_ref = args

    image = image.astype(np.float32) / 65535.
    st = ST(image)
    st.computeEigens()

    np.copyto(image_u, solveUV(image_u_ref, st, smoothing_weight=.5, axis='u'))
    np.copyto(image_v, solveUV(image_v_ref, st, smoothing_weight=.5, axis='v'))

current_level = 0

def on_key(event, images, images_u, images_v):
    global current_level
    if event.key == ' ':  # space press
        current_level += 1
        if current_level >= len(images):
            plt.close()
        else:
            plt.clf()
            show_image(images, images_u, images_v)

def show_image(images, images_u, images_v):
    global current_level
    print('Level ', current_level)

    plt.subplot(1, 3, 1)
    plt.imshow(images[current_level], cmap='gray', aspect='equal')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    colormap = cmap.Colormap("tab20", interpolation="nearest")
    image = colormap(images_u[current_level])
    plt.imshow(image, aspect='equal')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    colormap = cmap.Colormap("tab20", interpolation="nearest")
    image = colormap(images_v[current_level])
    plt.imshow(image, aspect='equal')
    plt.axis('off')

    plt.draw()

def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="")
    # parser.add_argument("input_ome_zarr",
    #                     help="input ome zarr directory")
    parser.add_argument(
            "--num_threads", 
            type=int, 
            default=cpu_count(), 
            help="Advanced: Number of threads to use for processing. Default is number of CPUs")

    args = parser.parse_args()

    # input_ome_zarr = args.input_ome_zarr
    num_threads = args.num_threads

    level_start, chunk = 3, 128
    umb = np.array([4008, 2304]) # x, y
    x0, y0, w0, h0 = 1000, 1000, chunk*(2**level_start), chunk*(2**level_start)

    z_scroll = zarr.open('./evol1/scroll.zarr/', mode='r')
    images, images_u, images_v = createInitUV(z_scroll, umb, x0, y0, w0, h0)

    for level in reversed(range(level_start)):
        print('Top-Down: solving level ', level)
        tasks = []

        for i in range(2**(level_start-level)):
            for j in range(2**(level_start-level)):
                w, h = chunk, chunk
                x, y = w * i, h * j

                image = images[level][y:y+h, x:x+w]
                image_u = images_u[level][y:y+h, x:x+w]
                image_v = images_v[level][y:y+h, x:x+w]

                image_u_ref = cv2.resize(images_u[level+1], (0, 0), fx=2, fy=2)
                image_v_ref = cv2.resize(images_v[level+1], (0, 0), fx=2, fy=2)
                image_u_ref = image_u_ref[y:y+h, x:x+w]
                image_v_ref = image_v_ref[y:y+h, x:x+w]

                tasks.append((image, image_u, image_v, image_u_ref, image_v_ref))

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(updateUV, tasks), total=len(tasks)))

    for level in range(level_start+1):
        print('Bottom-Up: merging level ', level)

        image_u = images_u[level].copy()
        image_v = images_v[level].copy()

        for i in range(2**(level_start-level)):
            for j in range(2**(level_start-level)):
                w, h = chunk, chunk
                x, y = w * i, h * j

        # for i in range(2**(level_start-level)):
        #     for j in range(2**(level_start-level)):
        #         w, h = chunk, chunk
        #         x, y = w * i, h * j

        #         h_full, w_full = images_u[level].shape

        #         xi = images_v[level][y:y+h, x].copy()
        #         bii = images_v[level][y:y+h, max(x-1, 0)].copy()
        #         xi = xi[:, np.newaxis]
        #         bii = bii[:, np.newaxis]
        #         bi = (bii + xi) / 2
        #         image_v_l = images_v[level][y:y+h, x:x+w].copy() + (bi - xi)

        #         xi = images_u[level][y:y+h, x].copy()
        #         bii = images_u[level][y:y+h, max(x-1, 0)].copy()
        #         xi = xi[:, np.newaxis]
        #         bii = bii[:, np.newaxis]
        #         bi = (bii + xi) / 2
        #         image_u_l = images_u[level][y:y+h, x:x+w].copy() + (bi - xi)
        #         if (level == 0):
        #             if (i == 4 and j == 7):
        #                 print('right: ')
        #                 print(xi[:5, 0])
        #                 print(bii[:5, 0])
        #                 print(bi[:5, 0])
        #                 print(image_u_l[:, 0][:5])
        #                 # image_u_r[:, :5] = 0
        #                 # image_u_l[:, :5] = 0

        #         xi = images_v[level][y:y+h, x+w-1].copy()
        #         bii = images_v[level][y:y+h, min(x+w, w_full-1)].copy()
        #         xi = xi[:, np.newaxis]
        #         bii = bii[:, np.newaxis]
        #         bi = (bii + xi) / 2
        #         image_v_r = images_v[level][y:y+h, x:x+w].copy() + (bi - xi)

        #         xi = images_u[level][y:y+h, x+w-1].copy()
        #         bii = images_u[level][y:y+h, min(x+w, w_full-1)].copy()
        #         xi = xi[:, np.newaxis]
        #         bii = bii[:, np.newaxis]
        #         bi = (bii + xi) / 2
        #         image_u_r = images_u[level][y:y+h, x:x+w].copy() + (bi - xi)
        #         if (level == 0):
        #             if (i == 3 and j == 7):
        #                 print('left: ')
        #                 print(xi[:5, 0])
        #                 print(bii[:5, 0])
        #                 print(bi[:5, 0])
        #                 print(image_u_r[:, -1][:5])
        #                 # image_u_r[:, -5:] = 0
        #                 # image_u_l[:, -5:] = 0
        #         yc, xc = np.mgrid[0:h, 0:w]
        #         yc, xc = yc/(h-1), xc/(w-1)
        #         image_u[y:y+h, x:x+w] = (1-xc) * image_u_l + xc * image_u_r
        #         image_v[y:y+h, x:x+w] = (1-xc) * image_v_l + xc * image_v_r

        # images_u[level] = image_u
        # images_v[level] = image_v

        # image_u = images_u[level].copy()
        # image_v = images_v[level].copy()

        # for i in range(2**(level_start-level)):
        #     for j in range(2**(level_start-level)):
        #         w, h = chunk, chunk
        #         x, y = w * i, h * j

        #         h_full, w_full = images_u[level].shape

        #         xi = images_v[level][y, x:x+w].copy()
        #         bii = images_v[level][max(y-1, 0), x:x+w].copy()
        #         xi = xi[np.newaxis, :]
        #         bii = bii[np.newaxis, :]
        #         bi = (bii + xi) / 2
        #         image_v_t = images_v[level][y:y+h, x:x+w].copy() + (bi - xi)

        #         xi = images_u[level][y, x:x+w].copy()
        #         bii = images_u[level][max(y-1, 0), x:x+w].copy()
        #         xi = xi[np.newaxis, :]
        #         bii = bii[np.newaxis, :]
        #         bi = (bii + xi) / 2
        #         image_u_t = images_u[level][y:y+h, x:x+w].copy() + (bi - xi)
        #         # image_u_t = applyInterp(images_u[level][y:y+h, x:x+w].copy(), xi, bi)
        #         if (level == 0):
        #             if (i == 2 and j == 4):
        #                 print('top: ')
        #                 print(xi[0, :5])
        #                 print(bii[0, :5])
        #                 print(bi[0, :5])
        #                 print(image_u_t[0, :][:5])

        #         xi = images_v[level][y+h-1, x:x+w].copy()
        #         bii = images_v[level][min(y+h, h_full-1), x:x+w].copy()
        #         xi = xi[np.newaxis, :]
        #         bii = bii[np.newaxis, :]
        #         bi = (bii + xi) / 2
        #         image_v_b = images_v[level][y:y+h, x:x+w].copy() + (bi - xi)

        #         xi = images_u[level][y+h-1, x:x+w].copy()
        #         bii = images_u[level][min(y+h, h_full-1), x:x+w].copy()
        #         xi = xi[np.newaxis, :]
        #         bii = bii[np.newaxis, :]
        #         bi = (bii + xi) / 2
        #         image_u_b = images_u[level][y:y+h, x:x+w].copy() + (bi - xi)
        #         # image_u_b = applyInterp(images_u[level][y:y+h, x:x+w].copy(), xi, bi)
        #         if (level == 0):
        #             if (i == 2 and j == 3):
        #                 print('bottom: ')
        #                 print(xi[0, :5])
        #                 print(bii[0, :5])
        #                 print(bi[0, :5])
        #                 print(image_u_b[-1, :][:5])

        #         yc, xc = np.mgrid[0:h, 0:w]
        #         yc, xc = yc/(h-1), xc/(w-1)
        #         image_u[y:y+h, x:x+w] = (1-yc) * image_u_t + yc * image_u_b
        #         image_v[y:y+h, x:x+w] = (1-yc) * image_v_t + yc * image_v_b

        # images_u[level] = image_u
        # images_v[level] = image_v

    # decimation = 2**level_start
    # images_u[level_start] = images_u[0][::decimation,::decimation]
    # images_v[level_start] = images_v[0][::decimation,::decimation]


    plt.figure()
    show_image(images, images_u, images_v)
    plt.connect('key_press_event', lambda event: on_key(event, images, images_u, images_v))
    plt.show()

if __name__ == '__main__':
    sys.exit(main())
