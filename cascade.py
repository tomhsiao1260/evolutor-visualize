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

def solveUV0(basew, st, smoothing_weight, axis='u'):
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

def solveUV1(basew, st, smoothing_weight, cross_weight, axis='u'):
    icw = 1.-cross_weight
    uvec = st.vector_u
    coh = st.coherence.copy()
    coh = coh[:,:,np.newaxis]
    wuvec = coh*uvec

    shape = wuvec.shape[:2]
    sparse_u_cross_g = ImageViewer.sparseVecOpGrad(wuvec, is_cross=True)
    sparse_u_dot_g = ImageViewer.sparseVecOpGrad(wuvec, is_cross=False)
    sparse_grad = ImageViewer.sparseGrad(shape)
    sgx, sgy = ImageViewer.sparseGrad(shape, interleave=False)
    hxx = sgx.transpose() @ sgx
    hyy = sgy.transpose() @ sgy
    hxy = sgx @ sgy

    A = sparse.vstack((icw*sparse_u_dot_g, cross_weight*sparse_u_cross_g, smoothing_weight*hxx, smoothing_weight*hyy))

    b = np.zeros((A.shape[0]), dtype=np.float64)
    if axis == 'v':
        b[:basew.size] = 1.*coh.flatten()*icw
    else:
        b[basew.size:2*basew.size] = 1.*coh.flatten()*icw

    x = solveAxEqb(A, b)
    out = x.reshape(basew.shape)
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
    return np.zeros((h, w))

    umb[1] += .5
    iys, ixs = np.mgrid[y:y+h, x:x+w]
    # print("mg", ixs.shape, iys.shape)
    # iys gives row ids, ixs gives col ids
    theta = np.arctan2(iys-umb[1], ixs-umb[0])
    print("theta", theta.shape, theta.min(), theta.max())
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

def calculateUV(args):
    image_u, image_v, st = args

    v0 = solveUV0(image_v, st, smoothing_weight=.5, axis='v')
    v0 -= np.min(v0)
    v0 /= np.max(v0)

    ImageViewer.alignUVVec(v0, st)

    v1 = solveUV1(image_v, st, smoothing_weight=.2, cross_weight=.7, axis='v')
    v1 -= np.min(v1)
    v1 /= np.max(v1)

    uvec, ucoh = ImageViewer.synthesizeUVecArray(v1)
    np.copyto(st.vector_u, uvec)
    # np.copyto(st.coherence, ucoh)

    u1 = solveUV1(image_u, st, smoothing_weight=.5, cross_weight=.5, axis='u')
    u1 -= np.min(u1)
    u1 /= np.max(u1)

    np.copyto(image_u, u1)
    np.copyto(image_v, v1)

def updateUV(args):
    image_u, image_v, st = args

    v1 = solveUV1(image_v, st, smoothing_weight=.2, cross_weight=.7, axis='v')
    v1 -= np.min(v1)
    v1 /= np.max(v1)

    u1 = solveUV1(image_u, st, smoothing_weight=.5, cross_weight=.5, axis='u')
    u1 -= np.min(u1)
    u1 /= np.max(u1)

    np.copyto(image_u, u1)
    np.copyto(image_v, v1)

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

class Struct: pass

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

# x-axis: 0, y-axis: 1, 45 degree: 0.5
def create_angle_array(n):
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    angles = np.arctan2(y,x)
    values = np.zeros_like(angles)
    values = (angles / (np.pi / 2))
    return values

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

    umb = np.array([4008, 2304]) # x, y
    level_start, chunk = 3, 128
    # x0, y0, w0, h0 = 3000, 3000, chunk*(2**level_start), chunk*(2**level_start)
    # x0, y0, w0, h0 = 2000, 2000, chunk*(2**level_start), chunk*(2**level_start)
    x0, y0, w0, h0 = 1000, 1000, chunk*(2**level_start), chunk*(2**level_start)

    z_scroll = zarr.open('./evol1/scroll.zarr/', mode='r')
    images, images_u, images_v = createInitUV(z_scroll, umb, x0, y0, w0, h0)
    _, images_u_, images_v_ = createInitUV(z_scroll, umb, x0, y0, w0, h0)
    images_st = []

    print('Compute Eigens ...')
    for image in tqdm(images):
        image = image.astype(np.float32) / 65535.
        st = ST(image)
        st.computeEigens()
        images_st.append(st)

    for level in reversed(range(level_start+1)):
        # if (level == level_start): continue
        if (level == 0): continue
        if (level == 1): continue
        print('Top-Down: solving level ', level)
        tasks = []

        for i in range(2**(level_start-level)):
            for j in range(2**(level_start-level)):
                w, h = chunk, chunk
                x, y = w*i, h*j

                image = images[level][y:y+h, x:x+w]
                image_u = images_u[level][y:y+h, x:x+w]
                image_v = images_v[level][y:y+h, x:x+w]

                st = Struct()
                st.vector_u = images_st[level].vector_u[y:y+h, x:x+w]
                st.vector_v = images_st[level].vector_v[y:y+h, x:x+w]
                st.coherence = images_st[level].coherence[y:y+h, x:x+w]

                tasks.append((image_u, image_v, st))

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(calculateUV, tasks), total=len(tasks)))

    for level in reversed(range(level_start+1)):
        if (level == level_start): continue
        if (level == 0): continue
        if (level == 1): continue
        print('Top-Down: solving level ', level)
        tasks = []
        num = 2**(level_start-level)+1

        for i in range(num):
            for j in range(num):
                w, h = chunk, chunk
                x, y = w*i - w//2, h*j - h//2

                if (i == 0): x = 0
                if (j == 0): y = 0
                if (i == 0 or i == num-1): w = chunk//2
                if (j == 0 or j == num-1): h = chunk//2

                image = images[level][y:y+h, x:x+w]
                image_u = images_u_[level][y:y+h, x:x+w]
                image_v = images_v_[level][y:y+h, x:x+w]

                st = Struct()
                st.vector_u = images_st[level].vector_u[y:y+h, x:x+w]
                st.vector_v = images_st[level].vector_v[y:y+h, x:x+w]
                st.coherence = images_st[level].coherence[y:y+h, x:x+w]

                tasks.append((image_u, image_v, st))

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(updateUV, tasks), total=len(tasks)))

    for level in reversed(range(level_start+1)):
        if (level == level_start): continue
        if (level == 0): continue
        if (level == 1): continue

        x, y, w, h = chunk//2, 0, chunk, chunk//2
        image_a = images_v_[level][y:y+h, x:x+w].copy()

        x, y, w, h = chunk, 0, chunk//2, chunk
        image_b = images_v[level][y:y+h, x:x+w].copy()

        x, y, w, h = chunk//2, 0, chunk//2, chunk//2
        image_a_i = image_a[y:y+h, x:x+w].copy()

        x, y, w, h = 0, 0, chunk//2, chunk//2
        image_b_i = image_b[y:y+h, x:x+w].copy()

        f = chunk//2
        image_c = (image_a_i + image_b_i) / 2
        image_b -= image_b[f-1:f,:] - image_c[f-1:f,:]
        image_a -= image_a[:,f-1:f] - image_c[:,0:1]

        mask_p = images_v[level].copy()

        x, y, w, h = chunk//2, 0, chunk//2, chunk//2
        mask_p[y:y+h, x:x+w] = image_a[:, :w]
        amin, amax = np.min(image_a[:, :w]), np.max(image_a[:, :w])
        x, y, w, h = chunk, 0, chunk//2, chunk//2
        mask_p[y:y+h, x:x+w] = image_c[:, :]
        cmin, cmax = np.min(image_c[:, :]), np.max(image_c[:, :])
        x, y, w, h = chunk, chunk//2, chunk//2, chunk//2
        mask_p[y:y+h, x:x+w] = image_b[h:, :]
        bmin, bmax = np.min(image_b[h:, :]), np.max(image_b[h:, :])

        mmin = min(amin, bmin, cmin)
        mmax = max(amax, bmax, cmax)

        x, y, w, h = chunk//2, 0, chunk//2, chunk//2
        mask_p[y:y+h, x:x+w] -= mmin
        mask_p[y:y+h, x:x+w] /= mmax - mmin
        x, y, w, h = chunk, 0, chunk//2, chunk//2
        mask_p[y:y+h, x:x+w] -= mmin
        mask_p[y:y+h, x:x+w] /= mmax - mmin
        x, y, w, h = chunk, chunk//2, chunk//2, chunk//2
        mask_p[y:y+h, x:x+w] -= mmin
        mask_p[y:y+h, x:x+w] /= mmax - mmin

        x, y, w, h = chunk//2, chunk//2, chunk, chunk//2
        image_a = images_v_[level][y:y+h, x:x+w].copy()

        x, y, w, h = chunk//2, 0, chunk//2, chunk
        image_b = images_v[level][y:y+h, x:x+w].copy()

        x, y, w, h = 0, 0, chunk//2, chunk//2
        image_a_i = image_a[y:y+h, x:x+w].copy()

        x, y, w, h = 0, chunk//2, chunk//2, chunk//2
        image_b_i = image_b[y:y+h, x:x+w].copy()

        f = chunk//2
        image_c = (image_a_i + image_b_i) / 2
        image_b -= image_b[f-1:f,:] - image_c[0:1,:]
        image_a -= image_a[:,f-1:f] - image_c[:,f-1:f]

        mask_q = images_v[level].copy()

        x, y, w, h = chunk, chunk//2, chunk//2, chunk//2
        mask_q[y:y+h, x:x+w] = image_a[:, w:]
        amin, amax = np.min(image_a[:, w:]), np.max(image_a[:, w:])
        x, y, w, h = chunk//2, chunk//2, chunk//2, chunk//2
        mask_q[y:y+h, x:x+w] = image_c[:, :]
        cmin, cmax = np.min(image_c[:, :]), np.max(image_c[:, :])
        x, y, w, h = chunk//2, 0, chunk//2, chunk//2
        mask_q[y:y+h, x:x+w] = image_b[:h, :]
        bmin, bmax = np.min(image_b[:h, :]), np.max(image_b[:h, :])

        mmin = min(amin, bmin, cmin)
        mmax = max(amax, bmax, cmax)

        x, y, w, h = chunk, chunk//2, chunk//2, chunk//2
        mask_q[y:y+h, x:x+w] -= mmin
        mask_q[y:y+h, x:x+w] /= mmax - mmin
        x, y, w, h = chunk//2, chunk//2, chunk//2, chunk//2
        mask_q[y:y+h, x:x+w] -= mmin
        mask_q[y:y+h, x:x+w] /= mmax - mmin
        x, y, w, h = chunk//2, 0, chunk//2, chunk//2
        mask_q[y:y+h, x:x+w] -= mmin
        mask_q[y:y+h, x:x+w] /= mmax - mmin

        x, y, w, h = chunk//2, 0, chunk//2, chunk//2
        mp = np.min(mask_p[y:y+h, x:x+w])
        mq = np.min(mask_q[y:y+h, x:x+w])
        op = np.max(mask_p[y:y+h, x:x+w])
        oq = np.max(mask_q[y:y+h, x:x+w])

        x, y, w, h = chunk, chunk//2, chunk//2, chunk//2
        mp = min(np.min(mask_p[y:y+h, x:x+w]), mp)
        mq = min(np.min(mask_q[y:y+h, x:x+w]), mq)
        op = max(np.max(mask_p[y:y+h, x:x+w]), op)
        oq = max(np.max(mask_q[y:y+h, x:x+w]), oq)

        # mask_p -= mp
        # mask_p /= op - mp
        # mask_p *= oq - mq
        # mask_p += mq

        cp = mask_p[chunk//2-1, chunk]
        cq = mask_q[chunk//2, chunk-1]
        mask_p += (cp+cq)/2 - cp
        mask_q += (cp+cq)/2 - cq

        angle_array = create_angle_array(chunk//2)
        mix_a = angle_array.T
        mix_b = angle_array[::-1, ::-1]

        images_v[level] = mask_p
        x, y, w, h = chunk//2, chunk//2, chunk//2, chunk//2
        images_v[level][y:y+h, x:x+w] = mask_q[y:y+h, x:x+w]

        x, y, w, h = chunk, chunk//2, chunk//2, chunk//2
        images_v[level][y:y+h, x:x+w] = mix_a * mask_p[y:y+h, x:x+w]
        images_v[level][y:y+h, x:x+w] += (1-mix_a) * mask_q[y:y+h, x:x+w]
        x, y, w, h = chunk//2, 0, chunk//2, chunk//2
        images_v[level][y:y+h, x:x+w] = mix_b * mask_p[y:y+h, x:x+w]
        images_v[level][y:y+h, x:x+w] += (1-mix_b) * mask_q[y:y+h, x:x+w]

        x, y, w, h = chunk//2, 0, chunk, chunk
        images_v[level][y:y+h, x:x+w] -= np.min(images_v[level][y:y+h, x:x+w])
        images_v[level][y:y+h, x:x+w] /= np.max(images_v[level][y:y+h, x:x+w])

        # images_v[level] = mask_p
        # images_v_[level] = mask_q

    plt.figure(figsize=(10, 4))
    show_image(images, images_u, images_v)
    plt.connect('key_press_event', lambda event: on_key(event, images, images_v_, images_v))
    plt.show()

if __name__ == '__main__':
    sys.exit(main())
