import sys
import zarr
import cmap
import cv2
import math
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
    out += basew
    return out

def solveUV_(basew, st, axis='u'):
    cross_weight = 0.7
    smoothing_weight = 0.2

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
    # NOTE multiplication by decimation factor
    if axis == 'v':
        b[:basew.size] = 1.*coh.flatten()*icw
    else:
        b[basew.size:2*basew.size] = 1.*coh.flatten()*icw
    x = solveAxEqb(A, b)
    out = x.reshape(basew.shape)
    return out

# given an array filled with radius values, align
# the structure tensor u vector with the gradient of
# the radius
def alignUVVec(rad0, st):
    uvec = st.vector_u
    shape = uvec.shape[:2]
    sparse_grad = ImageViewer.sparseGrad(shape)
    delr_flat = sparse_grad @ rad0.flatten()
    delr = delr_flat.reshape(uvec.shape)
    # print("delr", delr[iy,ix])
    dot = (uvec*delr).sum(axis=2)
    # print("dot", dot[iy,ix])
    print("dots", (dot<0).sum())
    print("not dots", (dot>=0).sum())
    st.vector_u[dot<0] *= -1
    st.vector_v[dot<0] *= -1

    # Replace vector interpolator by simple interpolator
    st.vector_u_interpolator = ST.createInterpolator(st.vector_u)
    st.vector_v_interpolator = ST.createInterpolator(st.vector_v)

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

    # theta = solveUV(image_u_ref, st, smoothing_weight=.5, axis='u')
    rad = solveUV(image_v_ref, st, smoothing_weight=.5, axis='v')

    alignUVVec(rad, st)

    img = drawLine(st, image)
    np.copyto(args[0], (img*65535).astype('uint16'))

    rad = solveUV_(rad, st, axis='v')
    rad -= np.min(rad)
    rad /= np.max(rad)

    # theta = solveUV_(theta, st, axis='u')
    # theta -= np.min(theta)
    # theta /= np.max(theta)

    # np.copyto(image_u, theta)
    np.copyto(image_v, rad)
    # np.copyto(image_v, image_v_ref)

def solveY1(basew, st, smoothing_weight, cross_weight):
    cross_weight = 0.95
    icw = 1.-cross_weight

    uvec = st.vector_u
    coh = st.coherence.copy()

    coh = coh[:,:,np.newaxis]
    wuvec = coh*uvec

    shape = wuvec.shape[:2]
    sparse_u_cross_g = ImageViewer.sparseVecOpGrad(wuvec, is_cross=True)
    sparse_u_dot_g = ImageViewer.sparseVecOpGrad(wuvec, is_cross=False)
    sparse_grad = ImageViewer.sparseGrad(shape)
    sparse_theta = ImageViewer.sparseDiagonal(shape)
    sgx, sgy = ImageViewer.sparseGrad(shape, interleave=False)
    hxx = sgx.transpose() @ sgx
    hyy = sgy.transpose() @ sgy
    hxy = sgx @ sgy

    # theta_weight = 0.0
    sparse_all = sparse.vstack((icw*sparse_u_dot_g, cross_weight*sparse_u_cross_g, smoothing_weight*hxx, smoothing_weight*hyy))
    # sparse_all = sparse.vstack((icw*sparse_u_dot_g, cross_weight*sparse_u_cross_g, theta_weight * sparse_theta, smoothing_weight*hxx, smoothing_weight*hyy))

    A = sparse_all
    # print("A", A.shape, A.dtype)

    b = np.zeros((A.shape[0]), dtype=np.float64)
    b[:basew.size] = 1.*coh.flatten()*icw
    # b[basew.size*2:basew.size*3] = basew.flatten()*theta_weight
    x = solveAxEqb(A, b)
    out = x.reshape(basew.shape)
    return out

# given a radius array, create u vectors from the
# normalized gradients of that array.
def synthesizeUVecArray(rad):
    gradx, grady = computeGrad(rad)
    uvec = np.stack((gradx, grady), axis=2)
    # print("suvec", uvec.shape)
    luvec = np.sqrt((uvec*uvec).sum(axis=2))
    lnz = luvec != 0
    # print("ll", uvec.shape, luvec.shape, lnz.shape)
    uvec[lnz] /= luvec[lnz][:,np.newaxis]
    coh = np.full(rad.shape, 1.)
    coh[:,-1] = 0
    coh[-1,:] = 0
    return uvec, coh

def computeGrad(arr):
    oldshape = arr.shape
    shape = arr.shape
    sparse_grad = ImageViewer.sparseGrad(shape)

    # NOTE division by decimation
    grad_flat = (sparse_grad @ arr.flatten())
    grad = grad_flat.reshape(shape[0], shape[1], 2)
    gradx = grad[:,:,0]
    grady = grad[:,:,1]
    return gradx, grady

def solveX0(st, rad, theta, dot_weight, smoothing_weight, theta_weight):
    uvec, coh = synthesizeUVecArray(rad)

    oldshape = rad.shape
    coh = coh[:,:,np.newaxis]
    weight = coh.copy()

    wuvec = weight*uvec
    rwuvec = wuvec
    # rwuvec = rad[:,:,np.newaxis]*wuvec

    shape = theta.shape
    sparse_grad = ImageViewer.sparseGrad(shape)
    # sparse_grad = ImageViewer.sparseGrad(shape, rad)
    sparse_u_cross_g = ImageViewer.sparseVecOpGrad(rwuvec, is_cross=True)
    sparse_u_dot_g = ImageViewer.sparseVecOpGrad(rwuvec, is_cross=False)
    sparse_theta = ImageViewer.sparseDiagonal(shape)
    # sparse_all = sparse.vstack((sparse_u_cross_g, dot_weight*sparse_u_dot_g, smoothing_weight*sparse_grad, theta_weight*sparse_theta))
    sparse_all = sparse.vstack((sparse_u_cross_g, smoothing_weight*sparse_grad))

    b_dot = np.zeros((sparse_u_dot_g.shape[0]), dtype=np.float64)
    b_cross = weight.flatten()
    b_grad = np.zeros((sparse_grad.shape[0]), dtype=np.float64)
    b_theta = theta.flatten()
    b_all = np.concatenate((b_cross, smoothing_weight*b_grad))

    x = solveAxEqb(sparse_all, b_all)
    out = x.reshape(shape)
    return out

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

def drawLine(st, image):
    wh, ww = image.shape

    dh = 15//2
    bar0 = [1, 1]
    zoom = 1
    w0i,h0i = 0, 0
    w0i -= bar0[0]
    h0i -= bar0[1]
    dhi = 2*dh/zoom
    w0i = int(math.floor(w0i/dhi))*dhi
    h0i = int(math.floor(h0i/dhi))*dhi
    w0i += bar0[0]
    h0i += bar0[1]
    w0,h0 = w0i,h0i
    dpw = np.mgrid[h0:wh:2*dh, w0:ww:2*dh].transpose(1,2,0)
    # switch from y,x to x,y coordinates
    dpw = dpw[:,:,::-1]
    # print ("dpw", dpw.shape, dpw.dtype, dpw[0,5])
    dpi = dpw
    # interpolators expect y,x ordering
    dpir = dpi[:,:,::-1]
    # print ("dpi", dpi.shape, dpi.dtype, dpi[0,5])
    uvs = st.vector_u_interpolator(dpir)
    vvs = st.vector_v_interpolator(dpir)
    # print("vvs", vvs.shape, vvs.dtype, vvs[0,5])
    # coherence = st.coherence_interpolator(dpir)
    coherence = st.linearity_interpolator(dpir)
    # testing
    # coherence[:] = .5
    # print("coherence", coherence.shape, coherence.dtype, coherence[0,5])
    linelen = 25.//2

    lvecs = linelen*vvs*coherence[:,:,np.newaxis]
    x0 = dpw
    x1 = dpw+lvecs

    lines = np.concatenate((x0,x1), axis=2)
    lines = lines.reshape(-1,1,2,2).astype(np.int32)
    # cv2.polylines(outrgb, lines, False, (255,255,0), 1)

    lvecs = linelen*uvs*coherence[:,:,np.newaxis]

    # x1 = dpw+lvecs
    x1 = dpw+.6*lvecs
    lines = np.concatenate((x0,x1), axis=2)
    lines = lines.reshape(-1,1,2,2).astype(np.int32)
    cv2.polylines(image, lines, False, (0,255,0), 1)

    xm = dpw-.5*lvecs//2
    xp = dpw+.5*lvecs//2
    lines = np.concatenate((xm,xp), axis=2)
    lines = lines.reshape(-1,1,2,2).astype(np.int32)
    # cv2.polylines(outrgb, lines, False, (0,255,0), 1)

    points = dpw.reshape(-1,1,1,2).astype(np.int32)
    cv2.polylines(image, points, True, (0,255,255), 3)

    return image

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
    x0, y0, w0, h0 = 1000, 1000, chunk*(2**level_start), chunk*(2**level_start)
    # level_start, chunk = 1, 128
    # x0, y0, w0, h0 = 1000, 1000+chunk*6, chunk*(2**level_start), chunk*(2**level_start)

    z_scroll = zarr.open('./evol1/scroll.zarr/', mode='r')
    images, images_u, images_v = createInitUV(z_scroll, umb, x0, y0, w0, h0)

    for level in reversed(range(level_start+1)):
        if (level != level_start and level != 0): continue
        print('Top-Down: solving level ', level)
        tasks = []

        for i in range(2**(level_start-level)):
            for j in range(2**(level_start-level)):
                w, h = chunk, chunk
                x, y = w * i, h * j

                image = images[level][y:y+h, x:x+w]
                image_u = images_u[level][y:y+h, x:x+w]
                image_v = images_v[level][y:y+h, x:x+w]

                # image_u_ref = images_u[level]
                # image_v_ref = images_v[level]
                image_u_ref = cv2.resize(images_u[level+1], (0, 0), fx=2, fy=2)
                image_v_ref = cv2.resize(images_v[level+1], (0, 0), fx=2, fy=2)
                image_u_ref = image_u_ref[y:y+h, x:x+w]
                image_v_ref = image_v_ref[y:y+h, x:x+w]

                tasks.append((image, image_u, image_v, image_u_ref, image_v_ref))

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(updateUV, tasks), total=len(tasks)))

    # for level in range(1):

    #     for i in range(2**(level_start-level)):
    #         for j in range(2**(level_start-level)):
    #             w, h = chunk, chunk
    #             x, y = w * i, h * j

    #             image_v = images_v[level][y:y+h, x:x+w]

    #             gradx, grady = computeGrad(image_v)
    #             # uvec = np.stack((gradx, grady), axis=2)
    #             # # print("suvec", uvec.shape)
    #             # luvec = np.sqrt((uvec*uvec).sum(axis=2))
    #             # lnz = luvec != 0
    #             # uvec[lnz] /= luvec[lnz][:,np.newaxis]

    #             images_u[level][y:y+h, x:x+w] = gradx
    #             images_v[level][y:y+h, x:x+w] = grady

    image = images_v[0]
    st = ST(image)
    st.computeEigens()

    for level in range(1):

        for i in range(2**(level_start-level)):
            for j in range(2**(level_start-level)):
                w, h = chunk, chunk
                x, y = w * i, h * j

                image = images_v[0][y:y+h, x:x+w]
                st_ = ST(image)
                st_.computeEigens()

                st.vector_u[y:y+h, x:x+w] = st_.vector_u
                st.coherence[y:y+h, x:x+w] = st_.coherence

    # st.vector_u = st.vector_u[::8, ::8, :]
    # st.coherence = st.coherence[::8, ::8]
    st.vector_u = cv2.resize(st.vector_u, (0, 0), fx=1/8, fy=1/8)
    st.coherence = cv2.resize(st.coherence, (0, 0), fx=1/8, fy=1/8)

    images_v[level_start] = solveUV(images_v[level_start], st, smoothing_weight=.5, axis='v')

    # for level in reversed(range(level_start+1)):
    #     if (level != 0): continue
    #     print('Bottom-Up: merging level ', level)

    #     tasks = []

    #     for i in range(2**(level_start-level)):
    #         for j in range(2**(level_start-level)):
    #             w, h = chunk, chunk
    #             x, y = w * i, h * j

    #             image_v = images_v[level][y:y+h, x:x+w]

    #             image_v_ref = cv2.resize(images_v[level+1], (0, 0), fx=2, fy=2)
    #             image_v_ref = image_v_ref[y:y+h, x:x+w]

    #             image_v -= np.min(image_v)
    #             image_v /= np.max(image_v)
    #             image_v *= np.max(image_v_ref) - np.min(image_v_ref)
    #             image_v += np.min(image_v_ref)

    #     for i in range(2**(level_start-level)):
    #         for j in range(2**(level_start-level)):
    #             w, h = chunk, chunk
    #             x, y = w * i, h * j

    #             image_v = images_v[level][y:y+h, x:x+w]

    #             if (i%2 == 0):
    #                 xi = images_v[level][y:y+h, x+w-1]
    #                 bi = images_v[level][y:y+h, x+w]
    #                 bi = (bi + xi) / 2
    #             else:
    #                 xi = images_v[level][y:y+h, x]
    #                 bi = images_v[level][y:y+h, x-1]
    #                 bi = (bi + xi) / 2

    #             images_v[level][y:y+h, x:x+w] = image_v + (bi - xi)

    #     for i in range(2**(level_start-level)):
    #         for j in range(2**(level_start-level)):
    #             w, h = chunk, chunk
    #             x, y = w * i, h * j

    #             image_v = images_v[level][y:y+h, x:x+w]

    #             if (j%2 == 0):
    #                 xi = images_v[level][y+h-1, x:x+w]
    #                 bi = images_v[level][y+h, x:x+w]
    #                 bi = (bi + xi) / 2
    #             else:
    #                 xi = images_v[level][y, x:x+w]
    #                 bi = images_v[level][y-1, x:x+w]
    #                 bi = (bi + xi) / 2

    #             images_v[level][y:y+h, x:x+w] = image_v + (bi - xi)

    plt.figure()

    # plt.subplot(1, 3, 1)
    # plt.imshow(images[0], cmap='gray', aspect='equal')
    # plt.axis('off')

    # plt.subplot(1, 3, 2)
    # colormap = cmap.Colormap("tab20", interpolation="nearest")
    # image = colormap(images_u[0])
    # plt.imshow(image, aspect='equal')
    # plt.axis('off')

    # plt.subplot(1, 3, 3)
    # colormap = cmap.Colormap("tab20", interpolation="nearest")
    # image = colormap(images_v[0])
    # plt.imshow(image, aspect='equal')
    # plt.axis('off')

    show_image(images, images_u, images_v)
    plt.connect('key_press_event', lambda event: on_key(event, images, images_u, images_v))
    plt.show()

if __name__ == '__main__':
    sys.exit(main())
