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

def solveAxEqb(A, b):
    # print("solving Ax = b", A.shape, b.shape)
    At = A.transpose()
    AtA = At @ A
    # print("AtA", AtA.shape, sparse.issparse(AtA))
    # print("ata", ata.shape, ata.dtype, ata[ata!=0], np.argwhere(ata))
    # asum = np.abs(AtA).sum(axis=0)
    # print("asum", np.argwhere(asum==0))
    Atb = At @ b
    # print("Atb", Atb.shape, sparse.issparse(Atb))
    N = AtA.shape[0]
    AtA = AtA + sparse.eye(N, format='csc') * 1e-6

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

def solveUV1(basew, st, axis='u'):
    h, w = basew.shape
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    uvec = np.mean(st.vector_u, axis=(0,1))
    if axis == 'v':
        out = uvec[0] * x + uvec[1] * y
    else:
        out = -uvec[1] * x + uvec[0] * y
    out -= np.min(out)
    out /= np.max(out)
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

def updateST(args):
    image_v, st = args

    v0 = solveUV0(image_v, st, smoothing_weight=.5, axis='v')
    if (np.min(v0) != np.max(v0)):
        v0 -= np.min(v0)
        v0 /= np.max(v0)

    ImageViewer.alignUVVec(v0, st)

    v1 = solveUV1(image_v, st, axis='v')
    if (np.min(v1) != np.max(v1)):
        v1 -= np.min(v1)
        v1 /= np.max(v1)

    uvec, ucoh = ImageViewer.synthesizeUVecArray(v1)
    np.copyto(st.vector_u, uvec)
    # np.copyto(st.coherence, ucoh)
    np.copyto(image_v, v1)

def updateU(args):
    image_u, st = args
    u1 = solveUV1(image_u, st, axis='u')
    np.copyto(image_u, u1)

def updateV(args):
    image_v, st = args
    v1 = solveUV1(image_v, st, axis='v')
    np.copyto(image_v, v1)

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

def show_image(image, image_u, image_v):
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray', aspect='equal')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    colormap = cmap.Colormap("tab20", interpolation="nearest")
    image = colormap(image_u)
    plt.imshow(image, aspect='equal')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    colormap = cmap.Colormap("tab20", interpolation="nearest")
    image = colormap(image_v)
    plt.imshow(image, aspect='equal')
    plt.axis('off')

    plt.draw()

def merge_chunk(chunk_v, chunk_h):
    height, width = chunk_v.shape

    # vertical rectangle (left)
    x, y, w, h = 0, 0, width//2, height
    rect_v = chunk_v[y:y+h, x:x+w]
    # horizontal rectangle (top)
    x, y, w, h = 0, 0, width, height//2
    rect_h = chunk_h[y:y+h, x:x+w]
    # merge the cross region
    l_left_top = merge_rectangle(rect_v, rect_h)

    # vertical rectangle (right)
    x, y, w, h = width//2, 0, width//2, height
    rect_v = chunk_v[y:y+h, x:x+w]
    # horizontal rectangle (bottom)
    x, y, w, h = 0, height//2, width, height//2
    rect_h = chunk_h[y:y+h, x:x+w]
    # merge the cross region (reversed)
    l_right_bottom = merge_rectangle(rect_v[::-1, ::-1], rect_h[::-1, ::-1])
    l_right_bottom = l_right_bottom[::-1, ::-1]

    chunk_m = merge_Lshape(l_left_top, l_right_bottom)

    return chunk_m

def merge_rectangle(rect_v, rect_h):
    height, width = rect_v.shape[0], rect_h.shape[1]
    rect_m = np.zeros((height, width))

    # top-left
    x, y, w, h = 0, 0, width//2, height//2
    region0 = (rect_v[y:y+h, x:x+w] + rect_h[y:y+h, x:x+w]) / 2
    rect_m[y:y+h, x:x+w] = region0
    # top-right
    x, y, w, h = width//2, 0, width//2, height//2
    region1 = rect_h[y:y+h, x:x+w]
    region1 += region0[:, x-1:x] - rect_h[:, x:x+1]
    rect_m[y:y+h, x:x+w] = region1
    # bottom-left
    x, y, w, h = 0, height//2, width//2, height//2
    region2 = rect_v[y:y+h, x:x+w]
    region2 += region0[y-1:y, :] - rect_v[y:y+1, :]
    rect_m[y:y+h, x:x+w] = region2

    # normalize
    value_min = min(np.min(region0), np.min(region1), np.min(region2))
    value_max = max(np.max(region0), np.max(region1), np.max(region2))
    if (value_max != value_min):
        rect_m -= value_min
        rect_m /= value_max - value_min
    return rect_m

def merge_Lshape(l_left_top, l_right_bottom):
    height, width = l_left_top.shape
    chunk = np.zeros((height, width))

    # center calibration
    c_lt = l_left_top[height//2-1, width//2-1]
    c_rb = l_right_bottom[height//2, width//2]
    l_left_top += (c_rb - c_lt) / 2
    l_right_bottom += (c_lt - c_rb) / 2

    angle_array = create_angle_array(height//2)
    angle_region1 = angle_array.copy()[::-1, :]
    angle_region2 = angle_array.copy()[::-1, :].T

    # region 0 (left, top)
    x, y, w, h = 0, 0, width//2, height//2
    region0 = l_left_top[y:y+h, x:x+w]
    chunk[y:y+h, x:x+w] = region0
    # region 1 mixed (right, top)
    x, y, w, h = width//2, 0, width//2, height//2
    region1 = l_left_top[y:y+h, x:x+w] * angle_region1
    region1 += l_right_bottom[y:y+h, x:x+w] * (1 - angle_region1)
    chunk[y:y+h, x:x+w] = region1
    # region 2 mixed (left, bottom)
    x, y, w, h = 0, height//2, width//2, height//2
    region2 = l_left_top[y:y+h, x:x+w] * angle_region2
    region2 += l_right_bottom[y:y+h, x:x+w] * (1 - angle_region2)
    chunk[y:y+h, x:x+w] = region2
    # region 3 (right, bottom)
    x, y, w, h = width//2, height//2, width//2, height//2
    region3 = l_right_bottom[y:y+h, x:x+w]
    chunk[y:y+h, x:x+w] = region3

    # normalize
    if (np.min(chunk) != np.max(chunk)):
        chunk -= np.min(chunk)
        chunk /= np.max(chunk)
    return chunk

# x-axis: 0, y-axis: 1, 45 degree: 0.5
def create_angle_array(n):
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    angles = np.arctan2(y,x)
    values = np.zeros_like(angles)
    values = (angles / (np.pi / 2))
    return values

def merge_bridge(b_left, b_middle, b_right):
    height, width_bl = b_left.shape
    height, width_bm = b_middle.shape
    height, width_br = b_right.shape
    chunk = np.zeros((height, width_bl + width_br))

    # left
    x, y, w, h = 0, 0, width_bl, height
    b_left += b_middle[:, 0:1] - b_left[:, -width_bm//2-1:-width_bm//2]
    chunk[y:y+h, x:x+w] = b_left
    # right
    x, y, w, h = width_bl, 0, width_br, height
    b_right += b_middle[:, -2:-1] - b_right[:, width_bm//2-1:width_bm//2]
    chunk[y:y+h, x:x+w] = b_right
    # middle
    x, y, w, h = width_bl - width_bm//2, 0, width_bm, height
    chunk[y:y+h, x:x+w] = b_middle
    return chunk

def merge_window(window_o, window_p):
    chunk = window_o.shape[0] // 2
    chunk_hh = np.zeros((chunk*2, chunk*2))
    chunk_vv = np.zeros((chunk*2, chunk*2))

    x, y, w, h = chunk//2, 0, chunk, chunk
    chunk_v = window_o[y:y+h, x:x+w].copy()
    chunk_h = window_p[y:y+h, x:x+w].copy()
    b_middle = merge_chunk(chunk_v, chunk_h)

    x, y, w, h = 0, 0, chunk, chunk
    b_left = window_o[y:y+h, x:x+w].copy()
    x, y, w, h = chunk, 0, chunk, chunk
    b_right = window_o[y:y+h, x:x+w].copy()

    x, y, w, h = 0, 0, chunk*2, chunk
    chunk_hh[y:y+h, x:x+w] = merge_bridge(b_left, b_middle, b_right)

    x, y, w, h = chunk//2, chunk, chunk, chunk
    chunk_v = window_o[y:y+h, x:x+w].copy()
    chunk_h = window_p[y:y+h, x:x+w].copy()
    b_middle = merge_chunk(chunk_v, chunk_h)

    x, y, w, h = 0, chunk, chunk, chunk
    b_left = window_o[y:y+h, x:x+w].copy()
    x, y, w, h = chunk, chunk, chunk, chunk
    b_right = window_o[y:y+h, x:x+w].copy()

    x, y, w, h = 0, chunk, chunk*2, chunk
    chunk_hh[y:y+h, x:x+w] = merge_bridge(b_left, b_middle, b_right)

    x, y, w, h = 0, chunk//2, chunk, chunk
    chunk_v = window_p[y:y+h, x:x+w].copy()
    chunk_h = window_o[y:y+h, x:x+w].copy()
    b_middle = merge_chunk(chunk_v, chunk_h)

    x, y, w, h = 0, 0, chunk, chunk
    b_left = window_o[y:y+h, x:x+w].copy()
    x, y, w, h = 0, chunk, chunk, chunk
    b_right = window_o[y:y+h, x:x+w].copy()

    x, y, w, h = 0, 0, chunk, chunk*2
    chunk_vv[y:y+h, x:x+w] = merge_bridge(b_left.T, b_middle.T, b_right.T).T

    x, y, w, h = chunk, chunk//2, chunk, chunk
    chunk_v = window_p[y:y+h, x:x+w].copy()
    chunk_h = window_o[y:y+h, x:x+w].copy()
    b_middle = merge_chunk(chunk_v, chunk_h)

    x, y, w, h = chunk, 0, chunk, chunk
    b_left = window_o[y:y+h, x:x+w].copy()
    x, y, w, h = chunk, chunk, chunk, chunk
    b_right = window_o[y:y+h, x:x+w].copy()

    x, y, w, h = chunk, 0, chunk, chunk*2
    chunk_vv[y:y+h, x:x+w] = merge_bridge(b_left.T, b_middle.T, b_right.T).T
    return merge_chunk(chunk_vv, chunk_hh)

def merge_level(image_o, image_p, n):
    image_oo = image_o.copy()
    image_pp = image_o.copy()
    size = image_o.shape[0]
    chunk = size//n

    for i in range(n):
        for j in range(n):
            x, y, w, h = i*chunk, j*chunk, chunk, chunk
            image_oo[y:y+h, x:x+w] = merge_window(image_o[y:y+h, x:x+w], image_p[y:y+h, x:x+w])

    for i in range(n-1):
        x, y, w, h = (i+1)*chunk//2, 0, chunk, chunk
        image_pp[y:y+h, x:x+w] = merge_window(image_o[y:y+h, x:x+w], image_p[y:y+h, x:x+w])
        x, y, w, h = (i+1)*chunk//2, size-chunk, chunk, chunk
        image_pp[y:y+h, x:x+w] = merge_window(image_o[y:y+h, x:x+w], image_p[y:y+h, x:x+w])
        x, y, w, h = 0, (i+1)*chunk//2, chunk, chunk
        image_pp[y:y+h, x:x+w] = merge_window(image_o[y:y+h, x:x+w], image_p[y:y+h, x:x+w])
        x, y, w, h = size-chunk, (i+1)*chunk//2, chunk, chunk
        image_pp[y:y+h, x:x+w] = merge_window(image_o[y:y+h, x:x+w], image_p[y:y+h, x:x+w])

    for i in range(n-1):
        for j in range(n-1):
            x, y, w, h = (i+1)*chunk//2, (j+1)*chunk//2, chunk, chunk
            image_pp[y:y+h, x:x+w] = merge_window(image_o[y:y+h, x:x+w], image_p[y:y+h, x:x+w])

    return image_oo, image_pp

def merge_split(image_o, image_p, split):
    image_oo = image_o.copy()
    image_pp = image_o.copy()

    while split > 1:
        split = split // 2
        image_oo, image_pp = merge_level(image_oo, image_pp, split)

    return image_oo

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

    level = 3
    num_plot = 7
    decimation = 2**level
    umb = np.array([3800, 2304]) // decimation
    z_scroll = zarr.open('./evol1/scroll.zarr/', mode='r')
    z_scroll_u = zarr.open('./evol1/scroll_u.zarr/', mode='a')
    z_scroll_v = zarr.open('./evol1/scroll_v.zarr/', mode='a')

    # ni, nj, n, chunk = 8*12, 10*12, 4, 50
    # ni, nj, n, chunk = 8*12, 10*12, 16, 12
    # ni, nj, n, chunk = 100, 100, 32, 12
    # ni, nj, n, chunk = 4*12, 10*12, 32, 12
    # ni, nj, n, chunk = 4*12, 10*12, 64, 6
    n, chunk = 32, 12
    ni, nj = umb[0] - chunk*n//2, umb[1] - chunk*n//2

    w0, h0 = chunk*n, chunk*n
    x0, y0 = 128//decimation + ni, 0//decimation + nj

    image = z_scroll[level][0, y0:y0+h0, x0:x0+w0]
    image_uo = createThetaArray(umb, x0, y0, w0, h0)
    image_vo = createRadiusArray(umb, x0, y0, w0, h0)
    image_up = image_uo.copy()
    image_vp = image_vo.copy()
    theta = image_uo.copy()

    print('Compute Eigens ...')
    image = image.astype(np.float32) / 65535.
    st = ST(image)
    st.computeEigens()

    print('Compute image_uo, image_vo')
    u_tasks, v_tasks = [], []

    for i in range(n):
        for j in range(n):
            w, h = chunk, chunk
            x, y = w*i, h*j

            image_u = image_uo[y:y+h, x:x+w]
            image_v = image_vo[y:y+h, x:x+w]

            image_st = Struct()
            image_st.vector_u = st.vector_u[y:y+h, x:x+w]
            image_st.vector_v = st.vector_v[y:y+h, x:x+w]
            image_st.coherence = st.coherence[y:y+h, x:x+w]
            image_st.isotropy = st.isotropy[y:y+h, x:x+w]

            u_tasks.append((image_u, image_st))
            v_tasks.append((image_v, image_st))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(updateST, v_tasks), total=len(v_tasks)))
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(updateU, u_tasks), total=len(u_tasks)))

    print('Compute image_up, image_vp')
    u_tasks, v_tasks = [], []

    for i in range(n+1):
        for j in range(n+1):
            w, h = chunk, chunk
            x, y = w*i - w//2, h*j - h//2

            if (i == 0): x = 0
            if (j == 0): y = 0
            if (i == 0 or i == n): w = chunk//2
            if (j == 0 or j == n): h = chunk//2

            image_u = image_up[y:y+h, x:x+w]
            image_v = image_vp[y:y+h, x:x+w]

            image_st = Struct()
            image_st.vector_u = st.vector_u[y:y+h, x:x+w]
            image_st.vector_v = st.vector_v[y:y+h, x:x+w]
            image_st.coherence = st.coherence[y:y+h, x:x+w]
            image_st.isotropy = st.isotropy[y:y+h, x:x+w]

            u_tasks.append((image_u, image_st))
            v_tasks.append((image_v, image_st))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(updateU, u_tasks), total=len(u_tasks)))
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(updateV, v_tasks), total=len(v_tasks)))

    plt.figure(figsize=(13, 4))
    colormap = cmap.Colormap("tab20", interpolation="nearest")

    plt.subplot(1, num_plot, 1)
    plt.imshow(image, cmap='gray', aspect='equal')
    plt.axis('off')

    plt.subplot(1, num_plot, 2)
    plt.imshow(colormap(image_uo), aspect='equal')
    plt.axis('off')

    plt.subplot(1, num_plot, 3)
    plt.imshow(colormap(image_up), aspect='equal')
    plt.axis('off')

    plt.subplot(1, num_plot, 4)
    plt.imshow(colormap(image_vo), aspect='equal')
    plt.axis('off')

    plt.subplot(1, num_plot, 5)
    plt.imshow(colormap(image_vp), aspect='equal')
    plt.axis('off')

    print('merge image_vo & image_vp')

    print('merge image_uo & image_up')
    split = n // 2
    x, y, w, h = 0, 0, chunk*n//2, chunk*n//2
    t_left = merge_split(image_uo[y:y+h, x:x+w], image_up[y:y+h, x:x+w], split)

    split = n // 2
    x, y, w, h = chunk*n//4, 0, chunk*n//2, chunk*n//2
    t_mid = merge_split(image_uo[y:y+h, x:x+w], image_up[y:y+h, x:x+w], split)

    split = n // 2
    x, y, w, h = chunk*n//2, 0, chunk*n//2, chunk*n//2
    t_right = merge_split(image_uo[y:y+h, x:x+w], image_up[y:y+h, x:x+w], split)

    split = n // 2
    x, y, w, h = 0, chunk*n//2, chunk*n//2, chunk*n//2
    b_left = merge_split(image_uo[y:y+h, x:x+w], image_up[y:y+h, x:x+w], split)

    split = n // 2
    x, y, w, h = chunk*n//4, chunk*n//2, chunk*n//2, chunk*n//2
    b_mid = merge_split(image_uo[y:y+h, x:x+w], image_up[y:y+h, x:x+w], split)

    split = n // 2
    x, y, w, h = chunk*n//2, chunk*n//2, chunk*n//2, chunk*n//2
    b_right = merge_split(image_uo[y:y+h, x:x+w], image_up[y:y+h, x:x+w], split)

    split = n // 2
    x, y, w, h = chunk*n//2, 0, chunk*n//2, chunk*n//2
    r_top = merge_split(image_uo[y:y+h, x:x+w], image_up[y:y+h, x:x+w], split)

    split = n // 2
    x, y, w, h = chunk*n//2, chunk*n//4, chunk*n//2, chunk*n//2
    r_mid = merge_split(image_uo[y:y+h, x:x+w], image_up[y:y+h, x:x+w], split)

    split = n // 2
    x, y, w, h = chunk*n//2, chunk*n//2, chunk*n//2, chunk*n//2
    r_bot = merge_split(image_uo[y:y+h, x:x+w], image_up[y:y+h, x:x+w], split)

    x, y, w, h = 0, 0, chunk*n, chunk*n//2
    image_uo[y:y+h, x:x+w] = merge_bridge(t_left, t_mid, t_right)
    image_uo[y:y+h, x:x+w] -= np.min(image_uo[y:y+h, x:x+w])
    image_uo[y:y+h, x:x+w] /= np.max(image_uo[y:y+h, x:x+w])

    x, y, w, h = 0, chunk*n//2, chunk*n, chunk*n//2
    image_uo[y:y+h, x:x+w] = merge_bridge(b_left, b_mid, b_right)
    image_uo[y:y+h, x:x+w] -= np.min(image_uo[y:y+h, x:x+w])
    image_uo[y:y+h, x:x+w] /= np.max(image_uo[y:y+h, x:x+w])

    x, y, w, h = chunk*n//2, 0, chunk*n//2, chunk*n
    image_vo[y:y+h, x:x+w] = merge_bridge(r_top.T, r_mid.T, r_bot.T).T
    image_vo[y:y+h, x:x+w] -= np.min(image_vo[y:y+h, x:x+w])
    image_vo[y:y+h, x:x+w] /= np.max(image_vo[y:y+h, x:x+w])

    plt.subplot(1, num_plot, 6)
    plt.imshow(colormap(image_uo), aspect='equal')
    plt.axis('off')

    plt.subplot(1, num_plot, 7)
    plt.imshow(colormap(image_vo), aspect='equal')
    plt.axis('off')

    plt.show()

    # # save u, v result
    # z_scroll_u[level][0, y0:y0+h0, x0:x0+w0] = (image_uo * 65535).astype(np.uint16)
    # z_scroll_v[level][0, y0:y0+h0, x0:x0+w0] = (image_vo * 65535).astype(np.uint16)

if __name__ == '__main__':
    sys.exit(main())
