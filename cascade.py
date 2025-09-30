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

    v1 = solveUV1(image_v, st, smoothing_weight=.2, cross_weight=.7, axis='v')
    if (np.min(v1) != np.max(v1)):
        v1 -= np.min(v1)
        v1 /= np.max(v1)

    uvec, ucoh = ImageViewer.synthesizeUVecArray(v1)
    np.copyto(st.vector_u, uvec)
    # np.copyto(st.coherence, ucoh)
    np.copyto(image_v, v1)

def updateU(args):
    image_u, st = args

    # u1 = solveUV0(image_u, st, smoothing_weight=.5, axis='u')
    u1 = solveUV1(image_u, st, smoothing_weight=.1, cross_weight=.5, axis='u')
    if (np.min(u1) != np.max(u1)):
        u1 -= np.min(u1)
        u1 /= np.max(u1)
    np.copyto(image_u, u1)

def updateV(args):
    image_v, st = args

    v1 = solveUV1(image_v, st, smoothing_weight=.2, cross_weight=.7, axis='v')
    if (np.min(v1) != np.max(v1)):
        v1 -= np.min(v1)
        v1 /= np.max(v1)
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

    level, n, chunk = 3, 8, 100
    decimation = 2**level
    w0, h0 = chunk*n, chunk*n
    x0, y0 = 128//decimation, 0//decimation
    umb = np.array([3900, 2304]) // decimation
    z_scroll = zarr.open('./evol1/scroll.zarr/', mode='r')
    z_scroll_u = zarr.open('./evol1/scroll_u.zarr/', mode='a')
    z_scroll_v = zarr.open('./evol1/scroll_v.zarr/', mode='a')

    image = z_scroll[level][0, y0:y0+h0, x0:x0+w0]
    image_uo = createThetaArray(umb, x0, y0, w0, h0)
    image_vo = createRadiusArray(umb, x0, y0, w0, h0)
    image_up = image_uo.copy()
    image_vp = image_vo.copy()
    theta = image_uo.copy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray', aspect='equal')
    plt.axis('off')

    print('Compute Eigens ...')
    image = image.astype(np.float32) / 65535.
    st = ST(image)
    st.computeEigens()

    print('Radius Calculation ...')
    print('image_vo: solving level ', level)
    tasks = []

    for i in range(n):
        for j in range(n):
            w, h = chunk, chunk
            x, y = w*i, h*j

            image_v = image_vo[y:y+h, x:x+w]

            image_st = Struct()
            image_st.vector_u = st.vector_u[y:y+h, x:x+w]
            image_st.vector_v = st.vector_v[y:y+h, x:x+w]
            image_st.coherence = st.coherence[y:y+h, x:x+w]

            tasks.append((image_v, image_st))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(updateST, tasks), total=len(tasks)))

    print('image_vp: solving level ', level)
    tasks = []

    for i in range(n+1):
        for j in range(n+1):
            w, h = chunk, chunk
            x, y = w*i - w//2, h*j - h//2

            if (i == 0): x = 0
            if (j == 0): y = 0
            if (i == 0 or i == n): w = chunk//2
            if (j == 0 or j == n): h = chunk//2

            image_v = image_vp[y:y+h, x:x+w]

            image_st = Struct()
            image_st.vector_u = st.vector_u[y:y+h, x:x+w]
            image_st.vector_v = st.vector_v[y:y+h, x:x+w]
            image_st.coherence = st.coherence[y:y+h, x:x+w]

            tasks.append((image_v, image_st))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(updateV, tasks), total=len(tasks)))

    print('merge image_vo & image_vp')
    image_vo, image_vp = merge_level(image_vo, image_vp, 4)
    image_vo, image_vp = merge_level(image_vo, image_vp, 2)
    image_vo, image_vp = merge_level(image_vo, image_vp, 1)

    plt.subplot(1, 3, 2)
    colormap = cmap.Colormap("tab20", interpolation="nearest")
    image_v = colormap(image_vo)
    plt.imshow(image_v, aspect='equal')
    plt.axis('off')

    xp, yp = umb[0]-x0, umb[1]-y0
    H, W = image_vo.shape

    inertia = 0.97
    # inertia = 0.95
    radius, step_size, max_steps = 50, 10, 10000
    center = np.array([yp, xp], dtype=np.float32)
    angles = np.linspace(0, 2*np.pi, 100, endpoint=False)
    all_points, all_angles = [], []

    # edge interpolate handling (still a bit messy)
    lmin, lmax, rmin, rmax = H//2, H//2, H//2, H//2
    tmin, tmax, bmin, bmax = W//2, W//2, W//2, W//2
    angle_lmin, angle_lmax, angle_rmin, angle_rmax = 0, 0, 0, 0
    angle_tmin, angle_tmax, angle_bmin, angle_bmax = 0, 0, 0, 0

    print('Theta Calculation ...')
    # plt.subplot(1, 3, 3)
    # plt.imshow(image_vo, aspect='equal', cmap='gray')
    # plt.scatter([xp], [yp], color='green')
    # plt.axis('off')

    for angle in angles:
        path = []
        pos = center.copy()
        sin, cos = np.sin(angle), np.cos(angle)
        prev_dir = np.array([sin, cos])

        for i in range(1, 10):
            pos += prev_dir * radius * 0.1
            path.append(pos.copy())
            all_points.append(pos.copy())
            all_angles.append(angle)

        for _ in range(max_steps):
            y, x = int(round(pos[0])), int(round(pos[1]))
            if not (0 <= y < H and 0 <= x < W): break
            field_dir = st.vector_u[y, x][::-1]
            if np.linalg.norm(field_dir) == 0:
                field_dir = prev_dir
            else:
                field_dir = field_dir / np.linalg.norm(field_dir)
            curr_dir = inertia * prev_dir + (1 - inertia) * field_dir
            pos += curr_dir * step_size
            prev_dir = curr_dir

            # edge handling
            finish = False
            if not (0 <= pos[0] < H and 0 <= pos[1] < W):
                if not (0 <= pos[0]):
                    if (tmin > x): angle_tmin, tmin = angle, x
                    if (tmax < x): angle_tmax, tmax = angle, x
                if not (pos[0] < H):
                    if (bmin > x): angle_bmin, bmin = angle, x
                    if (bmax < x): angle_bmax, bmax = angle, x
                if not (0 <= pos[1]):
                    if (lmin > y): angle_lmin, lmin = angle, y
                    if (lmax < y): angle_lmax, lmax = angle, y
                if not (pos[1] < W):
                    if (rmin > y): angle_rmin, rmin = angle, y
                    if (rmax < y): angle_rmax, rmax = angle, y

                # pos[0] = np.clip(y, 0, H - 1)
                # pos[1] = np.clip(x, 0, W - 1)
                finish = True

            path.append(pos.copy())
            all_points.append(pos.copy())
            all_angles.append(angle)
            if finish: break

        path = np.array(path)
        # plt.plot(path[:, 1], path[:, 0], lw=1)

    # left top corner
    angle_lt = lmin * angle_lmin + tmin * angle_tmin
    angle_lt /= lmin + tmin
    all_points.append(np.array([0, 0]))
    all_angles.append(angle_lt)
    # left bottom corner
    angle_lb = lmax * angle_lmax + bmin * angle_bmin
    angle_lb /= lmax + bmin
    all_points.append(np.array([H-1, 0]))
    all_angles.append(angle_lb)
    # right top corner
    angle_rt = rmin * angle_rmin + tmax * angle_tmax
    angle_rt /= rmin + tmax
    all_points.append(np.array([0, W-1]))
    all_angles.append(angle_rt)
    # right bottom corner
    angle_rb = rmax * angle_rmax + bmax * angle_bmax
    angle_rb /= rmax + bmax
    all_points.append(np.array([H-1, W-1]))
    all_angles.append(angle_rb)

    all_points = np.array(all_points)
    all_angles = np.array(all_angles)

    ys, xs = np.mgrid[0:H, 0:W]
    grid_pos = np.stack([ys.ravel(), xs.ravel()], axis=-1)   # (H*W, 2)

    all_sin, all_cos = np.sin(all_angles), np.cos(all_angles)
    image_sin = griddata(all_points, all_sin, grid_pos, method='linear')
    image_cos = griddata(all_points, all_cos, grid_pos, method='linear')
    image_uo = np.arctan2(image_sin, image_cos).reshape(H, W)
    image_uo = np.nan_to_num(image_uo, nan=0.0)
    image_uo -= np.min(image_uo)
    image_uo /= np.max(image_uo)

    plt.subplot(1, 3, 3)
    colormap = cmap.Colormap("tab20", interpolation="nearest")
    image_u = colormap(image_uo)
    plt.imshow(image_u, aspect='equal')
    plt.axis('off')

    plt.show()

    # save u, v result
    z_scroll_u[level][0, y0:y0+h0, x0:x0+w0] = (image_uo * 65535).astype(np.uint16)
    z_scroll_v[level][0, y0:y0+h0, x0:x0+w0] = (image_vo * 65535).astype(np.uint16)

if __name__ == '__main__':
    sys.exit(main())
