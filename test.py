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

# ymin = a*xmin + b, ymax = a*xmax + b
def scale_shift(x, y):
    l = x.flatten().shape[0]
    xmax, xmin = np.max(x), np.min(x)
    ymax, ymin = np.max(y), np.min(y)
    if (xmax - xmin > 0.005 * l):
        a = (ymax - ymin) / (xmax - xmin)
        b = (ymin * xmax - ymax * xmin) / (xmax - xmin)
    else:
        a = 1
        b = (ymax + ymin) / 2 - (xmax + xmin) / 2
    return a, b

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

# def on_key(event, images, images_u, images_v):
#     global current_level
#     if event.key == ' ':  # space press
#         current_level += 1
#         if current_level >= len(images):
#             plt.close()
#         else:
#             plt.clf()
#             show_image(images, images_u, images_v)

class Struct: pass

def merge_chunk(image_h, image_v, debug=False, mode=0):
    chunk_h, chunk_v = image_h.copy(), image_v.copy()
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

    if debug:
        if mode==0: return chunk_h
        if mode==1: return chunk_v
        if mode==2: return l_left_top
        if mode==3: return l_right_bottom

    chunk_m = merge_Lshape(l_left_top, l_right_bottom)
    return chunk_m

def merge_rectangle(image_v, image_h):
    rect_v, rect_h = image_v.copy(), image_h.copy()
    height, width = rect_v.shape[0], rect_h.shape[1]
    ws, hs = rect_v.shape[1], rect_h.shape[0]
    rect_m = np.zeros((height, width))

    c1 = rect_v[hs-1, ws-1]
    c2 = rect_h[hs-1, ws-1]
    c = (c1 + c2) / 2
    rect_v -= c1 + c
    rect_h -= c2 + c

    # xs, ys = np.linspace(0, 1, width-ws+1), np.linspace(0, 1, height-hs+1)

    # top-left (region0)
    x, y, w, h = 0, 0, ws, hs
    # region0 = (rect_v[y:y+h, x:x+w] + rect_h[y:y+h, x:x+w]) / 2
    # angle_array = 0
    angle_array = create_angle_array(ws)[::-1,::-1]
    region0 = (rect_v[y:y+h, x:x+w] * (1-angle_array) + rect_h[y:y+h, x:x+w] * angle_array)
    rect_m[y:y+h, x:x+w] = region0
    # top-right (region1)
    x, y, w, h = ws-1, 0, width-ws+1, hs
    a, b = scale_shift(rect_h[:, x:x+1], region0[:, -1:])
    # rect_h = a * rect_h + b
    diff1 = region0[:, -1:] - rect_h[:, x:x+1]
    rect_m[y:y+h, x:x+w] = rect_h[y:y+h, x:x+w] + diff1 * 0.0
    # bottom-left (region2)
    x, y, w, h = 0, hs-1, ws, height-hs+1
    a, b = scale_shift(rect_v[y:y+1, :], region0[-1:, :])
    # rect_v = a * rect_v + b
    diff2 = region0[-1:, :] - rect_v[y:y+1, :]
    rect_m[y:y+h, x:x+w] = rect_v[y:y+h, x:x+w] + diff2 * 0.0

    angle_array = create_angle_array(ws)[::-1,::-1]
    region11 = (region0 - diff1 * 0.5) * angle_array
    region22 = (region0 - diff2 * 0.5) * (1 - angle_array)
    x, y, w, h = 0, 0, ws, hs
    # rect_m[y:y+h, x:x+w] = region11 + region22
    return rect_m

def merge_Lshape(l_left_top, l_right_bottom):
    h0, w0 = l_left_top.shape
    chunk = np.zeros((h0, w0))

    # center calibration
    c_lt = l_left_top[h0//2, w0//2-1] + l_left_top[h0//2-1, w0//2]
    c_lt /= 2
    c_rb = l_right_bottom[h0//2, w0//2-1] + l_right_bottom[h0//2-1, w0//2]
    c_rb /= 2
    l_left_top -= c_lt
    l_right_bottom -= c_rb
    # l_left_top += (c_rb - c_lt) / 2
    # l_right_bottom += (c_lt - c_rb) / 2

    # scale calibration
    x, y, w, h = w0//2, 0, w0//2, h0//2
    a = l_left_top[y:y+h, x:x+w]
    b = l_right_bottom[y:y+h, x:x+w]
    x, y, w, h = 0, h0//2, w0//2, h0//2
    c = l_left_top[y:y+h, x:x+w]
    d = l_right_bottom[y:y+h, x:x+w]
    # scale factor
    lt_max = max(np.max(a), np.max(c))
    lt_min = min(np.min(a), np.min(c))
    rb_max = max(np.max(b), np.max(d))
    rb_min = min(np.min(b), np.min(d))
    scale = (lt_max - lt_min) / (rb_max - rb_min + 1e-7)
    # l_right_bottom = l_right_bottom * scale

    angle_array = create_angle_array(h0//2)
    angle_region1 = angle_array.copy()[::-1, :]
    angle_region2 = angle_array.copy()[::-1, :].T
 
    # region 0 (left, top)
    x, y, w, h = 0, 0, w0//2, h0//2
    region0 = l_left_top[y:y+h, x:x+w]
    chunk[y:y+h, x:x+w] = region0
    # region 1 mixed (right, top)
    x, y, w, h = w0//2, 0, w0//2, h0//2
    region1 = l_left_top[y:y+h, x:x+w] * angle_region1
    region1 += l_right_bottom[y:y+h, x:x+w] * (1 - angle_region1)
    chunk[y:y+h, x:x+w] = region1
    # region 2 mixed (left, bottom)
    x, y, w, h = 0, h0//2, w0//2, h0//2
    region2 = l_left_top[y:y+h, x:x+w] * angle_region2
    region2 += l_right_bottom[y:y+h, x:x+w] * (1 - angle_region2)
    chunk[y:y+h, x:x+w] = region2
    # region 3 (right, bottom)
    x, y, w, h = w0//2, h0//2, w0//2, h0//2
    region3 = l_right_bottom[y:y+h, x:x+w]
    chunk[y:y+h, x:x+w] = region3
    return chunk

# x-axis: 0, y-axis: 1, 45 degree: 0.5
def create_angle_array(n):
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    angles = np.arctan2(y,x)
    values = np.zeros_like(angles)
    values = (angles / (np.pi / 2))
    return values

def merge_bridge(bl, bm, br, shift, debug=False):
    b_left, b_middle, b_right = bl.copy(), bm.copy(), br.copy()

    height, width_bl = b_left.shape
    height, width_bm = b_middle.shape
    height, width_br = b_right.shape
    chunk = np.zeros((height, width_bl + width_br))

    if debug:
        x, y, w, h = 0, 0, width_bl, height
        chunk[y:y+h, x:x+w] = b_left
        x, y, w, h = width_bl, 0, width_br, height
        chunk[y:y+h, x:x+w] = b_right
        x, y, w, h = shift, 0, width_bm, height
        chunk[y:y+h, x:x+w] = b_middle
        return chunk

    # left
    x, y, w, h = 0, 0, width_bl, height
    m_edge = b_middle[:, 0:1]
    l_edge = b_left[:, shift:shift+1]
    a, b = scale_shift(l_edge, m_edge)
    # b_left = a * b_left + b
    l_edge = b_left[:, shift:shift+1]
    chunk[y:y+h, x:x+w] = b_left + (m_edge - l_edge)

    # right
    x, y, w, h = width_bl, 0, width_br, height
    sh = shift + width_bm - width_bl - 1
    m_edge = b_middle[:, -1:]
    r_edge = b_right[:, sh:sh+1]
    a, b = scale_shift(r_edge, m_edge)
    # b_right = a * b_right + b
    r_edge = b_right[:, sh:sh+1]
    chunk[y:y+h, x:x+w] = b_right + (m_edge - r_edge)

    # middle
    x, y, w, h = shift, 0, width_bm, height
    chunk[y:y+h, x:x+w] = b_middle
    return chunk

# align A to B
def align(chunk_a, chunk_b):
    chunk = chunk_a.copy()
    amax, amin = np.max(chunk_a), np.min(chunk_a)
    bmax, bmin = np.max(chunk_b), np.min(chunk_b)

    chunk -= amin
    chunk /= amax - amin
    chunk *= bmax - bmin
    chunk += bmin
    return chunk

def merge_window(window_o, window_p):
    chunk = window_o.shape[0] // 2
    chunk_hh = np.zeros((chunk*2, chunk*2))
    chunk_vv = np.zeros((chunk*2, chunk*2))
    debug_list = [np.zeros((chunk*2, chunk*2)) for _ in range(16)]

    debug_list[14] = normalize(window_o.copy())
    debug_list[15] = normalize(window_p.copy())

    x, y, w, h = chunk//2, 0, chunk, chunk
    chunk_v = window_o[y:y+h, x:x+w].copy()
    chunk_h = window_p[y:y+h, x:x+w].copy()
    b_middle = merge_chunk(chunk_h, chunk_v)

    x, y, w, h = 0, 0, chunk, chunk
    b_left = window_o[y:y+h, x:x+w].copy()
    x, y, w, h = chunk, 0, chunk, chunk
    b_right = window_o[y:y+h, x:x+w].copy()

    x, y, w, h, shift = 0, 0, chunk*2, chunk, chunk//2
    chunk_hh[y:y+h, x:x+w] = merge_bridge(b_left, b_middle, b_right, shift)
    debug_list[13][y:y+h, x:x+w] = merge_bridge(b_left, merge_chunk(chunk_h, chunk_v, True, 0), b_right, shift, True)
    debug_list[12][y:y+h, x:x+w] = merge_bridge(b_left, merge_chunk(chunk_h, chunk_v, True, 1), b_right, shift, True)
    debug_list[11][y:y+h, x:x+w] = merge_bridge(b_left, merge_chunk(chunk_h, chunk_v, True, 2), b_right, shift, True)
    debug_list[10][y:y+h, x:x+w] = merge_bridge(b_left, merge_chunk(chunk_h, chunk_v, True, 3), b_right, shift, True)
    debug_list[9][y:y+h, x:x+w] = merge_bridge(b_left, merge_chunk(chunk_h, chunk_v), b_right, shift, True)
    normalize(debug_list[13][y:y+h, x:x+w])
    normalize(debug_list[12][y:y+h, x:x+w])
    normalize(debug_list[11][y:y+h, x:x+w])
    normalize(debug_list[10][y:y+h, x:x+w])
    normalize(debug_list[9][y:y+h, x:x+w])

    x, y, w, h = chunk//2, chunk, chunk, chunk
    chunk_v = window_o[y:y+h, x:x+w].copy()
    chunk_h = window_p[y:y+h, x:x+w].copy()
    b_middle = merge_chunk(chunk_h, chunk_v)

    x, y, w, h = 0, chunk, chunk, chunk
    b_left = window_o[y:y+h, x:x+w].copy()
    x, y, w, h = chunk, chunk, chunk, chunk
    b_right = window_o[y:y+h, x:x+w].copy()

    x, y, w, h, shift = 0, chunk, chunk*2, chunk, chunk//2
    chunk_hh[y:y+h, x:x+w] = merge_bridge(b_left, b_middle, b_right, shift)
    debug_list[13][y:y+h, x:x+w] = merge_bridge(b_left, merge_chunk(chunk_h, chunk_v, True, 0), b_right, shift, True)
    debug_list[12][y:y+h, x:x+w] = merge_bridge(b_left, merge_chunk(chunk_h, chunk_v, True, 1), b_right, shift, True)
    debug_list[11][y:y+h, x:x+w] = merge_bridge(b_left, merge_chunk(chunk_h, chunk_v, True, 2), b_right, shift, True)
    debug_list[10][y:y+h, x:x+w] = merge_bridge(b_left, merge_chunk(chunk_h, chunk_v, True, 3), b_right, shift, True)
    debug_list[9][y:y+h, x:x+w] = merge_bridge(b_left, merge_chunk(chunk_h, chunk_v), b_right, shift, True)
    normalize(debug_list[13][y:y+h, x:x+w])
    normalize(debug_list[12][y:y+h, x:x+w])
    normalize(debug_list[11][y:y+h, x:x+w])
    normalize(debug_list[10][y:y+h, x:x+w])
    normalize(debug_list[9][y:y+h, x:x+w])

    x, y, w, h = 0, chunk//2, chunk, chunk
    chunk_v = window_p[y:y+h, x:x+w].copy()
    chunk_h = window_o[y:y+h, x:x+w].copy()
    b_middle = merge_chunk(chunk_h, chunk_v)

    x, y, w, h = 0, 0, chunk, chunk
    b_left = window_o[y:y+h, x:x+w].copy()
    x, y, w, h = 0, chunk, chunk, chunk
    b_right = window_o[y:y+h, x:x+w].copy()

    x, y, w, h, shift = 0, 0, chunk, chunk*2, chunk//2
    chunk_vv[y:y+h, x:x+w] = merge_bridge(b_left.T, b_middle.T, b_right.T, shift).T
    debug_list[8][y:y+h, x:x+w] = merge_bridge(b_left.T, merge_chunk(chunk_h, chunk_v, True, 0).T, b_right.T, shift, True).T
    debug_list[7][y:y+h, x:x+w] = merge_bridge(b_left.T, merge_chunk(chunk_h, chunk_v, True, 1).T, b_right.T, shift, True).T
    debug_list[6][y:y+h, x:x+w] = merge_bridge(b_left.T, merge_chunk(chunk_h, chunk_v, True, 2).T, b_right.T, shift, True).T
    debug_list[5][y:y+h, x:x+w] = merge_bridge(b_left.T, merge_chunk(chunk_h, chunk_v, True, 3).T, b_right.T, shift, True).T
    debug_list[4][y:y+h, x:x+w] = merge_bridge(b_left.T, merge_chunk(chunk_h, chunk_v).T, b_right.T, shift, True).T
    normalize(debug_list[8][y:y+h, x:x+w])
    normalize(debug_list[7][y:y+h, x:x+w])
    normalize(debug_list[6][y:y+h, x:x+w])
    normalize(debug_list[5][y:y+h, x:x+w])
    normalize(debug_list[4][y:y+h, x:x+w])

    x, y, w, h = chunk, chunk//2, chunk, chunk
    chunk_v = window_p[y:y+h, x:x+w].copy()
    chunk_h = window_o[y:y+h, x:x+w].copy()
    b_middle = merge_chunk(chunk_h, chunk_v)

    x, y, w, h = chunk, 0, chunk, chunk
    b_left = window_o[y:y+h, x:x+w].copy()
    x, y, w, h = chunk, chunk, chunk, chunk
    b_right = window_o[y:y+h, x:x+w].copy()

    x, y, w, h, shift = chunk, 0, chunk, chunk*2, chunk//2
    chunk_vv[y:y+h, x:x+w] = merge_bridge(b_left.T, b_middle.T, b_right.T, shift).T
    debug_list[8][y:y+h, x:x+w] = merge_bridge(b_left.T, merge_chunk(chunk_h, chunk_v, True, 0).T, b_right.T, shift, True).T
    debug_list[7][y:y+h, x:x+w] = merge_bridge(b_left.T, merge_chunk(chunk_h, chunk_v, True, 1).T, b_right.T, shift, True).T
    debug_list[6][y:y+h, x:x+w] = merge_bridge(b_left.T, merge_chunk(chunk_h, chunk_v, True, 2).T, b_right.T, shift, True).T
    debug_list[5][y:y+h, x:x+w] = merge_bridge(b_left.T, merge_chunk(chunk_h, chunk_v, True, 3).T, b_right.T, shift, True).T
    debug_list[4][y:y+h, x:x+w] = merge_bridge(b_left.T, merge_chunk(chunk_h, chunk_v).T, b_right.T, shift, True).T
    normalize(debug_list[8][y:y+h, x:x+w])
    normalize(debug_list[7][y:y+h, x:x+w])
    normalize(debug_list[6][y:y+h, x:x+w])
    normalize(debug_list[5][y:y+h, x:x+w])
    normalize(debug_list[4][y:y+h, x:x+w])

    final_chunk = merge_chunk(chunk_hh, chunk_vv)
    debug_list[3] = merge_chunk(chunk_hh, chunk_vv, True, 0)
    debug_list[2] = merge_chunk(chunk_hh, chunk_vv, True, 1)
    debug_list[1] = merge_chunk(chunk_hh, chunk_vv, True, 2)
    debug_list[0] = merge_chunk(chunk_hh, chunk_vv, True, 3)
    normalize(debug_list[3])
    normalize(debug_list[2])
    normalize(debug_list[1])
    normalize(debug_list[0])

    return final_chunk, debug_list

def normalize(chunk):
    chunk -= np.min(chunk)
    chunk /= np.max(chunk)
    return chunk

def normalize_chunk(image, n, mode='o'):
    height, width = image.shape
    chunk = width // n

    if mode == 'o':
        for i in range(n):
            for j in range(n):
                x, y, w, h = i*chunk, j*chunk, chunk, chunk
                image[y:y+h, x:x+w] = normalize(image[y:y+h, x:x+w])
    else:
        for i in range(n+1):
            for j in range(n+1):
                w, h = chunk, chunk
                x, y = w*i - w//2, h*j - h//2

                if (i == 0): x = 0
                if (j == 0): y = 0
                if (i == 0 or i == n): w = chunk//2
                if (j == 0 or j == n): h = chunk//2
                image[y:y+h, x:x+w] = normalize(image[y:y+h, x:x+w])

def merge_level(image_o, image_p, split):
    image_oo = image_o.copy()
    image_pp = image_p.copy()
    size = image_o.shape[0]
    n = split // 2
    chunk = size//n

    debug_list = []

    for i in range(n):
        for j in range(n):
            x, y, w, h = i*chunk, j*chunk, chunk, chunk
            image_oo[y:y+h, x:x+w], debug_chunk_list = merge_window(image_o[y:y+h, x:x+w], image_p[y:y+h, x:x+w])

            if not debug_list:
                for _ in debug_chunk_list:
                    debug_list.append(np.zeros_like(image_o))
            for d, image in enumerate(debug_chunk_list):
                debug_list[d][y:y+h, x:x+w] = image

    x, y, w, h = 0, 0, chunk, chunk
    image_pp[y:y+h, x:x+w], _ = merge_window(image_o[y:y+h, x:x+w], image_p[y:y+h, x:x+w])
    x, y, w, h = size-chunk, 0, chunk, chunk
    image_pp[y:y+h, x:x+w], _ = merge_window(image_o[y:y+h, x:x+w], image_p[y:y+h, x:x+w])
    x, y, w, h = 0, size-chunk, chunk, chunk
    image_pp[y:y+h, x:x+w], _ = merge_window(image_o[y:y+h, x:x+w], image_p[y:y+h, x:x+w])
    x, y, w, h = size-chunk, size-chunk, chunk, chunk
    image_pp[y:y+h, x:x+w], _ = merge_window(image_o[y:y+h, x:x+w], image_p[y:y+h, x:x+w])

    for i in range(n-1):
        x, y, w, h = (2*i+1)*chunk//2, 0, chunk, chunk
        image_pp[y:y+h, x:x+w], _ = merge_window(image_o[y:y+h, x:x+w], image_p[y:y+h, x:x+w])
        x, y, w, h = (2*i+1)*chunk//2, size-chunk, chunk, chunk
        image_pp[y:y+h, x:x+w], _ = merge_window(image_o[y:y+h, x:x+w], image_p[y:y+h, x:x+w])
        x, y, w, h = 0, (2*i+1)*chunk//2, chunk, chunk
        image_pp[y:y+h, x:x+w], _ = merge_window(image_o[y:y+h, x:x+w], image_p[y:y+h, x:x+w])
        x, y, w, h = size-chunk, (2*i+1)*chunk//2, chunk, chunk
        image_pp[y:y+h, x:x+w], _ = merge_window(image_o[y:y+h, x:x+w], image_p[y:y+h, x:x+w])

    for i in range(n-1):
        for j in range(n-1):
            x, y, w, h = (2*i+1)*chunk//2, (2*j+1)*chunk//2, chunk, chunk
            image_pp[y:y+h, x:x+w], _ = merge_window(image_o[y:y+h, x:x+w], image_p[y:y+h, x:x+w])

    return image_oo, image_pp, debug_list

def merge_split(image_o, image_p, split):
    image_oo = image_o.copy()
    image_pp = image_p.copy()

    while split > 1:
        image_oo, image_pp = merge_level(image_oo, image_pp, split)
        split = split // 2

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

    level = 3
    decimation = 2**level
    umb = np.array([3900, 2304]) // decimation

    # x0, y0, n, chunk = 8*12, 10*12, 4, 50
    # x0, y0, n, chunk = 8*12, 10*12, 16, 12
    # x0, y0, n, chunk = 100, 100, 32, 12
    # x0, y0, n, chunk = 4*12, 10*12, 32, 12
    # x0, y0, n, chunk = 4*12, 10*12, 64, 6

    n, chunk = 2, 30
    w0, h0 = chunk*n, chunk*n
    x0, y0 = umb[0] - chunk*n//2, umb[1] - chunk*n
    # x0, y0 = umb[0] - chunk*n//2, umb[1] - chunk*n//2 # circle

    image_uo = createThetaArray(umb, x0, y0, w0, h0)
    image_vo = createRadiusArray(umb, x0, y0, w0, h0)
    image_up = image_uo.copy()
    image_vp = image_vo.copy()
    radius =  image_vo.copy()
    theta = image_uo.copy()

    print('Compute Eigens ...')
    st = Struct()
    st.vector_u = np.zeros((h0, w0, 2), dtype=np.float32)
    st.vector_v = np.zeros((h0, w0, 2), dtype=np.float32)
    st.coherence = np.zeros((h0, w0), dtype=np.float32)

    m = n
    for i in range(m):
        for j in range(m):
            w, h = chunk, chunk
            cx, cy = chunk*n//2, chunk*n
            x, y = w*i + cx - chunk*m//2, h*j + cy - chunk*m
            # cx, cy = chunk*n//2, chunk*n//2 # circle
            # x, y = w*i + cx - chunk*m//2, h*j + cy - chunk*m//2 # circle

            uvec = np.array([x+chunk//2-cx, y+chunk//2-cy])
            norm = np.linalg.norm(uvec)
            uvec = uvec / norm
            st.vector_u[y:y+h, x:x+w] = uvec

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

            u_tasks.append((image_u, image_st))
            v_tasks.append((image_v, image_st))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(updateU, u_tasks), total=len(u_tasks)))
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(updateV, v_tasks), total=len(v_tasks)))

    # for i in range(2):
    #     for j in range(2):
    #         w, h = chunk, chunk
    #         cx, cy = chunk*n//2, chunk*n//2
    #         x, y = w*i + cx - w, h*j + cy - h

    #         image_uo[y:y+h, x:x+w] = theta[y:y+h, x:x+w]
    # for i in range(3):
    #     for j in range(3):
    #         if (i==0 and j==0): continue
    #         if (i==2 and j==0): continue
    #         if (i==0 and j==2): continue
    #         if (i==2 and j==2): continue

    #         w, h = chunk, chunk
    #         cx, cy = chunk*n//2, chunk*n//2
    #         x, y = w*i + cx - w*3//2, h*j + cy - h*3//2

    #         image_up[y:y+h, x:x+w] = theta[y:y+h, x:x+w]

    # 1. 想一下要怎麼有系統的測試問題的每個環節 (solved)
    # 2. vo 為原始 radius 時正常嗎？尺度縮小時會不會發散？ (solved)
    # 3. vo 為 st 圓心推導產生的簡單梯度，合併結果正常嗎？ (solved)
    # 4. vo 為原始卷軸資料時，能跑回原本的結果嗎？ (solved)
    # 5. uo 為 st 圓心推導產生的簡單梯度，合併結果正常嗎？
    # 6. uo 跑看看原始卷軸資料，有問題嗎？
    # 7. 上面都正常後，才能思考 theta 的合併問題...

    # 我暫時把 scale_shift, Lshape scale 註解掉了，那些是形變的來源

    image_uo_original = image_uo.copy()
    image_up_original = image_up.copy()
    image_vo_original = image_vo.copy()
    image_vp_original = image_vp.copy()

    print('merge image_vo & image_vp')
    # split = n
    # image_vo, image_vp, debug_list = merge_level(image_vo, image_vp, split)
    # split = split // 2
    # image_vo, image_vp, debug_list = merge_level(image_vo, image_vp, split)
    # split = split // 2
    # image_vo, image_vp, debug_list = merge_level(image_vo, image_vp, split)
    # split = split // 2
    # image_vo, image_vp, debug_list = merge_level(image_vo, image_vp, split)
    # split = split // 2
    # # image_vo, image_vp, debug_list = merge_level(image_vo, image_vp, split)
    # # split = split // 2

    # normalize_chunk(image_vo, split, 'o')
    # normalize_chunk(image_vp, split, 'p')

    print('merge image_uo & image_up')
    split = n
    image_uo, image_up, debug_list = merge_level(image_uo, image_up, split)
    split = split // 2
    # image_uo, image_up, debug_list = merge_level(image_uo, image_up, split)
    # split = split // 2
    # image_uo, image_up, debug_list = merge_level(image_uo, image_up, split)
    # split = split // 2
    # image_uo, image_up, debug_list = merge_level(image_uo, image_up, split)
    # split = split // 2
    # # image_uo, image_up, debug_list = merge_level(image_uo, image_up, split)
    # # split = split // 2

    normalize_chunk(image_uo, split, 'o')
    normalize_chunk(image_up, split, 'p')

    row_num, col_num = 3, 6
    fig, axes = plt.subplots(row_num, col_num, figsize=(9, 5))
    colormap = cmap.Colormap("tab20", interpolation="nearest")
    for ax in axes.flat: ax.axis('off')
    col_list = [0] * row_num

    row = 0
    axes[row, col_list[row]].imshow(colormap(image_uo_original), aspect='equal')
    axes[row, col_list[row]].set_title('uo_original')
    col_list[row] += 1

    row = 0
    axes[row, col_list[row]].imshow(colormap(image_uo), aspect='equal')
    axes[row, col_list[row]].set_title('uo_final')
    col_list[row] += 1

    debug_i = 0
    for row in range(row_num):
        for col in range(col_num):
            if (col < col_list[row]): continue
            if (len(debug_list) < debug_i + 1): continue

            axes[row, col_list[row]].imshow(colormap(debug_list[debug_i]), aspect='equal')
            axes[row, col_list[row]].set_title('debug_' + str(debug_i))
            col_list[row] += 1
            debug_i += 1

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    sys.exit(main())
