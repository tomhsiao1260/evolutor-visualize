import sys
import cmap
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
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

# ymin = a*xmin + b, ymax = a*xmax + b
def scale_shift(x, y):
    xmax, xmin = np.max(x), np.min(x)
    ymax, ymin = np.max(y), np.min(y)
    if (xmax != xmin):
        a = (ymax - ymin) / (xmax - xmin)
        b = (ymin * xmax - ymax * xmin) / (xmax - xmin)
    else:
        a = 1
        b = (ymax + ymin) / 2 - (xmax + xmin) / 2
    return a, b

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

def merge_rectangle(image_v, image_h):
    rect_v, rect_h = image_v.copy(), image_h.copy()
    height, width = rect_v.shape[0], rect_h.shape[1]
    ws, hs = rect_v.shape[1], rect_h.shape[0]
    rect_m = np.zeros((height, width))

    # top-left
    x, y, w, h = 0, 0, ws, hs
    region0 = (rect_v[y:y+h, x:x+w] + rect_h[y:y+h, x:x+w]) / 2
    rect_m[y:y+h, x:x+w] = region0
    # top-right
    x, y, w, h = ws-1, 0, width-ws+1, hs
    a, b = scale_shift(rect_h[:, x:x+1], region0[:, -1:])
    rect_h = a * rect_h + b
    region1 = rect_h[y:y+h, x:x+w]
    region1 += region0[:, -1:] - rect_h[:, x:x+1]
    rect_m[y:y+h, x:x+w] = region1
    # bottom-left
    x, y, w, h = 0, height-hs-1, ws, height-hs+1
    a, b = scale_shift(rect_v[y:y+1, :], region0[-1:, :])
    rect_v = a * rect_v + b
    region2 = rect_v[y:y+h, x:x+w]
    region2 += region0[-1:, :] - rect_v[y:y+1, :]
    rect_m[y:y+h, x:x+w] = region2

    # normalize
    value_min = min(np.min(region0), np.min(region1), np.min(region2))
    value_max = max(np.max(region0), np.max(region1), np.max(region2))
    if (value_max != value_min):
        rect_m -= value_min
        rect_m /= value_max - value_min
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

    # region 1 (right, top)
    x, y, w, h = w0//2, 0, w0//2, h0//2
    a = l_left_top[y:y+h, x:x+w]
    b = l_right_bottom[y:y+h, x:x+w]
    # region 2 (left, bottom)
    x, y, w, h = 0, h0//2, w0//2, h0//2
    c = l_left_top[y:y+h, x:x+w]
    d = l_right_bottom[y:y+h, x:x+w]
    # scale factor
    scale = np.sum(a)+np.sum(c)+1e-7
    scale /= np.sum(b)+np.sum(d)+1e-7
    l_right_bottom *= scale

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

def merge_bridge(b_left, b_middle, b_right, shift):
    height, width_bl = b_left.shape
    height, width_bm = b_middle.shape
    height, width_br = b_right.shape
    chunk = np.zeros((height, width_bl + width_br))

    # left
    x, y, w, h = 0, 0, width_bl, height
    m_edge = b_middle[:, 0:1]
    l_edge = b_left[:, shift:shift+1]
    a, b = scale_shift(l_edge, m_edge)
    b_left = a * b_left + b
    l_edge = b_left[:, shift:shift+1]
    chunk[y:y+h, x:x+w] = b_left + (m_edge - l_edge)

    # right
    x, y, w, h = width_bl, 0, width_br, height
    sh = shift + width_bm - width_bl - 1
    m_edge = b_middle[:, -1:]
    r_edge = b_right[:, sh:sh+1]
    a, b = scale_shift(r_edge, m_edge)
    b_right = a * b_right + b
    r_edge = b_right[:, sh:sh+1]
    chunk[y:y+h, x:x+w] = b_right + (m_edge - r_edge)

    # middle
    x, y, w, h = shift, 0, width_bm, height
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

    x, y, w, h, shift = 0, 0, chunk*2, chunk, chunk//2
    chunk_hh[y:y+h, x:x+w] = merge_bridge(b_left, b_middle, b_right, shift)

    x, y, w, h = chunk//2, chunk, chunk, chunk
    chunk_v = window_o[y:y+h, x:x+w].copy()
    chunk_h = window_p[y:y+h, x:x+w].copy()
    b_middle = merge_chunk(chunk_v, chunk_h)

    x, y, w, h = 0, chunk, chunk, chunk
    b_left = window_o[y:y+h, x:x+w].copy()
    x, y, w, h = chunk, chunk, chunk, chunk
    b_right = window_o[y:y+h, x:x+w].copy()

    x, y, w, h, shift = 0, chunk, chunk*2, chunk, chunk//2
    chunk_hh[y:y+h, x:x+w] = merge_bridge(b_left, b_middle, b_right, shift)

    x, y, w, h = 0, chunk//2, chunk, chunk
    chunk_v = window_p[y:y+h, x:x+w].copy()
    chunk_h = window_o[y:y+h, x:x+w].copy()
    b_middle = merge_chunk(chunk_v, chunk_h)

    x, y, w, h = 0, 0, chunk, chunk
    b_left = window_o[y:y+h, x:x+w].copy()
    x, y, w, h = 0, chunk, chunk, chunk
    b_right = window_o[y:y+h, x:x+w].copy()

    x, y, w, h, shift = 0, 0, chunk, chunk*2, chunk//2
    chunk_vv[y:y+h, x:x+w] = merge_bridge(b_left.T, b_middle.T, b_right.T, chunk//2).T

    x, y, w, h = chunk, chunk//2, chunk, chunk
    chunk_v = window_p[y:y+h, x:x+w].copy()
    chunk_h = window_o[y:y+h, x:x+w].copy()
    b_middle = merge_chunk(chunk_v, chunk_h)

    x, y, w, h = chunk, 0, chunk, chunk
    b_left = window_o[y:y+h, x:x+w].copy()
    x, y, w, h = chunk, chunk, chunk, chunk
    b_right = window_o[y:y+h, x:x+w].copy()

    x, y, w, h, shift = chunk, 0, chunk, chunk*2, chunk//2
    chunk_vv[y:y+h, x:x+w] = merge_bridge(b_left.T, b_middle.T, b_right.T, shift).T
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
    num_plot, n, chunk = 6, 4, 96
    x0, y0 = 0, 0
    w0, h0 = chunk*n, chunk*n
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

    # x, y, w, h = w0//4, 0, w0//4, h0//4
    # b_left = image_uo[y:y+h, x:x+w].copy()
    # x, y, w, h = 3*w0//8, 0, w0//4, h0//4
    # b_middle = image_uo[y:y+h, x:x+w].copy()
    # x, y, w, h = w0//2, 0, w0//4, h0//4
    # b_right = image_uo[y:y+h, x:x+w].copy()
    # x, y, w, h, shift = w0//4, 0, w0//2, h0//4, 3*w0//8-w0//4
    # image_uo[y:y+h, x:x+w] = merge_bridge(b_left, b_middle, b_right, shift)

    shx, shy = w0//6, w0//6

    # vertical rectangle (left)
    x, y, w, h = shx+0, shy+0, w0//4, h0//2
    rect_v = image_uo[y:y+h, x:x+w].copy()
    rect_v -= np.min(rect_v)
    rect_v /= np.max(rect_v)
    # horizontal rectangle (top)
    x, y, w, h = shx+0, shy+0, w0//2, h0//4
    rect_h = image_uo[y:y+h, x:x+w].copy()
    rect_h -= np.min(rect_h)
    rect_h /= np.max(rect_h)
    # merge the cross region
    l_left_top = merge_rectangle(rect_v, rect_h)
    # x, y, w, h = sh+0, 0, w0//2, h0//2
    # image_uo[y:y+h, x:x+w] = l_left_top

    # vertical rectangle (right)
    x, y, w, h = shx+w0//4, shy+0, w0//4, h0//2
    rect_v = image_uo[y:y+h, x:x+w].copy()
    rect_v -= np.min(rect_v)
    rect_v /= np.max(rect_v)
    # horizontal rectangle (bottom)
    x, y, w, h = shx+0, shy+h0//4, w0//2, h0//4
    rect_h = image_uo[y:y+h, x:x+w].copy()
    rect_h -= np.min(rect_h)
    rect_h /= np.max(rect_h)
    # merge the cross region (reversed)
    l_right_bottom = merge_rectangle(rect_v[::-1, ::-1], rect_h[::-1, ::-1])
    l_right_bottom = l_right_bottom[::-1, ::-1]
    # x, y, w, h = sh+0, 0, w0//2, h0//2
    # a = np.max(image_uo[y:y+h, x:x+w])
    # image_uo[y:y+h, x:x+w] = l_right_bottom
    # image_uo[y:y+h, x:x+w] *= a

    x, y, w, h = shx+0, shy+0, w0//2, h0//2
    a = np.max(image_uo[y:y+h, x:x+w])
    b = np.min(image_uo[y:y+h, x:x+w])
    image_uo[y:y+h, x:x+w] = merge_Lshape(l_left_top, l_right_bottom)
    image_uo[y:y+h, x:x+w] -= b
    image_uo[y:y+h, x:x+w] *= (a-b)

    # x, y, w, h = w0//5, 0, w0//2, h0//2
    # l_left_top = image_uo[y:y+h, x:x+w]
    # x, y, w, h = w0//5, 0, w0//2, h0//2
    # l_right_bottom = image_uo[y:y+h, x:x+w]
    # x, y, w, h = w0//5, 0, w0//2, h0//2
    # a = np.max(image_uo[y:y+h, x:x+w])
    # image_uo[y:y+h, x:x+w] = merge_chunk(l_left_top, l_right_bottom)
    # image_uo[y:y+h, x:x+w] *= a

    # merge_bridge (correct)
    # merge_rectangle (correct)
    # merge_Lshape (correct)

    plt.subplot(1, num_plot, 3)
    plt.imshow(colormap(image_vo), aspect='equal')
    plt.axis('off')

    plt.subplot(1, num_plot, 4)
    plt.imshow(colormap(image_uo), aspect='equal')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    sys.exit(main())




















