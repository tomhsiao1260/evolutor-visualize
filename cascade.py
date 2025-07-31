import sys
import zarr
import argparse
from wind2d import ImageViewer

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

def createUVArray(image):
    ivs, ius = np.mgrid[:image.shape[0], :image.shape[1]]
    return ius, ivs

def solveUV(basew, st, smoothing_weight, axis='u'):
    vecu = st.vector_u
    coh = st.coherence[:,:,np.newaxis]
    wvecu = coh*vecu
    shape = wvecu.shape[:2]

    if axis=='u':
        sparse_uxg = ImageViewer.sparseVecOpGrad(wvecu, is_cross=True)
    else:
        sparse_uxg = ImageViewer.sparseVecOpGrad(wvecu, is_cross=False)

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

def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="")
    parser.add_argument("input_ome_zarr",
                        help="input ome zarr directory")

    args = parser.parse_args()

    input_ome_zarr = args.input_ome_zarr

    y0, x0, chunk = 896, 896, 128

    z_scroll = zarr.open(input_ome_zarr, mode='r')

    data_scroll = z_scroll[0, y0:y0+chunk, x0:x0+chunk]

    print(data_scroll.shape)


if __name__ == '__main__':
    sys.exit(main())
