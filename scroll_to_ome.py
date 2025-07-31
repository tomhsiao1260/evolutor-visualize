import sys
import zarr
import copy
import json
import tifffile
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import skimage.transform
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

# return None if succeeds, err string if fails
def create_ome_dir(zarrdir):
    # complain if directory already exists
    if zarrdir.exists():
        err = "Directory %s already exists"%zarrdir
        print(err)
        return err

    try:
        zarrdir.mkdir()
    except Exception as e:
        err = "Error while creating %s: %s"%(zarrdir, e)
        print(err)
        return err

def create_ome_headers(zarrdir, nlevels):
    zattrs_dict = {
        "multiscales": [
            {
                "axes": [
                    {
                        "name": "z",
                        "type": "space"
                    },
                    {
                        "name": "y",
                        "type": "space"
                    },
                    {
                        "name": "x",
                        "type": "space"
                    }
                ],
                "datasets": [],
                "name": "/",
                "version": "0.4"
            }
        ]
    }

    dataset_dict = {
        "coordinateTransformations": [
            {
                "scale": [
                ],
                "type": "scale"
            }
        ],
        "path": ""
    }
    
    zgroup_dict = { "zarr_format": 2 }

    datasets = []
    for l in range(nlevels):
        ds = copy.deepcopy(dataset_dict)
        ds["path"] = "%d"%l
        scale = 2.**l
        ds["coordinateTransformations"][0]["scale"] = [scale]*3
        # print(json.dumps(ds, indent=4))
        datasets.append(ds)
    zad = copy.deepcopy(zattrs_dict)
    zad["multiscales"][0]["datasets"] = datasets
    json.dump(zgroup_dict, (zarrdir / ".zgroup").open("w"), indent=4)
    json.dump(zad, (zarrdir / ".zattrs").open("w"), indent=4)

def tifs2zarr(tiffdir, zarrdir, chunk_size, obytes=0, slices=None, maxgb=None):
    tiff0 = tifffile.imread(tiffdir)

    ny0, nx0 = tiff0.shape
    otype = tiff0.dtype

    cx = nx0
    cy = ny0

    store = zarr.NestedDirectoryStore(zarrdir)
    tzarr = zarr.open(
            store=store, 
            shape=(1, cy, cx), 
            chunks=(1, chunk_size, chunk_size),
            dtype = otype,
            write_empty_chunks=False,
            fill_value=0,
            compressor=None,
            mode='w', 
            )

    tzarr[0,:,:] = tiff0

def divp1(s, c):
    n = s // c
    if s%c > 0:
        n += 1
    return n

def process_chunk(args):
    idata, odata, z, y, x, cz, cy, cx, algorithm = args
    ibuf = idata[2*z*cz:(2*z*cz+2*cz),
                 2*y*cy:(2*y*cy+2*cy),
                 2*x*cx:(2*x*cx+2*cx)]
    if np.max(ibuf) == 0:
        return  # Skip if the block is empty to save computation

    # pad ibuf to even in all directions
    ibs = ibuf.shape
    pad = (ibs[0]%2, ibs[1]%2, ibs[2]%2)
    if any(pad):
        ibuf = np.pad(ibuf, 
                      ((0,pad[0]),(0,pad[1]),(0,pad[2])), 
                      mode="symmetric")

    # algorithms:
    if algorithm == "nearest":
        obuf = ibuf[::2, ::2, ::2]
    elif algorithm == "gaussian":
        obuf = np.round(skimage.transform.rescale(ibuf, .5, preserve_range=True))
    elif algorithm == "mean":
        obuf = np.round(skimage.transform.downscale_local_mean(ibuf, (2,2,2)))
    else:
        raise ValueError(f"algorithm {algorithm} not valid")

    odata[z*cz:(z*cz+cz),
          y*cy:(y*cy+cy),
          x*cx:(x*cx+cx)] = np.round(obuf)

def resize(zarrdir, old_level, num_threads, algorithm="mean"):
    idir = zarrdir / ("%d"%old_level)
    if not idir.exists():
        err = f"input directory {idir} does not exist"
        print(err)
        return err
    
    odir = zarrdir / ("%d"%(old_level+1))
    idata = zarr.open(idir, mode="r")
    print("Creating level",old_level+1,"  input array shape", idata.shape, " algorithm", algorithm)

    cz, cy, cx = idata.chunks
    sz, sy, sx = idata.shape
    store = zarr.NestedDirectoryStore(odir)
    odata = zarr.open(
            store=store,
            shape=(1, divp1(sy,2), divp1(sx,2)),
            chunks=idata.chunks,
            dtype=idata.dtype,
            write_empty_chunks=False,
            fill_value=0,
            compressor=None,
            mode='w',
            )

    # Prepare tasks
    tasks = [(idata, odata, z, y, x, cz, cy, cx, algorithm) for z in range(divp1(sz, 2*cz))
                                                             for y in range(divp1(sy, 2*cy))
                                                             for x in range(divp1(sx, 2*cx))]

    # Use ThreadPoolExecutor to process blocks in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(process_chunk, tasks), total=len(tasks)))

    print("Processing complete")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_tiff_dir", 
        help="Directory of tiff files")
    parser.add_argument(
            "output_zarr_ome_dir", 
            help="Name of directory that will contain OME/zarr datastore")
    parser.add_argument(
            "--chunk_size",
            type=int, 
            default=128, 
            help="Size of chunk")
    parser.add_argument(
            "--obytes",
            type=int,
            default=0,
            help="number of bytes per pixel in output")
    parser.add_argument(
            "--nlevels", 
            type=int, 
            default=6, 
            help="Number of subdivision levels to create, including level 0")
    parser.add_argument(
            "--max_gb", 
            type=float, 
            default=None, 
            help="Maximum amount of memory (in Gbytes) to use; None means no limit")
    parser.add_argument(
            "--zarr_only", 
            action="store_true", 
            help="Create a simple Zarr data store instead of an OME/Zarr hierarchy")
    parser.add_argument(
            "--overwrite", 
            action="store_true", 
            # default=False,
            help="Overwrite the output directory, if it already exists")
    parser.add_argument(
            "--num_threads", 
            type=int, 
            default=cpu_count(), 
            help="Advanced: Number of threads to use for processing. Default is number of CPUs")
    parser.add_argument(
            "--algorithm",
            choices=['mean', 'gaussian', 'nearest'],
            default="mean",
            help="Advanced: algorithm used to sub-sample the data")
    parser.add_argument(
            "--ranges", 
            help="Advanced: output only a subset of the data.  Example (in xyz order): 2500:3000,1500:4000,500:600")
    parser.add_argument(
            "--first_new_level", 
            type=int, 
            default=None, 
            help="Advanced: If some subdivision levels already exist, create new levels, starting with this one")

    args = parser.parse_args()

    zarrdir = Path(args.output_zarr_ome_dir)

    tiffdir = Path(args.input_tiff_dir)
    if not tiffdir.exists():
        print("Input TIFF directory",tiffdir,"does not exist")
        return 1

    chunk_size = args.chunk_size
    nlevels = args.nlevels
    num_threads = args.num_threads
    algorithm = args.algorithm
    first_new_level = args.first_new_level
    if first_new_level is not None and first_new_level < 1:
        print("first_new_level must be at least 1")

    if first_new_level is None:
        err = create_ome_dir(zarrdir)
        if err is not None:
            print("error returned:", err)
            return 1

    err = create_ome_headers(zarrdir, nlevels)
    if err is not None:
        print("error returned:", err)
        return 1

    print("Creating level 0")
    err = tifs2zarr(tiffdir, zarrdir/"0", chunk_size)
    if err is not None:
        print("error returned:", err)
        return 1

    # for each level (1 and beyond):
    existing_level = 0
    if first_new_level is not None:
        existing_level = first_new_level-1
    for l in range(existing_level, nlevels-1):
        err = resize(zarrdir, l, num_threads, algorithm)
        if err is not None:
            print("error returned:", err)
            return 1

if __name__ == '__main__':
    sys.exit(main())
