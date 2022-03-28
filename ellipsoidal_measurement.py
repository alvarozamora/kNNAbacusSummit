import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as Tree
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from time import time
from tqdm import tqdm
from os.path import exists

BOXSIZE = 1100
VOLUME = BOXSIZE * BOXSIZE * BOXSIZE
LOS = 2

def compute_ellipsoidal_kNN(pos, s, k=1, axis = 'z', Nrand = 10**8, buffer=16):

    assert buffer > k, "Buffer not big enough. Rough recommendation is >8*k"

    # Build KDTree on data
    tree = Tree(pos, leafsize=buffer, compact_nodes=True, balanced_tree=True, boxsize=BOXSIZE)

    # Query tree
    query = np.random.uniform(size=(Nrand,3))*BOXSIZE
    _, ids = tree.query(query, k=buffer, workers=-1)

    # Displacement vectors
    # None allows broadcasting across all k Neighbors
    # abs() snaps back values to positive quadrant
    # minimum() takes care of periodic boundary conditions
    D = np.abs(pos[ids] - query[:,None])
    D = np.minimum(D, BOXSIZE - D)

    # Reweight based on stretch
    if axis.lower() == 'z':
        D = np.sqrt( s * D[:,:,0]**2 + s * D[:,:,1]**2 + D[:,:,2]**2 / s**2 )
    elif axis.lower() == 'y':
        D = np.sqrt( s * D[:,:,0]**2 + s * D[:,:,2]**2 + D[:,:,1]**2 / s**2 )
    elif axis.lower() == 'x':
        D = np.sqrt( s * D[:,:,1]**2 + s * D[:,:,2]**2 + D[:,:,0]**2 / s**2 )
    else:
        assert False, "Invalid Axis mode"

    # Re-sort neighbors
    r = []; rids = []
    for (i, sort) in enumerate(np.argsort(D, axis=1)):
        r.append(D[i][sort][:k])
        rids.append(ids[i][sort][:k])
    r = np.array(r)
    rids = np.array(rids)

    return r, ids


def plot_kNN(z, x, sz, sx, labels, compress = 1000):

    # Sort distances
    z = np.percentile(z, np.arange(1,1000)/10, axis=0)
    x = np.percentile(x, np.arange(1,1000)/10, axis=0)
    sz = np.percentile(sz, np.arange(1,1000)/10, axis=0)
    sx = np.percentile(sx, np.arange(1,1000)/10, axis=0)

    # Compute CDF, pCDF
    cdf = np.arange(1, len(z)+1)/len(z)
    pcdf = np.minimum(cdf, 1-cdf)

    # Plot CDF, pCDF
    for q in range(len(z.T)):

        real_ratio = cdf/np.interp(z[:,q], x[:,q], cdf)
        red_ratio =  cdf/np.interp(sz[:,q], sx[:,q], cdf)
        ratio_min = np.concat((real_ratio,red_ratio)).min()
        ratio_max = np.concat((real_ratio,red_ratio)).max()
        mm_diff = ratio_max-ratio_min

        plt.figure(q, figsize=(14,12))

        plt.subplot(221)
        plt.title("Real Space")
        plt.loglog(z[:,q], pcdf, label=f"z{labels[q]}")
        plt.loglog(x[:,q], pcdf, label=f"x{labels[q]}")
        plt.ylabel(f"{labels[q]}NN CDF")
        plt.xlabel("Distance")
        plt.ylim(1e-3)
        plt.legend()
        plt.grid(alpha=0.3)

        plt.subplot(223)
        plt.semilogx(z[:,q], real_ratio, label=f"(z/x){labels[q]}")
        plt.legend()
        plt.ylabel(f"CDF Ratio")
        plt.xlabel("Distance")
        plt.grid()
        plt.ylim(ratio_min - mm_diff*0.05, ratio_max + mm_diff*0.05)
        plt.savefig(f"{labels[q]}", dpi=230)

        plt.subplot(222)
        plt.title("Redshift Space")
        plt.loglog(sz[:,q], pcdf, label=f"sz{labels[q]}")
        plt.loglog(sx[:,q], pcdf, label=f"sx{labels[q]}")
        plt.ylabel(f"{labels[q]}NN CDF")
        plt.xlabel("Distance")
        plt.ylim(1e-3)
        plt.legend()
        plt.grid(alpha=0.3)

        plt.subplot(224)
        plt.semilogx(sz[:,q], red_ratio, label=f"(sz/sx){labels[q]}")
        plt.legend()
        plt.ylabel(f"CDF Ratio")
        plt.xlabel("Distance")
        plt.grid()
        plt.ylim(ratio_min - mm_diff*0.05, ratio_max + mm_diff*0.05)
   
        plt.savefig(f"{labels[q]}", dpi=230)

def ellipsoidal_kNN_on_subsamples(data, N, k=1, s=4.0):

    # Compute number of sumbsamples
    expectation = int(N * VOLUME / 1e9)
    nsubs: int = len(data) // expectation
    excess: int = len(data) % expectation

    # Ensure random + split data in subsamples
    subsamples = np.split(np.random.permutation(data)[:-excess], nsubs)

    # Iterate through subsamples and accumulate measurements
    z = []
    x = []
    Nrand = 10**9//nsubs; print(f"Using {Nrand} randoms per subsample; there are {nsubs} subsamples")
    for subsample in subsamples:
        assert subsample.shape == (expectation, 3)

        z_, _= compute_ellipsoidal_kNN(subsample, s, k=k, axis='z', Nrand = Nrand)
        x_, _ = compute_ellipsoidal_kNN(subsample, s, k=k, axis='x', Nrand = Nrand)
        z.append(z_); x.append(x_)
    
    # Package as one array and return
    z = np.concatenate(z, axis=0); x = np.concatenate(x, axis=0)
    return z, x

def process_box(file: str, N: int, k: int = 1, s = 2):


    outname = file.replace("/","-") + '.npz'
    if not exists(outname):

        # Load catalog
        start = time()
        cat = CompaSOHaloCatalog(file, cleaned=False)
        data = np.array(cat.halos['x_com']%BOXSIZE%BOXSIZE, dtype=np.float64)
        vel = cat.halos['v_com']
        print(f"loaded data/vel in {time()-start:.2f} seconds")

        # Real space measurements
        z, x = ellipsoidal_kNN_on_subsamples(data, N, k, s)
        print(f"Real space ellipsoidal kNN in {time()-start:.2f} seconds")

        # Convert positions to redshift space
        start = time()
        data = (data + (1.0 + cat.header['Redshift'])/cat.header['H0'] * np.eye(3)[None, LOS] * vel)%BOXSIZE
        del vel
        sz, sx = ellipsoidal_kNN_on_subsamples(data, N, k, s)

        np.savez(file.replace("/","-") + '.npz', z=z,x=x,sz=sz,sx=sx)
        print(f"Redshift space ellipsoidal kNN in {time()-start:.2f} seconds")

        # Return results
        return z, x, sz, sx
    else:
        print(f"loading {outname}")
        data = np.load(outname)
        z, x, sz, sx = data["z"], data["x"], data["sz"], data["sx"]
        return z, x, sz, sx


if __name__ == "__main__":

    # Number density
    # N / Gpc^3
    N: int = 10**4

    # Stretch factor for ellipsoids
    s = 2

    # Neighbors (cannot go that high with kDTree + reweighting)
    k = 1

    # Abacus boxes to use
    boxes = [
        '/oak/stanford/orgs/kipac/users/pizza/AbacusSummit/AbacusSummit_fixedbase_c000_ph098/halos/z1.025',
        '/oak/stanford/orgs/kipac/users/pizza/AbacusSummit/AbacusSummit_fixedbase_c000_ph099/halos/z1.025'
    ]

    # Process each box and accumulate measurements
    z = []; sz = []
    x = []; sx = []
    for box in tqdm(boxes):
        z_, x_, sz_, sx_ = process_box(box, N, k=k, s=s)
        z.append(z_); x.append(x_)
        sz.append(sz_); sx.append(sx_)
    z = np.concatenate(z, axis=0); z = np.sort(z)
    x = np.concatenate(x, axis=0); x = np.sort(x)
    sz = np.concatenate(sz, axis=0); sz = np.sort(sz)
    sx = np.concatenate(sx, axis=0); sx = np.sort(sx)

    # Save measurements + metadata
    np.savez("measurements.npz", z=z, x=x, sz=sz, sx=sx, boxes = np.array(boxes))

    # Plot results
    plot_kNN(z, x, sz, sx, ["1"])
