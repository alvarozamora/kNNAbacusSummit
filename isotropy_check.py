print("importing modules")
from importlib.abc import Finder
from AbacusCosmos import Halos
from scipy.spatial import KDTree as Tree
import matplotlib.pyplot as plt
import numpy as np
import glob
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
print("imported modules")

BOXSIZE = 1100
FINDER = "FoF"
PRODUCT = f"{BOXSIZE}box_planck"
Z = f"{1.025:.3f}"


def load_catalog(i):
    
    dir_to_prods= f"/oak/stanford/orgs/kipac/users/pizza/AbacusSummit"
    cat = Halos.make_catalog_from_dir(dirname=f'{dir_to_prods}/AbacusSummit_fixedbase_c000_ph0{98+i}/halos/halos/z1.025',
                                    load_subsamples=False, load_pids=False)

    return cat


def get_kNN_angular_dist(pos, Nr=10**6, boxsize=1100):

    k = 1

    # Build tree
    tree = Tree(pos, leafsize=128, compact_nodes=True, balanced_tree=True, boxsize=boxsize)

    # Generate query points and query tree
    queries = np.random.uniform(size=(Nr,3))*boxsize
    r, ids = tree.query(queries, k=[k], workers=32)

    # Compute displacement vectors
    D = pos[ids]
    assert D.shape == (Nr, k, 3)
    D = np.abs(D - queries[:, None, :]) # The None allows broadcasting over all k neighbors of given query. abs brings all pts to positive octant.
    D = np.minimum(D, boxsize-D) # If something wraps around due to periodic BCs, need smallest of distances componentwise.
    
    # Compute angles
    cost = D[:,:,2] / np.linalg.norm(D,axis=-1)

    return cost

def plot_pCDFs(cost):

    # Compute pcdf
    cost = np.sort(cost,axis=0)[::100]
    cdf = np.arange(1, len(cost)+1)/len(cost) # 1/N to N/N
    #pcdf = np.minimum(cdf, 1-cdf)

    # Plot
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(12, 8))
    
    a0.plot(cost, cost, label='isotropic')
    a0.plot(cost, cdf, '--', label='abacus')
    a0.grid(alpha=0.3)
    a0.legend()
    a0.set_ylabel('CDF')

    a1.plot(cost, cdf-cost[:,0])
    a1.grid(alpha=0.3)
    a1.set_xlabel(r"$\mu$")
    a1.set_ylabel('CDF Residual')

    # Save
    plt.savefig("kNN_angular_cdf.png",dpi=230)
    
def num_boxes():
    
    return 2

    

def main():

    Nq = 10**7
    costs = np.zeros(Nq)
    #for i in range(num_boxes()//2):
    datasets = glob.glob("/oak/stanford/orgs/kipac/abacus/AbacusSummit_base_c*_ph000/halos/z0.500")
    for (i, dataset) in enumerate(datasets):

        # Load catalogs
        #dataset = f'AbacusSummit_fixedbase_c000_ph0{98+i}/halos/z{Z}'

        print(dataset)
        cat = CompaSOHaloCatalog(dataset, cleaned=False)#, subsamples=dict(A=Fal,pos=True))

        # Extract positions
        cat = np.array(cat.halos['x_com']%BOXSIZE%BOXSIZE)
        cat = cat.astype(np.float64)
        print(cat.min(), cat.max(), cat.shape)
        
        # Compute kNN-angular dists
        oneNN = get_kNN_angular_dist(cat, Nq, BOXSIZE)[:,0]
        costs += np.sort(oneNN)/len(datasets)
        del cat
    
    # costs = np.concatenate(costs)
    # Plot pCDFs
    plot_pCDFs(costs)



if __name__ == "__main__":
    main()
