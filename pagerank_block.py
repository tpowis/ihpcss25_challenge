import numpy as np
import sys
import time
import torch
import os

np.set_printoptions(threshold=1000, linewidth=250)

#Iteration to beat: 72 in 10s for sneaky-graph on bridges2 system with C code.

#1 CPU core, double precision: 1,018,860,000 (n_block=30,000) 
#1 CPU core, single precision: 3,378,800,000 (n_block=50,000)
#1 GPU, single precision: 58,749,800,000 (n_block=200,000)

#Torch settings
#torch.set_num_threads(8)
chip = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_type = torch.float64
#torch_type = torch.float32
#torch_type = torch.float16


def main():

    # print("PyTorch max threads:", torch.get_num_threads())
    # print("PyTorch intraop threads:", torch.get_num_interop_threads())
    # print(torch.__config__.show())

    # print("Using", torch.get_num_threads(), "threads")
    # print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"))

    #return

    #Setting up the adjacency matrix
    ns = 1000

    #Nice array
    An = torch.ones((ns,ns), dtype=torch_type, device=chip) - torch.eye(ns, dtype=torch_type, device=chip)

    #Sneaky array
    As = torch.triu(torch.ones((ns, ns), dtype=torch_type, device=chip))
    As = torch.flip(As, dims=[1])
    mask = torch.eye(ns, dtype=torch.bool, device=chip)
    As[mask] = 0.0

    #Selecting the problem
    #A = An.clone() #Nice
    A = As.clone() #Sneaky

    #Construct mass matrix
    col_sums = A.sum(dim=0)
    col_sums[col_sums == 0.0] = 1.0
    M = A / col_sums

    #Teleportation vector
    v = 1.0/float(ns)*torch.ones(ns, dtype=torch_type, device=chip)

    #Other constants
    d = 0.85
    dM = d*M
    dv = (1.0 - d)*v

    #Initial condition and allocation for iteration
    r0 = 1.0/float(ns)*torch.ones(ns, dtype=torch_type, device=chip)
    r1 = torch.zeros(ns, dtype=torch_type, device=chip)

    #Block size
    n_block = 300000

    #Maximum fumber of iterations
    n_max = 1_000_000_000
    duration = 10.0

    #Allocating some memory for block
    dM_block = torch.zeros_like(dM)
    dv_block = torch.zeros_like(v)
    v_block = v.clone()    

    #Start timer
    start_time = time.perf_counter()

    #Block weight matrix
    dM_block.copy_(torch.linalg.matrix_power(dM, n_block))

    #Block teleportation vector
    for i in range(n_block):
        dv_block += v_block
        v_block.copy_(dM @ v_block)        
    dv_block *= (1.0 - d)

    total_time_block = time.perf_counter() - start_time

    #Pagerank algorithm
    for i in range(n_max):
        
        #Timing logic
        total_time1 = time.perf_counter() - start_time
        if total_time1 >= duration:
            break
        else:
            total_time0 = total_time1
        
        #Pagerank iteration
        r0.copy_(torch.matmul(dM_block,r0) + dv_block)


    #Subtract one iteration, since this was the last to occur before break.
    print("Block iterations %d setup in %.12f seconds" % (n_block, total_time_block))
    print("%d iterations achieved in %.12f seconds" % (n_block*(i-1), total_time0))

    for i in range(0,ns,100):
        print("PageRank of vertex %d: %.6f" % (i, r0[i]))

    print("Sum of all pageranks = %.12f." % (torch.norm(r0, p=1).item()))

    np.savetxt('pageranks.txt',r0.cpu().numpy(), fmt="%.15f")
    

#Running code
main()
