# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:49:14 2020

@author: Saint8312
"""

from mpi4py import MPI
import numpy as np

#if rank == 0:
#    data = {'a': 7, 'b': 3.14}
#    comm.send(data, dest=1, tag=11)
#elif rank == 1:
#    data = comm.recv(source=0, tag=11)
#    print(rank)
#    print(data)
    

def f_mul(a,b):
    return a*b

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    X = np.arange(22)
    Y = np.arange(23)[1:23]
    sigma = 0
    if rank == 0:
        for i in range(1,size):
            y_ = comm.recv(source=i)
            sigma+=y_
        print(rank, "result = ",sigma)

    elif rank != 0:
        y = f_mul(X[rank-1], Y[rank-1])
        print(rank, y)
        comm.send(y, dest=0)
        
    
    sigma=comm.bcast(sigma, root=0)
    print("after broadcast =",sigma, "rank = ",rank)
    
#    if rank == 0:
#        data = np.arange(11)
#    else:
#        data = None
#    data = comm.scatter(data, root=0)
#    print(data)
#    
    
    ###### algorithm to split processes to cores,just som algebra operations
    cores = 4
    processes = 12
    dist = np.array([processes//cores]*cores)
    if processes%cores > 0:
        remainder = processes%cores
        dist[:remainder] += 1
    print(dist, np.sum(dist), rank)