# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:49:14 2020

@author: Saint8312
"""

from mpi4py import MPI
import numpy as np
import time
    
def process_splitter(processes, cores):
    ###### algorithm to split processes to cores,just som algebra operations
    distribution = np.array([processes//cores]*cores)
    if processes%cores > 0:
        remainder = processes%cores
        distribution[:remainder] += 1
    length = len(distribution)
    split_positions = np.zeros(length-1)
    x = distribution[0]
    split_positions[0] = x
    for i in range(1, length-1):
        x += distribution[i]
        split_positions[i] = x
    dist_idxes = np.array_split(x, np.array(split_positions, dtype="int"))
    return dist_idxes
    
def data_splitter(data, cores):
    ###### algorithm to split data to cores,identical to process_splitter, each data element corresponds to one process
    processes = len(data)
    distribution = np.array([processes//cores]*cores)
    if processes%cores > 0:
        remainder = processes%cores
        distribution[:remainder] += 1
    new_data = []
    length = len(distribution)
    x = 0
    for i in range(length):
        temp_data = data[x : x + distribution[i]]
        x += distribution[i]
        if len(temp_data)>0:
            new_data.append(temp_data)
    return new_data
        
def f_mul(a,b):
    return a*b

def dot(A,B):
    ###pure single core func
    return sum(A[i]*B[i] for i in range(len(A)))



if __name__ == "__main__":
    
    def unit_test_single_core():
        start = time.time()
        length = int(1e7)
        y = np.linspace(0,1, length)
        sum_r = dot(y,y)
        print("sum r = ",sum_r)
        print("total time = ",time.time()-start)
        
    def unit_test_p2p():
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        A = np.arange(23)
        B = np.arange(24)[1:24]
        sigma = 0
        
        start_time = 0
        X = None; Y = None
        if rank == 0:
            start_time = time.time()
            #split the data
            X = data_splitter(A, size-1) #exclude 0th processor
            Y = data_splitter(B, size-1)
            print(X,Y)
            print("dot result = ",np.dot(A,B))
            
        #broadcast the splitted_data
        X = comm.bcast(X, root=0) 
        Y = comm.bcast(Y, root=0)
    #    start_time = comm.bcast(start_time, root=0)
        
        ###arbitrary operations start here
        if rank != 0:
            y=0
            for j in range(len(X[rank-1])):
                y += f_mul(X[rank-1][j], Y[rank-1][j])
                print("j",j, X[rank-1][j], Y[rank-1][j], y, rank)
            comm.send(y, dest=0)
                
        if rank == 0:
            for i in range(1, size):
                y = comm.recv(source=i)
                print("recv = ",y)
                sigma += y
            print("parallel result = ",sigma)
            print("time = ",time.time()-start_time)
    
    def unit_test_collective_comm():
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
                
        y=0
#        r=0
        if rank==0:
            start = time.time()
            length = int(1e7)
            y = np.linspace(0,1, length)
            y = data_splitter(y, size)
            print("data length = ",len(y))
        y = comm.bcast(y, root=0)
#        if rank!=0:
        s_local = time.time()
        r = dot(y[rank-1],y[rank-1])
        print(rank, r, time.time()-s_local)
        r = comm.gather(r, root=0)
        
        if rank == 0:
            print("r = ",r, sum(r), len(r))
            print("total time = ",time.time()-start)
    
    '''
    main actually starts here
    '''
    unit_test_collective_comm()
#    unit_test_single_core()