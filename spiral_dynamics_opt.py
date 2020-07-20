# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:03:45 2020

@author: Saint8312
"""
from mpi4py import MPI
from mpitest import data_splitter
import numpy as np
import sobol_seq #if deterministic-ish is needed


import time

def mat_R_ij(dim, i, j, theta):
    c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
    R = np.zeros((dim, dim))
    if dim == 2:
        R = np.array([[c, -s],[s, c]])
    else:
        for a in range(dim):
            for b in range(dim):
                if ((a==i) and (b==i)) or ((a==j) and (b==j)):
                    R[a][b] = c
                elif ((a==j) and (b==i)):
                    R[a][b] = s
                elif ((a==i) and (b==j)):
                    R[a][b] = -s
                else:
                    if(a==b):
                        R[a][b] = 1
                    else :
                        R[a][b] = 0
    return R


def transformation_matrix(mat_R_ij, dim, r, theta):
    '''
    generate the matrix of S(n) = r(n)*R(n), where r = contraction matrix, R = rotation matrix
    '''
    # generate r matrix
    mat_r = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            if i==j:
                mat_r[i][j]=r
                    
    # generate R(n) matrix
    mat_Rn = np.zeros((dim, dim))
#    c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
    if(dim<=2):
        c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
        rotate = np.array([[c, -s],[s, c]])
        contract = np.array([[r,0],[0,r]])
        Sn = np.matmul(contract, rotate)
    elif(dim>2):
        R=np.identity(dim)
        for i in range(dim-1):
            for j in range(i):
                R=np.matmul(R, mat_R_ij(dim, i,j, theta))
        mat_Rn = R
        # S(n) = r(n)*R(n)
        Sn = np.matmul(mat_r, mat_Rn)
    return Sn

def spiral_dynamics_optimization(F, S, R, m, theta, r, kmax, domain, *f_args, log=True):
    '''
    Function F optimization using spiral dynamics -> rotating & contracting points
    '''
    
    x_dim = domain.shape[0]
    
    if log:
        print("Init points = ",m)
        print("Theta = ",theta)
        print("r = ",r)
        print("iter_max = ",kmax)
        print("dimension = ",x_dim)
        print("domain = ",domain)
        print()
    
    # generate m init points using random uniform between function domain
#    x = np.array([[np.random.uniform(domain[j][0], domain[j][1]) for j in range(x_dim)] for i in range(m)])

    x = sobol_seq.i4_sobol_generate(x_dim, m) #now using sobol
    for i in range(x_dim):
        x.T[i] = x.T[i]*(domain[i][1]-domain[i][0])+domain[i][0]

    f = np.array([F(x_, *f_args) for x_ in x])

    # search the minimum/maximum (depends on the problem) of f(init_point), in this case, max
    x_star = x[np.argmax(f)]
            
    # rotate the points
    for k in range(kmax):
        x = np.array([rotate_point(x[i], x_star, S, R, x_dim, r, theta) for i in range(len(x))])
        f = np.array([F(x_, *f_args) for x_ in x])
        x_star_next = x[np.argmax(f)]
        if(F(x_star_next) > F(x_star)):
            x_star = np.copy(x_star_next)
        if log:
            print("Iteration\tx_star\tF(x_star)")
            print(str(k)+" "+str(x_star)+" "+str(F(x_star, *f_args)))
    return x_star, F(x_star, *f_args)


def spiral_dynamics_optimization_mpi(F, S, R, m, theta, r, kmax, domain, mpi_params, *f_args, log=True):
    '''
    Function F optimization using spiral dynamics -> rotating & contracting points
    mpi_params:
        {'comm', 'rank', 'size'}
    '''
    comm = mpi_params['comm']
    rank = mpi_params['rank']
    size = mpi_params['size']

    
    print(rank)

#    if rank == 0:
    x_dim = domain.shape[0]
    
    if log:
        print("Init points = ",m)
        print("Theta = ",theta)
        print("r = ",r)
        print("iter_max = ",kmax)
        print("dimension = ",x_dim)
        print("domain = ",domain)
        print()
    
    # generate m init points using random uniform between function domain
    x = np.array([[np.random.uniform(domain[j][0], domain[j][1]) for j in range(x_dim)] for i in range(m)])

#    x = sobol_seq.i4_sobol_generate(x_dim, m) #now using sobol
#    for i in range(x_dim):
#        x.T[i] = x.T[i]*(domain[i][1]-domain[i][0])+domain[i][0]

    f = np.array([F(x_, *f_args) for x_ in x])

    # search the minimum/maximum (depends on the problem) of f(init_point), in this case, max
    x_star = x[np.argmax(f)]
            
    # rotate the points
    for k in range(kmax):
        x = np.array([rotate_point(x[i], x_star, S, R, x_dim, r, theta) for i in range(len(x))])
        f = np.array([F(x_, *f_args) for x_ in x])
        x_star_next = x[np.argmax(f)]
        if(F(x_star_next) > F(x_star)):
            x_star = np.copy(x_star_next)
        if log:
            print("Iteration\tx_star\tF(x_star)")
            print(str(k)+" "+str(x_star)+" "+str(F(x_star, *f_args)))
    print(x_star, F(x_star, *f_args))
    return x_star, F(x_star, *f_args)

rotate_point = lambda x, x_star, S, R, x_dim, r, theta: np.matmul( S(R, x_dim, r, theta), x ) - np.matmul( ( S(R, x_dim, r, theta) - np.identity(x_dim) ), x_star )

if __name__ == "__main__":
    def unit_tests_single_core():
        start = time.time()
#        f = [lambda x : x[0]**2 + x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) -0.4*np.cos(4*np.pi*x[1]) +0.7]
#        F = lambda x : 1/( 1 + sum([abs(f_(x)) for f_ in f]) )
#        domain = np.array([[-100,100]]*2)    
#        x_star, F_x_star = spiral_dynamics_optimization(F, transformation_matrix, mat_R_ij, 20, 45, 0.95, 500, domain, log=False)
#        print("\nFinal value of x, F(x) is : ",x_star,",",F_x_star,"")
#        print("time = ",time.time()-start)
        
        f = [lambda x : 2*x[0]+x[1]+x[2]+x[3]+x[4]-6,
             lambda x : 2*x[1]+x[0]+x[2]+x[3]+x[4]-6,
             lambda x : 2*x[2]+x[0]+x[1]+x[3]+x[4]-6,
             lambda x : 2*x[3]+x[0]+x[2]+x[1]+x[4]-6,
             lambda x : x[0]*x[1]*x[2]*x[3]*x[4] - 1
             ]
        F = lambda x : 1/( 1 + sum([abs(f_(x)) for f_ in f]) )
        domain = np.array([[-10,10]]*5)  
        start = time.time()
        x_star, F_x_star = spiral_dynamics_optimization(F, transformation_matrix, mat_R_ij, 100, 45, 0.95, 500, domain, log=False)
        print("\nFinal value of x, F(x) is : ",x_star,",",F_x_star,"")
        print("time = ",time.time()-start)
    
    def unit_tests_multi_cores():
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        
        f = [lambda x : 2*x[0]+x[1]+x[2]+x[3]+x[4]-6,
             lambda x : 2*x[1]+x[0]+x[2]+x[3]+x[4]-6,
             lambda x : 2*x[2]+x[0]+x[1]+x[3]+x[4]-6,
             lambda x : 2*x[3]+x[0]+x[2]+x[1]+x[4]-6,
             lambda x : x[0]*x[1]*x[2]*x[3]*x[4] - 1
             ]
        F = lambda x : 1/( 1 + sum([abs(f_(x)) for f_ in f]) )
        domain = np.array([[-10,10]]*5)  
        start = time.time()
        mpi_params = {'comm':comm,'rank':rank,'size':size}
        x_star, F_x_star = spiral_dynamics_optimization_mpi(F, transformation_matrix, mat_R_ij, 10, 45, 0.95, 500, domain, mpi_params, log=False)
        print("\nFinal value of x, F(x) is : ",x_star,",",F_x_star,"")
        print("time = ",time.time()-start)
    
    def unit_tests_multi_cores_2():
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        

        f = [lambda x : 2*x[0]+x[1]+x[2]+x[3]+x[4]-6,
             lambda x : 2*x[1]+x[0]+x[2]+x[3]+x[4]-6,
             lambda x : 2*x[2]+x[0]+x[1]+x[3]+x[4]-6,
             lambda x : 2*x[3]+x[0]+x[2]+x[1]+x[4]-6,
             lambda x : x[0]*x[1]*x[2]*x[3]*x[4] - 1
             ]
        F = lambda x : 1/( 1 + sum([abs(f_(x)) for f_ in f]) )
        domain = np.array([[-10,10]]*5)  
        
        
        ###assign parameters
        S = transformation_matrix
        R = mat_R_ij
        m = 100
        theta = 45
        r = 0.95
        kmax = 500
        log = False
        
        ####SPO starts here
        x = None
        x_split = None
        x_dim = None
        if rank==0:
            start = time.time()

            x_dim = domain.shape[0]
            
            if log:
                print("Init points = ",m)
                print("Theta = ",theta)
                print("r = ",r)
                print("iter_max = ",kmax)
                print("dimension = ",x_dim)
                print("domain = ",domain)
                print()
            
            # generate m init points using random uniform between function domain
            x = sobol_seq.i4_sobol_generate(x_dim, m) #now using sobol
            for i in range(x_dim):
                x.T[i] = x.T[i]*(domain[i][1]-domain[i][0])+domain[i][0]
#            x = np.array([[np.random.uniform(domain[j][0], domain[j][1]) for j in range(x_dim)] for i in range(m)])
            x_split = data_splitter(x, size) #split data
        x_split = comm.bcast(x_split, root=0) #broadcast x vectors
        x_dim = comm.bcast(x_dim, root=0)
        
        x_pr = x_split[rank-1] #get data based on rank
        fvals = np.array([F(x_) for x_ in x_pr]) 
        x_star_pr = x_pr[np.argmax(fvals)]
        x_star_pr = comm.gather(x_star_pr, root=0)
        
        # search the minimum/maximum (depends on the problem) of f(init_point), in this case, max
        if rank==0:
            fvals = np.array([F(x_) for x_ in x_star_pr])
            x_star = x_star_pr[np.argmax(fvals)]
        else:
            x_star = None
        x_star = comm.bcast(x_star, root=0)
#        # rotate the points
        for k in range(kmax):
            x_pr = np.array([rotate_point(x_pr[i], x_star, S, R, x_dim, r, theta) for i in range(len(x_pr))])
            fvals = np.array([F(x_) for x_ in x_pr])
            x_star_next_pr = x_pr[np.argmax(fvals)]
#            print(rank, x_star_next_pr)
            x_star_next_pr = comm.gather(x_star_next_pr, root=0)
            ###gather xstars from all ranks
            if rank==0:
#                print(x_star_next_pr)
                fvals = np.array([F(x_) for x_ in x_star_next_pr])
                x_star_next = x_star_next_pr[np.argmax(fvals)]
                if(F(x_star_next) > F(x_star)):
                    x_star = np.copy(x_star_next)
            else:
                x_star = None
            x_star = comm.bcast(x_star, root=0)
#            print(rank, x_star, F(x_star))
            if log:
                print("Iteration\tx_star\tF(x_star)")
                print(str(k)+" "+str(x_star)+" "+str(F(x_star)))        
#            else:
#                x_star_next=None
#            x_star = comm.bcast(x_star, root=0)
#        
        if rank==0:
            print(x_star, F(x_star))
            print("time = ",time.time()-start)

            
#    unit_tests_single_core()
    unit_tests_multi_cores_2()

    
    