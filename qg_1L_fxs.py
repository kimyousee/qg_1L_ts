from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import cmath

def Z(x, y): ##zero matrix
    Z = PETSc.Mat().create()
    Z.setSizes([x, y]); Z.setFromOptions( ); Z.setUp()
    Z.zeroEntries()
    Z.assemble()
    return Z

def vec(init, incr, size): #used for making y. similar to [start:end:incr]
    vec = PETSc.Vec().createMPI(size, bsize=PETSc.DECIDE, comm=PETSc.COMM_WORLD)
    start,end = vec.getOwnershipRange()
    vec[start] = init + incr*start; start +=1
    for i in range(start,end):
        vec[i] = vec[i-1]+incr
    vec.assemble()
    return vec

def Dy(hy, Ny, Dy2=False): #used for making Dy and Dy2
    Dy = PETSc.Mat(); Dy.create()
    Dy.setSizes([Ny-1, Ny+1]); Dy.setType('aij'); Dy.setFromOptions( ); Dy.setUp()
    start, end = Dy.getOwnershipRange()

    if Dy2 == False:    denom = hy*2;  a = -1/denom; b = 0;        c = 1/denom
    else:               denom = hy**2; a =  1/denom; b = -2/denom; c = 1/denom

    if start == 0:
        Dy[0, 0:3] = [a, b, c]
        start += 1
    if end == Ny-1:
        Dy[Ny-2, Ny-2:Ny+1] = [a, b, c]
        end -= 1

    for i in xrange(start, end):
        Dy[i, i:i+3] = [a, b, c]

    Dy.assemble()
    return Dy

def Phi(y, Ly, Lj, Ny, f0=1.e-4, g0=9.81):
    Phi = PETSc.Vec().createMPI(Ny+1, comm=PETSc.COMM_WORLD)
    start, end = Phi.getOwnershipRange()
    ystant, yend = y.getOwnershipRange()

    for i in xrange(start,end):
        Phi[i] = np.tanh((y[i]-Ly/2)/Lj)
        
    Phi.assemble()
    Phi = -g0/f0*0.1*Phi
    return Phi

def U(Dy, Phi, Ny, y, Ly, Lj, g0=9.81, f0=1.e-4):
    U = PETSc.Vec().createMPI(Ny-1, comm=PETSc.COMM_WORLD)
    Dy.mult(Phi, U)
    U = -U
    U.assemble()

    return U

def etaB(y):
    etaB = PETSc.Vec().createMPI(y.getSize(), comm=PETSc.COMM_WORLD)
    start,end = etaB.getOwnershipRange()
    for i in xrange(start,end): etaB[i] = 0*y[i]
    etaB.assemble()
    return etaB

def Lap(Dy2, kx2, Ny):
    Lap = PETSc.Mat().create()
    Lap.setSizes([Ny-1,Ny-1]); Lap.setFromOptions(); Lap.setUp()

    Dy2Box = PETSc.Mat().createAIJ([Ny-1,Ny-1])
    bs,be = Dy2Box.getOwnershipRange()
    if bs == 0:
        Dy2Box[0,0:2] = Dy2[0,1:3]
        bs += 1
    if be == Ny-1:
        Dy2Box[Ny-2,Ny-3:Ny-1] = Dy2[Ny-2,Ny-2:Ny]
        be -= 1
    for i in xrange(bs,be):
        Dy2Box[i,i-1:i+2] = Dy2[i,i:i+3]
    Dy2Box.assemble()
    Lap = Dy2Box.copy()
    start,end = Lap.getOwnershipRange()

    for i in xrange(start,end):
        Lap[i,i] = Dy2Box[i,i] - kx2
    Lap.assemble()
    return Lap

def B(Lap, F0, Ny):
    B = PETSc.Mat().create()
    B.setSizes([Ny-1,Ny-1]); B.setFromOptions(); B.setUp()

    tempLap = -1*Lap
    B=tempLap.copy()
    start,end = B.getOwnershipRange()
    for i in xrange(start,end):
        B[i,i] = F0 - Lap[i,i]

    B.assemble()
    return B

def diag(v):
    size = v.getSize()
    M = PETSc.Mat().create()
    M.setSizes([size,size]); M.setFromOptions(); M.setUp()

    start,end = M.getOwnershipRange()
    for i in xrange(start,end): M[i,i] = v[i]

    M.assemble()
    return M

def A(U,Lap, F0,Dy,Q,Ny): #A = [diag(U(2:Ny))*(F0*I - Lap)] - diag(dQ(2:Ny))
    A = PETSc.Mat().create()
    A.setSizes([Ny-1,Ny-1]); A.setFromOptions(); A.setUp()

    Ud = diag(U)

    Lap = -1*Lap
    tempLap = Lap.copy()
    lstart,lend = tempLap.getOwnershipRange()
    for i in xrange(lstart,lend):
        tempLap[i,i] = F0 + Lap[i,i]
    tempLap.assemble()

    temp1=Ud.matMult(tempLap)
    temp1.assemble() #temp1 = diag(U(2:Ny))*(F0*I - Lap)

    temp2 = PETSc.Vec().createMPI(Ny-1, comm=PETSc.COMM_WORLD)
    Dy.mult(Q,temp2)
    temp2.assemble()

    dQ = diag(temp2)

    A = temp1-dQ
    A.assemble()
    return A
