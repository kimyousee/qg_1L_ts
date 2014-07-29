from __future__ import division
import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI
import numpy as np
from numpy import linalg as la
import qg_1L_fxs as fx
import matplotlib.pyplot as plt

Print = PETSc.Sys.Print # For printing with only 1 processor

rank = PETSc.COMM_WORLD.Get_rank()
opts = PETSc.Options()
nEV = opts.getInt('nev', 5)

def solve_eigensystem(A,B,grow,freq,mode,problem_type=SLEPc.EPS.ProblemType.GNHEP):

    # Set up slepc, generalized eig problem
    E = SLEPc.EPS(); E.create(comm=SLEPc.COMM_WORLD)

    E.setOperators(A,B); E.setDimensions(nEV, PETSc.DECIDE)
    E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_IMAGINARY)
    E.setFromOptions()
    
    E.solve()

    nconv = E.getConverged()
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()

    if nconv <= nEV: evals = nconv
    else: evals = nEV
    for i in range(evals):
        eigVal = E.getEigenvalue(i)

        E.getEigenvector(i,vr,vi)

        start,end = vi.getOwnershipRange()
        if start == 0: mode[0,i,cnt] = 0; start+=1
        if end == Ny: mode[Ny,i,cnt] = 0; end -=1

        for j in range(start,end):
            mode[j,i,cnt] = 1j*vi[j]; mode[j,i,cnt] = vr[j]


    E.getEigenvector(0,vr,vi)
    return vr

if __name__ == '__main__':

    Ny = opts.getInt('Ny',10)#400)

    Ly = 350e03
    Lj = 20e03
    Hm = 500

    f0 = 1.e-4
    bet = 0
    g0 = 9.81

    y = fx.vec(0, float(Ly)/float(Ny), Ny+1)
    hy = float(Ly)/float(Ny)

    Dy = fx.Dy(hy, Ny)
    Dy2 = fx.Dy(hy, Ny, Dy2=True)

    Phi = fx.Phi(y, Ly, Lj, Ny)
    U = fx.U(Dy, Phi, Ny, y, Ly, Lj, g0,f0)
    etaB = fx.etaB(y)

    F0 = f0**2/(g0*Hm)
    dkk = 2e-2

    kk = np.arange(dkk,2+dkk,dkk)/Lj
    nk = len(kk)

    # Temporary vector to store Dy2*Phi
    temp = PETSc.Vec().createMPI(Ny-1,comm=PETSc.COMM_WORLD)
    Dy2.mult(Phi,temp)
    temp.assemble()
    temp2 = PETSc.Vec().createMPI(Ny+1, comm=PETSc.COMM_WORLD)
    ts,te = temp2.getOwnershipRange()
    if ts == 0: temp2[0] = temp[0]; ts+=1
    if te == Ny+1: temp2[Ny] = temp[Ny-2]; te -= 1
    for i in range(ts,te):
        temp2[i] = temp[i-1]
    temp2.assemble()

    Q = temp2 - F0*Phi + bet*y + f0/Hm*etaB #Ny+1
    # Used to store im(eigVal), re(eigVal), and eigVec
    grow = np.zeros(nk)
    freq = np.zeros(nk)
    mode = np.zeros([Ny+1,nEV,nk], dtype=complex)
    # grOut = open('grow.dat', 'wb')
    # frOut = open('freq.dat', 'wb')
    # mdOut = open('mode.dat', 'wb')
    cnt = 0

    for kx in kk[0:nk]:

        kx2=kx**2
        # Build matrices using petsc matrices
        Lap = fx.Lap(Dy2, kx2, Ny)
        B = fx.B(Lap, F0, Ny)
        A = fx.A(U,Lap, F0,Dy,Q,Ny)

        # Solves system with slepc EPS, returns eigvec1
        sn = solve_eigensystem(A,B,grow,freq,mode)

        # Using time stepping
        A = -1j*A
        f = A.getVecRight()
        
        dt = 1e0
        dto2 = dt/2
        tol = 1e-12

        AeT = PETSc.Mat().createAIJ([Ny-1,Ny-1])
        AeT.setUp(); AeT.assemble()
        ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
        ksp.setOperators(B-dto2*A)
        ksp.setTolerances(1e-16)
        pc = ksp.getPC()
        pc.setType('none')
        ksp.setFromOptions()

        btemp = B+dto2*A
        btemp.assemble()

        bcol = PETSc.Vec().createMPI(Ny-1)
        xcol = PETSc.Vec().createMPI(Ny-1)
        bcol.setUp(); xcol.setUp()

        sc,ec = bcol.getOwnershipRange()
        for i in range(0,Ny-1):
            bcol[sc:ec] = btemp[sc:ec,i]
            bcol.assemble(); xcol.assemble()
            ksp.solve(bcol,xcol)
            AeT[sc:ec,i] = xcol[sc:ec]

        AeT.assemble()

        test = PETSc.Mat().createAIJ(B.getSize())
        test = B+dt/2*A
        test.assemble()

        count = 1
        error = 1
        max_it = 1e5

        while (error > tol) and (count < max_it):
            sn = AeT*sn
            if count % 100 == 0:
                sn = sn/sn.norm() #norm is petsc's norm for Vec
                Asn = PETSc.Vec().createMPI(Ny-1)
                Bsn = PETSc.Vec().createMPI(Ny-1)
                A.mult(sn,Asn)
                B.mult(sn,Bsn)

                scatter, AsnSeq = PETSc.Scatter.toAll(Asn)
                im = PETSc.InsertMode.INSERT_VALUES
                sm = PETSc.ScatterMode.FORWARD
                scatter.scatter(Asn,AsnSeq,im,sm)
                scatter, BsnSeq = PETSc.Scatter.toAll(Bsn)
                scatter.scatter(Bsn,BsnSeq,im,sm)
                lam = np.mean(AsnSeq/BsnSeq)

                error = la.norm(AsnSeq.array-lam*BsnSeq.array)
            count +=1
        if rank == 0:
            grow[cnt] = kx*lam.real
            freq[cnt] = -kx*lam.imag
        cnt = cnt+1

    # grOut.close(); frOut.close(); mdOut.close()
    # Plotting
    if rank == 0:
        ky = np.pi/Ly
        plt.plot(kk*Lj,grow*3600*24)
        plt.ylabel('1/day')
        plt.xlabel('k')
        plt.title('Growth Rate: 1-Layer QG')
        plt.savefig('Grow1L_QG.eps', format='eps', dpi=1000)
        # plt.show()
