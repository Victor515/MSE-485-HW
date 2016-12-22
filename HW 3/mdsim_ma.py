#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import pylab, math, cmath, random
# ------------------------------------------------------------------------
# PHY466/MSE485 Atomic Scale Simulations
# Homework 2: Introduction to Molecular Dynamics
# ------------------------------------------------------------------------
from mpmath import *
mp.pretty = False
from particleset import ParticleSet
"""
The ParticleSet class is designed to hold the position, velocity and accelerations of a set of particles. Initialization methods init_pos_cubic(cube_length) and init_vel(temperature) are provided for your convenience.

pset = ParticleSet(natom) will initialize a particle set pset
pset.size() will return the number of particles in pset

| --------------------------------------------------------- | ------------------- |
|                   for access of                           |    use method       |
| --------------------------------------------------------- | ------------------- |
| all particle positions in an array of shape (natom,ndim)  |    pset.all_pos()   |
| all particle velocities                                   |    pset.all_vel()   |
| all particle accelerations                                |    pset.all_accel() |
| particle i position in an array of shape (ndim)           |    pset.pos(i)      |
| particle i velocity                                       |    pset.vel(i)      |
| particle i acceleration                                   |    pset.accel(i)    |
| --------------------------------------------------------- | ------------------- |

| ----------------------------- | ------------------------------------ | 
|           to change           |             use method               |
| ----------------------------- | ------------------------------------ |
| all particle positions        |  pset.change_all_pos(new_pos_array)  |
| particle i position           |  pset.change_pos(i,new_pos)          |
| ditto for vel and accel       |  pset.change_*(i,new_*)              |
| ----------------------------- | ------------------------------------ |
"""

# Routines to ensure periodic boundary conditions that YOU must write.
def pos_in_box(mypos, box_length):
    """ Return position mypos in simulation box using periodic boundary conditions. The simulation box is a cube of size box_length^ndim centered around the origin vec{0}. """   
    new_pos = mypos.copy()
    ndim    = mypos.shape[0]
    for i in range(ndim):
        new_pos[i]=new_pos[i]-box_length*nint(new_pos[i]/box_length)
    return new_pos
# def pos_in_box

def displacement(iat, jat, pset, box_length):
    """ Return the displacement of the iat th particle relative to the jat th particle. Unlike the distance function, here you will return a vector instead of a scalar """
    pos1 = pset.pos(iat)
    pos2 = pset.pos(jat)
    ndim = pos1.shape[0]
    # calculate displacement of the iat th particle relative to the jat th particle
    for i in range(ndim):
        pos1[i] = pos1[i] - box_length*nint((pos1[i]-pos2[i])/box_length)
    disp = pos1 - pos2

    return disp
# end def distance

def distance(iat, jat, pset, box_length):
    """ return the distance between particle i and j according to the minimum image convention. """
    pos1 = pset.pos(iat)
    pos2 = pset.pos(jat)
    ndim = pos1.shape[0]
    # calculate distance with minimum image convention
    # np.linalg.norm() may be useful here
    for i in range(ndim):
        pos1[i] = pos1[i] - box_length*nint((pos1[i]-pos2[i])/box_length)
    dist = np.linalg.norm(pos1-pos2)

    return dist

# end def distance

# The Verlet time-stepping algorithm 
def verlet_next_pos(pos_t,vel_t,accel_t,dt):
    """
    We want to return position of the particle at the next moment t_plus_dt
    based on its position, velocity and acceleration at time t.  
    """
    pos_t_plus_dt = pos_t.copy()
    pos_t_plus_dt += vel_t*dt + 0.5*accel_t*dt*dt
    return pos_t_plus_dt
# end def verlet_next_pos

def verlet_next_vel(vel_t,accel_t,accel_t_plus_dt,dt):
    """
    We want to return velocity of the particle at the next moment t_plus_dt, 
    based on its velocity at time t, and its acceleration at time t and t_plus_dt
    """
    vel_t_plus_dt = vel_t.copy()
    vel_t_plus_dt += 0.5*(accel_t+accel_t_plus_dt)*dt
    return vel_t_plus_dt
# end def verlet_next_vel

# We want Lennard-Jones forces. YOU must write this.
# ------------------------------------------------------------------------
def Lennard_Jones(r):
    r6 = r*r*r*r*r*r 
    r12 = r6*r6
    return 4*(1.0/r12 - 1.0/r6)
def internal_force(iat,pset,box_length):
    """
    We want to return the force on atom 'iat' when we are given a list of 
    all atom positions. Note, pos is the position vector of the 
    1st atom and pos[0][0] is the x coordinate of the 1st atom. It may
    be convenient to use the 'displacement' function above. For example,
    disp = displacement( 0, 1, pset, box_length ) would give the position
    of the 1st atom relative to the 2nd, and disp[0] would then be the x coordinate
    of this displacement. Use the Lennard-Jones pair interaction. Be sure to avoid 
    computing the force of an atom on itself.
    """

    pos = pset.all_pos()  # positions of all particles
    mypos = pset.pos(iat) # position of the iat th particle
    nparticle = pos.shape[0]
    force = np.zeros(pset.ndim())
    f = np.zeros(pset.ndim())
    # calculate force
    for j in range(nparticle):
        if (j != iat):
            # The Lennard-Jones Potential
            r = distance(iat, j, pset, box_length)
            r_vector = displacement(iat, j, pset, box_length)
            r6i = r*r*r*r*r*r;
            r12i = r6i*r6i;
            f = (48/r12i/r-24/r6i/r)/r*r_vector
            force += f

    return force
# end def internal_force

def compute_energy(pset,box_length,mass):
    natom = pset.size()  # number of particles
    pos = pset.all_pos() # all particle positions 
    vel = pset.all_vel() # all particle velocies

    tot_kinetic   = 0.0
    tot_potential = 0.0 

    for i in range(natom):
        tot_kinetic += 0.5*mass*(vel[i,0]*vel[i,0]+vel[i,1]*vel[i,1]+vel[i,2]*vel[i,2])

    for i in range(0,natom-1):
        for j in range(i+1,natom):
            tot_potential += Lennard_Jones(distance(i, j, pset, box_length))

    tot_energy = tot_kinetic + tot_potential
    #return tot_energy 
    return (tot_kinetic, tot_potential, tot_energy)
# end def compute_energy

def compute_momentum(pset):
    momentum = np.zeros(3)
    natom = pset.size()  
    vel = pset.all_vel()    
    mass = pset.mass()
    for i in range (natom):
        momentum[0] += mass*vel[i][0]
        momentum[1] += mass*vel[i][1]
        momentum[2] += mass*vel[i][2]
    return momentum

def pair_correlation(pset, box_length, maxbin):
    natom = pset.size()  
    box_half =  box_length* 0.5
    delta_r  = float(box_half/maxbin)
    hist = np.zeros(maxbin+1)
    gr=[]
    for i in range(0,natom-1):
        for j in range(i+1,natom):
            rij = distance(i, j, pset, box_length)
            bin = nint(rij/delta_r)
            if (bin <= maxbin):
                hist[bin] += 2
    for k in range(0,len(hist)):
        rho = float(natom/(box_length**3))
        delta_V = float(4/3)*math.pi*((k+1)**3 - k**3)*(delta_r**3)
        gr.append(hist[k]/(natom*delta_V*rho))
    return gr

def legal_kvecs(maxk,box_length):
    kvecs = np.zeros(((maxk+1)**3,3))
    t = 0
    for i in range(0,maxk+1):
        for j in range(0,maxk+1):
            for k in range(0,maxk+1):
                kvecs[t][0] = float(2*math.pi/box_length)*i
                kvecs[t][1] = float(2*math.pi/box_length)*j
                kvecs[t][2] = float(2*math.pi/box_length)*k
                t += 1
    return np.array(kvecs)

def rhok(kvec, pset):
    rho = np.zeros(2)
    natom = pset.size()
    pos = pset.all_pos() 
    for i in range(natom):
        rho[0] += cmath.exp(-1j*(pos[i][0]*kvec[0] + pos[i][1]*kvec[1] + pos[i][2]*kvec[2]))
        rho[1] += cmath.exp(1j*(pos[i][0]*kvec[0] + pos[i][1]*kvec[1] + pos[i][2]*kvec[2]))
    return rho

def Sk(kvecs, pset):
    n_k = len(kvecs)
    natom = pset.size()
    sk_list = np.zeros(n_k)
    for i in range(n_k):
        kvec = kvecs[i]
        rho = rhok(kvec, pset)
        sk_list[i] = rho[0]*rho[1]/natom
    return sk_list

def velocity_correlation(v1,pset):
    v2 = pset.all_vel()
    vcorr = 0.0
    natom = pset.size()
    for i in range(0,natom):
        vcorr += v1[i][0]*v2[i][0] + v1[i][1]*v2[i][1] + v1[i][2]*v2[i][2]
    vcorr = float(vcorr/natom)
    return vcorr

def VMDOut(pset,i):
    natom = pset.size()  
    pos = pset.all_pos()
    outFile=open("myTrajectory.xyz","a")
    outFile.write("\n"+"Timestep: "+str(i)+"\n")
    for i in range(0,natom):
        outFile.write(str(i)+" "+str(pos[i][0])+" "+str(pos[i][1])+" "+str(pos[i][2])+"\n")
    outFile.close()

if __name__ == '__main__':
	
    num_atoms   = 64
    mass        = 48.0
    temperature = 0.728
    #temperature = 0.5
    #box_length  = 4
    box_length  = 4.2323167
    nsteps      = 1000
    nstart      = 300
    dt          = 0.01
    maxbin = 500
    maxk = 5
    eta = 0.15/dt
    temp_dis = 0.728
    #temp_dis = 0.5
    
    acceleration=np.zeros((1,3))
    new_pos = np.zeros((num_atoms,3))
    RDF = np.zeros(501)
    sk_list = np.zeros((maxk+1)*(maxk+1)*(maxk+1))
    radius = np.linspace(0,box_length/2,501)
    timeseries = np.linspace(0,(nsteps)*dt,nsteps)
    v_cor = []
    temp = []
    k_energy = []
    momen_x = []
    momen_y = []
    momen_z = []
    diffusivity = 0

    pset = ParticleSet(num_atoms,mass)
    pset.init_pos_cubic(box_length)
    pset.init_vel(temperature)
    vel_init = pset.all_vel()
    kvecs = legal_kvecs(maxk,box_length)
    #outFile=open("myTrajectory.xyz","w")
    #outFile.write(str(num_atoms))
    #outFile.close()

    for istep in range(nsteps):
        #print(pset.pos(0))
        #print(compute_energy(pset,box_length,mass))
        #print(istep, compute_energy(pset,box_length,mass))
        #VMDOut(pset,istep)
        d = compute_energy(pset,box_length,mass)
        temperature = d[0]/1.5/num_atoms
        temp.append(temperature)
        k_energy.append(d[0])
        momentum = compute_momentum(pset)
        momen_x.append(momentum[0])
        momen_y.append(momentum[1])
        momen_z.append(momentum[2])
        print(istep, momentum)
        # end print
        v_cor.append(velocity_correlation(vel_init,pset))
        # start force calculation
        for iat in range(num_atoms):
            if (istep != 0):
                acceleration = internal_force(iat,pset,box_length)/mass
                pset.change_accel( iat, acceleration )
            my_next_pos = verlet_next_pos( pset.pos(iat), pset.vel(iat), pset.accel(iat), dt)
            new_pos[iat] = pos_in_box(my_next_pos,box_length)
        pset.change_all_pos(new_pos)
        old_accel = pset.all_accel()
        for iat in range(num_atoms):
            acceleration = internal_force(iat,pset,box_length)/mass
            pset.change_accel( iat, acceleration )
        for iat in range(num_atoms):
            my_next_vel = verlet_next_vel( pset.vel(iat), old_accel[iat], pset.accel(iat), dt )
            pset.change_vel( iat, my_next_vel )
        # end for iat
        if (istep >= nstart):
            RDF += pair_correlation(pset, box_length, maxbin)
            sk_list += Sk(kvecs, pset)
    # end for istep

    #velocity correlation
    v_correlation = [x/float(v_cor[0]) for x in v_cor]
    for i in range(nsteps-1):
        diffusivity += dt/2*(v_correlation[i]+v_correlation[i+1])
    print ('diffusivity = ', diffusivity)

    #KS 
    sk_list = sk_list/(nsteps-nstart)
    kmags  = [np.linalg.norm(kvec) for kvec in kvecs]
    sk_arr = np.array(sk_list) # convert to numpy array if not already so 
    # average S(k) if multiple k-vectors have the same magnitude
    unique_kmags = np.unique(kmags)
    unique_sk    = np.zeros(len(unique_kmags))
    for iukmag in range(len(unique_kmags)):
        kmag    = unique_kmags[iukmag]
        idx2avg = np.where(kmags==kmag)
        unique_sk[iukmag] = np.mean(sk_arr[idx2avg])

# end __main__
