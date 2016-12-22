#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
import random

# ------------------------------------------------------------------------
# PHY466/MSE485 Atomic Scale Simulations
# Homework 2: Introduction to Molecular Dynamics
# ------------------------------------------------------------------------

from particleset import ParticleSet
"""
The ParticleSet class is designed to hold the position, velocity and accelerations of a set of particles. Initialization methods init_pos_cubic(cube_length) and
init_vel(temperature) are provided for your convenience.

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
# ------------------------------------------------------------------------
def pos_in_box(mypos, box_length):
    """ Return position mypos in simulation box using periodic boundary conditions.
        The simulation box is a cube of size box_length^ndim centered around the originvec{0}. """
     
    new_pos = mypos.copy()
    ndim    = mypos.shape[0] # calculate the number of rows of mypos
    # calculate new_pos
    for i in range(ndim):
        new_pos[i] = new_pos[i] - box_length * round(new_pos[i] / box_length)
    return new_pos
# def pos_in_box

def displacement(iat, jat, pset, box_length):
    """ Return the displacement of the iat th particle relative to the jat th particle.
        Unlike the distance function, here you will return a vector instead of a scalar """
    posi = pos_in_box(pset.pos(iat),box_length)
    posj = pos_in_box(pset.pos(jat),box_length)
    disp = posi.copy()
    L    = box_length / 2
    # calculate displacement of the iat th particle relative to the jat th particle
    # i.e. r_i - r_j
    # be careful about minimum image convention! i.e. periodic boundary
    for i in range(3):
        if posi[i] - posj[i] > L:
            disp[i] = posi[i] - (posj[i] + box_length)
        elif posi[i] - posj[i] < -L:
            disp[i] = (posi[i] + box_length) - posj[i]
        else:
            disp[i] = posi[i] - posj[i]
    return disp
# end def displacement

def distance(iat, jat, pset, box_length):
    """ return the distance between particle i and j according to the minimum image convention. """

    dist = 0.0
    posi = pos_in_box(pset.pos(iat),box_length)
    posj = pos_in_box(pset.pos(jat),box_length)
    dist_xyz = []
    L = box_length / 2 # shorthand for half of box_length
    
    # calculate distance with minimum image convention
    # np.linalg.norm() may be useful here
    for i in range(3):# assume that this is a 3-D problem
        if abs(posi[i] - posj[i]) > L:
            dist_xyz.append(box_length - abs(posi[i] - posj[i]))
        else:
            dist_xyz.append(abs(posi[i] - posj[i]))
    dist = (dist_xyz[0] ** 2 + dist_xyz[1] ** 2 + dist_xyz[2] ** 2) ** 0.5
    return dist

# end def distance

# The Verlet time-stepping algorithm that YOU must write, dt is time step
# ------------------------------------------------------------------------
def verlet_next_pos(pos_t,vel_t,accel_t,dt):
    """
    We want to return position of the particle at the next moment t_plus_dt
    based on its position, velocity and acceleration at time t.  
    """
    pos_t_plus_dt = pos_t.copy()
    pos_t_plus_dt += vel_t * dt + 0.5 * accel_t * (dt**2) 

    return pos_t_plus_dt
# end def verlet_next_pos

def verlet_next_vel(vel_t,accel_t,accel_t_plus_dt,dt):
    """
    We want to return velocity of the particle at the next moment t_plus_dt, 
    based on its velocity at time t, and its acceleration at time t and t_plus_dt
    """
    vel_t_plus_dt = vel_t.copy()
    vel_t_plus_dt += 0.5 * (accel_t + accel_t_plus_dt) * dt
    return vel_t_plus_dt
# end def verlet_next_vel

# We want Lennard-Jones forces. YOU must write this.
# ------------------------------------------------------------------------
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

    force = np.zeros(pset.ndim())
    # calculate force
    for n in range(pset.size()): # n is just for counting number
        if (n != iat):
            disp = displacement(n,iat,pset,box_length) #calculate the displacement from ith atom to any abitrary atom
            dist = distance(n,iat,pset,box_length) # calculate the distance between two atoms
            # calculate the force between these two atoms, using equations from Lesar textbook
            #force_x  = (-24 / dist ** 2) * (2 * (1 / (dist ** 12)) - 1 / (dist ** 6)) * disp[0]
            #force_y  = (-24 / dist ** 2) * (2 * (1 / (dist ** 12)) - 1 / (dist ** 6)) * disp[1]
            #force_z  = (-24 / dist ** 2) * (2 * (1 / (dist ** 12)) - 1 / (dist ** 6)) * disp[2]
            #force[0] += force_x
            #force[1] += force_y
            #force[2] += force_z
            force += (-24 / dist ** 2) * (2 * (1 / (dist ** 12)) - 1 / (dist ** 6)) * disp
    return force
# end def internal_force

def compute_energy(pset,box_length):
    natom = pset.size()  # number of particles
    pos = pset.all_pos() # all particle positions 
    vel = pset.all_vel() # all particle velocies

    tot_kinetic   = 0.0
    tot_potential = 0.0
    # calculate the kinetic energy
    for i in range(natom):
        tot_kinetic += 0.5 * pset.mass() * (vel[i,0] ** 2 + vel[i,1] ** 2 + vel[i,2] ** 2)
    # calculate the potential energy
    for m in range(natom):
        for n in range(m+1, natom):
            if n != m:
                tot_potential += 4.0 * (1.0 / (distance(n, m , pset, box_length) ** 12) - 1.0 / (distance(n, m , pset, box_length) ** 6))

    tot_energy = tot_kinetic + tot_potential
    return [tot_kinetic, tot_potential, tot_energy]
    #return tot_potential
# end def compute_energy

def VMDOut(R):
    outFile=open("G:\UIUC\course\MSE 485\HW2\myTrajectory.xyz","a")
    for i in range(0,len(R)):
        outFile.write(str(i)+" "+str(R[i][0])+" "+str(R[i][1])+" "+str(R[i][2])+"\n")
    outFile.close()
# end def VMDout
    
def compute_temperature(pset, box_length):
    kinetic = compute_energy(pset,box_length)[0]
    return (2 * kinetic / (3 * pset.size()), kinetic)
# end def compute_temperature

def compute_momentum(pset, box_length):
    vel = pset.all_vel()
    natom = pset.size()
    mass = pset.mass()
    momentum = np.zeros(pset.ndim())
    
    for i in range(natom):
        momentum += mass * vel[i,:]

    return momentum
# end def compute_momentum

def RDF(pset, box_length, nbins):
    dr = box_length / (2 * nbins)
    bins = np.zeros(nbins)
    natom = pset.size()

    # calculate the g(r)
    for i in range(natom - 1):
        for j in range(i + 1, natom):
            distance_ij = distance(i, j, pset, box_length)
            if distance_ij < box_length / 2:
                channel = int(distance_ij / dr)
                bins[channel] += 2
            
    # normalize the g(r)
    for num in range(nbins):
        rho = natom / (box_length ** 3)
        V = (4 / 3) * math.pi * ((num + 1) ** 3 - (num) ** 3) * (dr ** 3)
        bins[num] = bins[num] / (natom * V * rho)

    return bins
# end def RDF

def plot_RDF(bins, box_length, nbins):
    dr = box_length / (2 * nbins)
    r_list = []
    for num in range(nbins):
        r_list.append(dr * num)
    r_list = np.array(r_list)
    plt.plot(r_list,bins)
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.title("Pair Correlation Function")
    plt.show()

    
def legal_kvecs(maxk, box_length):
    kvecs = []
    # calculate a list of legal k vectors
    for i in itertools.product(range(maxk[0]),range(maxk[1]),range(maxk[2])):
        if i != (0,0,0):
            temp = np.array(i)
            kvecs.append(temp * 2 * math.pi / box_length)
    return np.array(kvecs)
# end def legal_kvecs

def rhok(kvec, pset):
    value = 0.0
    natom = pset.size()
    #computes \sum_j \exp(i * k \dot r_j)
    for i in range(natom):
        kr  = kvec.dot(pset.pos(i))
        value += np.cos(kr) + 1j * np.sin(kr)
    return value
# end def rhok

def Sk(kvecs, pset):
    sk_list = []
    natom = pset.size()
    for kvec in kvecs:
        r_k = rhok(kvec, pset)
        sk_list.append(abs(r_k) ** 2 / natom)
    """ computes structure factor for all k vectors in kList
     and returns a list of them """
    return np.array(sk_list)
# end def Sk

def plot_Sk(kvecs, sk_list):
    kmags  = [np.linalg.norm(kvec) for kvec in kvecs]
    sk_arr = np.array(sk_list) # convert to numpy array if not already so 
 
    # average S(k) if multiple k-vectors have the same magnitude
    unique_kmags = np.unique(kmags)
    unique_sk    = np.zeros(len(unique_kmags))
    for iukmag in range(len(unique_kmags)):
        kmag    = unique_kmags[iukmag]
        idx2avg = np.where(kmags==kmag)
        unique_sk[iukmag] = np.mean(sk_arr[idx2avg])
    # end for iukmag
 
    # visualize
    plt.plot(unique_kmags,unique_sk)
    plt.xlabel("|k|")
    plt.ylabel("S(|k|)")
    plt.title("Structure factor")
    plt.show()

def vv_correlation(pset, velocity_0):
    natom = pset.size()
    product = 0

    # sum up all atoms velocity correlation
    for i in range(natom):
        product += np.dot(pset.vel(i), velocity_0[i])

    # average over all atoms
    product = product / natom

    return product
# end def vv_correlation



if __name__ == '__main__':

    num_atoms   = 64
    mass        = 48.0
    temperature = 2.4
    density     = 1.0
    box_length  = (num_atoms / density) ** (1.0 / 3.0)
    nsteps      = 1000
    dt          = 0.032
    nbins       = 75
    tmax        = 150 # max time steps for velocity-velocity correlation
    eq_step     = 500
    nu          = 0.01 / dt

    # create and initialize a set of particles
    pset = ParticleSet(num_atoms,mass)
    pset.init_pos_cubic(box_length)
    pset.init_vel(temperature)

    #create empty bins for g(r)
    bins = np.zeros(nbins)

    #calculate the legak kvecs and create an empty sk_list
    kvecs = legal_kvecs([5,5,5], box_length)
    sk_list = np.zeros(kvecs.shape[0])

    #create empty arrays to store temperature
    T = []
    KE = []
    time_step = []

    #create an empty array to store velocity
    vel_pre = []
    vv_correlation_list = np.zeros(tmax)

    # molecular dynamics simulation loop
    for istep in range(nsteps):
   
        # update positions
        for iat in range(num_atoms):
            my_next_pos = verlet_next_pos( pset.pos(iat), pset.vel(iat), pset.accel(iat), dt)
            new_pos = pos_in_box(my_next_pos,box_length)
            pset.change_pos(iat,new_pos)
        # end for iat

        # Q/ When should forces be updated?
        new_accel = pset.all_accel()
	for iat in range(num_atoms):
            new_accel[iat] = internal_force(iat,pset,box_length) * 1 / pset.mass()

        # update velocities
        for iat in range(num_atoms):
            my_next_vel = verlet_next_vel( pset.vel(iat), pset.accel(iat), new_accel[iat], dt )
            pset.change_vel( iat, my_next_vel )
        # end for iat
	
	#update the acceleration
	pset.change_all_accel(new_accel)

	
        #Andersen thermostat
	sigma = (temperature / mass) ** 0.5
	temp_vel = np.zeros(3)
	for iat in range(num_atoms):
            if random.uniform(0,1) < dt * nu:
                temp_vel[0] = random.gauss(0,sigma)
                temp_vel[1] = random.gauss(0,sigma)
                temp_vel[2] = random.gauss(0,sigma)
                pset.change_vel( iat, temp_vel )
        # end for Andersen thermostat
	
        # calculate properties of the particles
        #print(istep, compute_energy(pset,box_length))
        #print(compute_momentum(pset,box_length))
        #print(istep, pset.pos(0))

        time_step.append(istep)
        T.append(compute_temperature(pset, box_length)[0])
        KE.append(compute_temperature(pset, box_length)[1])

        """if istep > eq_step - tmax:
            if len(vel_pre) < tmax:
                vel_pre.append(pset.all_vel())
            else:
                vel_pre = vel_pre[1:]
                vel_pre.append(pset.all_vel())"""

        
        #calculate g(r) and S(k)
        if istep > eq_step:
            sk_list += Sk(kvecs, pset)
            bins += RDF(pset, box_length, nbins)
            """if istep <= nsteps - tmax:
                for num in range(tmax):
                    vv_correlation_list[num] += vv_correlation(pset, vel_pre[tmax-num-1])"""

    # end for istep

    # plot g(r) and S(k)
    bins = bins / (nsteps - eq_step)
    sk_list = sk_list / (nsteps - eq_step)
    plot_Sk(kvecs, sk_list)
    plot_RDF(bins, box_length, nbins)
    
    # plot T versus time_step
    T = np.array(T)
    time_step = np.array(time_step)
    plt.plot(time_step, T)
    plt.xlabel('time step')
    plt.ylabel('T')
    plt.show()
    # end of plot T

    # plot Kinetic energy versus time_step
    KE = np.array(KE)
    plt.plot(time_step, KE)
    plt.xlabel('time step')
    plt.ylabel('kinetic energy')
    plt.show()
    #end of plot Kinetic

    """# calculate diffusion function and plot vv_correlation vs time
    vv_correlation_list = vv_correlation_list / (nsteps - eq_step - tmax )
    vv_correlation_list = vv_correlation_list / vv_correlation_list[0] # normalize
    diffusion_constant = sum(vv_correlation_list) * dt
    print(diffusion_constant)
    x_axis = dt * np.array(range(tmax))
    y_axis = vv_correlation_list
    plt.plot(x_axis, y_axis)
    plt.xlabel('t')
    plt.ylabel('<v(0)v(t)')
    plt.show()"""

# end __main__
