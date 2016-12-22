#!/usr/bin/env python
from __future__ import print_function
import numpy as np

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
    L = box_length / 2 # shorthand for half of box_length
    # calculate new_pos
    for i in range(ndim):
        if new_pos[i] > L or new_pos[i] < -L: # check every dimension of the postion of a particle, map it back if not in the box
            n = int(new_pos[i]/L)
            new_pos[i] = new_pos[i] - n*L
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

    return pos_t_plus_dt
# end def verlet_next_pos

def verlet_next_vel(vel_t,accel_t,accel_t_plus_dt):
    """
    We want to return velocity of the particle at the next moment t_plus_dt, 
    based on its velocity at time t, and its acceleration at time t and t_plus_dt
    """
    vel_t_plus_dt = vel_t.copy()
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

    return force
# end def internal_force

def compute_energy(pset,box_length):
    natom = pset.size()  # number of particles
    pos = pset.all_pos() # all particle positions 
    vel = pset.all_vel() # all particle velocies

    tot_kinetic   = 0.0
    tot_potential = 0.0 

    tot_energy = tot_kinetic + tot_potential
    return (tot_kinetic, tot_potential, tot_energy)
# end def compute_energy

if __name__ == '__main__':
    box_length = 1.0
    pset = ParticleSet(2,1)
    pset.change_pos(0,np.array([ 0.499, 0.000, 0.000 ]) )
    pset.change_pos(1,np.array([ -.499, 0.000, 0.000 ]) )
    print( distance(0,1,pset,box_length) )
    print( displacement(0,1,pset,box_length))
 
    pset.change_pos(0,np.array([ 0.001, 0.000, 0.000 ]) )
    pset.change_pos(1,np.array([ -.001, 0.000, 0.000 ]) )
    print( distance(0,1,pset,box_length) )
    print( displacement(0,1,pset,box_length))
# end __main__
