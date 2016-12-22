#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools

# ------------------------------------------------------------------------
# PHY466/MSE485 Atomic Scale Simulations
# Homework 2: Introduction to Molecular Dynamics
# ------------------------------------------------------------------------

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
# ------------------------------------------------------------------------
def pos_in_box(mypos, box_length):
    """ Return position mypos in simulation box using periodic boundary conditions. The simulation box is a cube of size box_length^ndim centered around the origin vec{0}. """
    
    new_pos = mypos.copy()
    ndim    = mypos.shape[0]
    # calculate new_pos
    
    # map the position back into the simulation box
    for idim in range(ndim):
        if new_pos[idim] >= 0:
            new_pos[idim] -= int(new_pos[idim]/box_length + 0.5) * box_length
        else:
            new_pos[idim] -= int(new_pos[idim]/box_length - 0.5) * box_length
    # end for
    
    return new_pos
# def pos_in_box

def displacement(iat, jat, pset, box_length):
    """ Return the displacement of the iat th particle relative to the jat th particle. Unlike the distance function, here you will return a vector instead of a scalar """
    posi = pset.pos(iat)
    posj = pset.pos(jat)
    disp = posi.copy()
    # calculate displacement of the iat th particle relative to the jat th particle
    # i.e. r_i - r_j
    # be careful about minimum image convention! i.e. periodic boundary
    ndim = len(posi) # the dimension
    
    # calculate displacement with minimum image convention
    for idim in range(ndim):
        disp[idim] = disp[idim] - posj[idim]
        disp[idim] -= int(disp[idim]/box_length)*box_length
        assert -box_length<disp[idim]<box_length   # the displacement in one dimension shouldn't exceed box_length
        if disp[idim] > 0.5*box_length:
            disp[idim] = disp[idim] - box_length
        elif disp[idim] < -0.5*box_length:
            disp[idim] = disp[idim] + box_length
    # end for
    
    return disp
# end def distance

def distance(iat, jat, pset, box_length):
    """ return the distance between particle i and j according to the minimum image convention. """

    dist = 0.0
    # calculate distance with minimum image convention
    # np.linalg.norm() may be useful here
    dist = np.linalg.norm(displacement(iat, jat, pset, box_length))
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
    ndim = len(pos_t) # dimension
    
    # Verlet
    for idim in range(ndim):
        pos_t_plus_dt[idim] += vel_t[idim]*dt + 0.5*accel_t[idim]*dt*dt
    # end for
    
    return pos_t_plus_dt
# end def verlet_next_pos

def verlet_next_vel(vel_t,accel_t,accel_t_plus_dt,dt):
    """
    We want to return velocity of the particle at the next moment t_plus_dt, 
    based on its velocity at time t, and its acceleration at time t and t_plus_dt
    """
    vel_t_plus_dt = vel_t.copy()
    ndim = len(vel_t)
    
    # Verlet
    for idim in range(ndim):
        vel_t_plus_dt[idim] += 0.5*dt*(accel_t[idim]+accel_t_plus_dt[idim])
    # end for
    
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
    
    force  = np.zeros(pset.ndim())
    natoms = pset.size()   # number of atoms
    ndim   = pset.ndim()   # dimension
    
    # calculate force
    for jat in range(natoms):
        # avoid calculating the mistaken self-exerting force
        if jat == iat:
            continue
        disp   = displacement(iat,jat,pset,box_length) # the displacement of atom i relative to atom j
        dist   = distance(iat,jat,pset,box_length)     # the distance between atom i and j
        ri     = 1.0/dist                 # r_inverse, i.e. 1/r
        r2i    = ri*ri                    # 1/r^2
        r6i    = r2i*r2i*r2i              # 1/r^6
        rforce = 24*r6i*ri * (2*r6i-1)    # radial force on atom i
        
        # add rforce into force
        for dim in range(ndim):
            force[dim] += rforce*disp[dim]*ri
        # end for
    # end for
    return force
# end def internal_force

# calculate the Lennard-Johns potential between iat th atom and jat th atom
def LJ_Potential(iat,jat,pset,box_length):
    dist = distance(iat,jat,pset,box_length) # the distance between the two atoms
    ri   = 1.0/dist                          # r_inverse, i.e. 1/r
    r6i  = ri**6                             # 1/r^6
    
    potential = 4.0*r6i*(r6i-1)
    
    return potential
#end def L-J_Potential

def compute_energy(pset,box_length):
    natom = pset.size()  # number of particles
    vel = pset.all_vel() # all particle velocities
    mass = pset.mass()   # particle mass
    ndim = pset.ndim()   # dimension

    tot_kinetic   = 0.0
    tot_potential = 0.0 

    # calculate total kinetic energy
    for atom in range(natom):
        for idim in range(ndim):
            tot_kinetic += 0.5*mass*vel[atom][idim]*vel[atom][idim]
    # end calculating total kinetic energy
    
    # calculate total potential energy
    for iat in range(natom):
        for jat in range(iat):
            tot_potential += LJ_Potential(iat,jat,pset,box_length)
    # end calculating total potential energy
    
    tot_energy = tot_kinetic + tot_potential
    
    temperature = tot_kinetic / (1.5*pset.size()) # instantaneous temperature
    return (tot_kinetic, tot_potential, tot_energy, temperature)
# end def compute_energy

def compute_momentum(pset,box_length):
    natom = pset.size()  # number of particles
    vel = pset.all_vel() # all particle velocities
    mass = pset.mass()   # particle mass
    ndim = pset.ndim()   # dimension
    
    tot_momentum = np.zeros(ndim) # initializing momentum
    
    # calculating momentum
    for iatom in range(natom):
        for idim in range(ndim):
            tot_momentum[idim] += mass*vel[iatom][idim]
        # end for
    # end for
    return tot_momentum
    
# end def compute_momentum

# calculate the pair correlation function g(r)
def compute_PCF(pset, box_length, nhis):
    rmax = box_length/2     # maximum r considered in g(r)
    dr = rmax/nhis          # resolution
    num_atoms = pset.size() # number of atoms
    g = np.zeros(nhis)      # the histogram of g(r)
    
    #scan over the possitions of the particles to determine the histogram
    for iat in range(0, num_atoms-1):
        for jat in range(iat+1, num_atoms):
            rij = distance(iat, jat, pset, box_length)
            if (rij < rmax):
                ihis = int(rij/dr) # determine the location in the histogram
                g[ihis] += 2       # contribution of particle iat and jat
            # end if
        # end for
    # end for
    
    return g
# end def compute_PCF

# plot the pair correlation function g(r)
def plot_PCF(nhis, dr, g):
    r = np.zeros(nhis)
    for ihis in range(nhis):
        r[ihis] = (ihis+1)*dr
    plt.plot(r,g)
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.title("Pair Correlation Function")
    plt.show()

# end def plot_PCF

# compute a list of the legal k-vectors for computing S(k)
def legal_kvecs(maxk, pset, box_length):
    kvecs=[]
    # calculate a list of legal k vectors
    for vec in itertools.product(range(maxk+1),repeat=pset.ndim()):
        if np.linalg.norm(vec)<1e-5:
            continue
        kvecs.insert(0, np.array(vec) * (2*math.pi/box_length))
    return np.array(kvecs)
# end def legal_kvecs

# Fourier transform to calculate rho_k
def rhok(kvec, pset):
    value = 0.0
    #computes \sum_j \exp(i * k \dot r_j)
    for iatom in range(pset.size()):
        kr = np.dot(kvec, pset.pos(iatom)) # dot product of k and atom position
        value += np.cos(kr) + 1j*np.sin(kr)
    return value
# end def

# calculate the structure factor S(k)
def Sk(kvecs, pset):
    """ computes structure factor for all k vectors in kList
     and returns a list of them """
    sk_list = np.zeros(len(kvecs))
    index = 0
    for ivec in kvecs:
        rho_k = rhok(ivec, pset)
        sk_list[index] += np.abs(rho_k)**2/pset.size()
        index += 1
    return sk_list
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
    plt.xlabel("k")
    plt.ylabel("S(|k|)")
    plt.title("Structure Factor")
    plt.show()
#end def plot_Sk

# calculate the velocity-velocity correlation function at a given time step
def VVCorr(v0, pset):
    VVC = 0.0 # velocity-velocity correlation
    for iat in range(pset.size()):
        VVC += np.dot(v0[iat], pset.vel(iat))
    VVC /= pset.size() # average VCC over atoms
    return VVC
# end def VVCorr

# plot velocity-velocity correlation function over time
def plot_VVCorr(VVC, dt):
    tmax=len(VVC)
    time = np.array(range(tmax))*dt # real time
    plt.plot(time, VVC)
    plt.xlabel("time")
    plt.ylabel("velocity-velocity correlation")
    plt.title("Velocity-Velocity Correlation Function")
    plt.xlim([-dt*tmax*0.01,dt*tmax*1.01])
    plt.show()
# end def plot_VVCorr

if __name__ == '__main__':

    num_atoms   = 64
    mass        = 48.0
    temperature = 0.728
    box_length  = 4.2323167
    nsteps      = 1000
    step_equil  = 200 # we believe that the system reaches equilbrium after step_equil
    dt          = 0.01

    # create and initialize a set of particles
    pset = ParticleSet(num_atoms,mass)
    pset.init_pos_cubic(box_length)
    pset.init_vel(temperature)
    
    # parameters useful in calculating g(r)
    nhis = 75           # the number of intervals in the histogram
    g = np.zeros(nhis)  # pair correlation function g(r)
    rmax = box_length/2 # maximum r considered in g(r)
    dr = rmax/nhis      # resolution
    V = box_length**3   # volume of the box
    
    # parameters useful in calculating S(k)
    maxk = 5               # the maximum of n_x, n_y and n_z in the k vector
    S_size = (maxk+1)**3-1 # the size of S list or the number
                           # of legal k vectors, eliminating vector [0,0,0]
    S = np.zeros(S_size)   # structure factor
    kvecs = legal_kvecs(maxk,pset,box_length) # legal k vectors
    
    # parameters useful in calculating v-v correlation and diffusion constant
    v_record =[]                 # store the velocities of all the atoms
    N_stored = nsteps-step_equil # the number of VVCs needed to be stored
    VVC = np.zeros(N_stored)     # velocity-velocity correlation function
    D = 0.0 # diffusion constant
    
    # molecular dynamics simulation loop
    for istep in range(nsteps):
        
        """# output momentum into external file
        tot_momentum = compute_momentum(pset,box_length)
        outFile  = open("momentum.dat", "a")
        for idim in range(pset.ndim()):
            outFile.write(str(tot_momentum[idim])+" ")
        outFile.write(str(np.linalg.norm(tot_momentum))+"\n")
        outFile.close()
        
        # output momentum of a single particle
        pnum = 0
        outFile = open("momentum_singlepart.dat", "a")
        for idim in range(pset.ndim()):
            outFile.write(str(mass*pset.vel(pnum)[idim])+" ")
        outFile.write(str(mass*np.linalg.norm(pset.vel(pnum)))+"\n")
        outFile.close()"""
        
        # update accelerations/forces
        for iat in range(num_atoms):
            iaccel = internal_force(iat,pset,box_length)
            for idim in range(pset.ndim()):
                iaccel[idim] /= mass
            pset.change_accel(iat,iaccel)
        # end for iat

        """calculate properties of the particles, printing kinetic, potential
           and total energies, along with the system temperature
        """
        
        # update positions
        for iat in range(num_atoms):
            my_next_pos = verlet_next_pos( pset.pos(iat), pset.vel(iat), pset.accel(iat), dt)
            new_pos = pos_in_box(my_next_pos,box_length)
            pset.change_pos(iat,new_pos)
        # end for iat
        
        # Q/ When should forces be updated?
        new_accel = pset.all_accel()

        # update velocities
        for iat in range(num_atoms):
            my_next_vel = verlet_next_vel( pset.vel(iat), pset.accel(iat), new_accel[iat], dt )
            pset.change_vel( iat, my_next_vel )
        # end for iat
        
        """After equilibrium, calculate g(r) and S(k)
           record velocities for calculating VVC
        """
        if istep >= step_equil:
            # calculate g(r)
            g_renewal = compute_PCF(pset, box_length, nhis)
            for ihis in range(nhis):
                g[ihis] += g_renewal[ihis]
            # end for
            
            # calculate S(k)
            S_renewal = Sk(kvecs, pset)
            for ik in range(S_size):
                S[ik] += S_renewal[ik]
            # end for
        
            # record velocities of all particles for calculating VVC
            v_record.insert(istep-step_equil, pset.all_vel())
        
    # end for istep
    
    # normalize and plot g(r)
    for ihis in range(nhis):
        dV = ((ihis+1)**3-ihis**3)*dr**3 
        dV *= (4/3)*math.pi # the volume of the shell considered
        num_ideal = num_atoms*dV/V  # number of ideal gas atoms in dV
        g[ihis] /= (num_ideal*num_atoms*(nsteps-step_equil)) # normalization
    # end for
    
    plot_PCF(nhis, dr, g)
    
    # average and plot S(k)
    for ik in range(S_size):
        S[ik] /= (nsteps-step_equil)
    plot_Sk(kvecs, S)
    # end for
    
    """calculate velocity-velocity correlation(unnormalized)
       VVC[t] is defined as <V(t0)V(t0+t)>
       where the angular bracket means averaging over t0 and particles
    """
    for t in range(N_stored):
        for t0 in range(N_stored-t):
            for iat in range(pset.size()):
                VVC[t] += np.dot(v_record[t0][iat], v_record[t0+t][iat])
            # end for
        # end for
        VVC[t] /= pset.size()  # average over all the particles
        VVC[t] /= (N_stored-t) # average over all the t0's
    # end for
    
    # normalize velocity-velocity correlation
    vvc0 = VVC[0] # normalize by dividing <v(t0)v(t0)>
    for t in range(N_stored):
        VVC[t] /= vvc0
    # end for
    
    # plot velocity-velocity correlation function
    plot_VVCorr(VVC, dt)
    
    # calculate diffusion constant
    for itau in range(N_stored):
        D += VVC[itau]*dt
    # end for
    print("Diffusion Constant = ", D)

# end __main__
