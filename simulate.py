# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 10:21:30 2025

@author: larat
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.tsa.stattools import acf

# parameters
L_values = [10, 15, 20] # add values here
J = 1 # coupling constant
beta = np.linspace(0.1, 0.8, 10) # inverse temperatures
Nthermalization = int(10e5) # number of thermalization steps
Nsample = 5000 # number of samples (= size of the Markov chain)

# lookup table for acceptance probabilities
dE_values = np.array([-8, -4, 0, 4, 8]) # possible energy changes
lookup_table = {dE: np.exp(-dE * beta) for dE in dE_values}

#save the results for all lattice sizes
results_dict = {}

def nn_sum(x, i, j, L):
    """
    Inputs:
        x: Spin configuration
        i, j: Indices describing the position of one spin
        L: lattice of size L*L

    Returns:
        Sum of the nearest neighbors of (i, j)
    """
    result = x[(i - 1) % L, j] + x[(i + 1) % L, j] + x[i, (j - 1) % L] + x[i, (j + 1) % L] # mod L due to periodic BC
    
    return int(result)

def total_energy(x):
    """
    Inputs:
        x: Spin configuraion

    Returns:
        Total energy of configuration x.
    """     
    # I use np.roll to avoid for loops. Otherwise it takes longer to compute the energy.
    neighbour_sum = np.roll(x, 1, axis=0) + np.roll(x, -1, axis=0) + np.roll(x, 1, axis=1) + np.roll(x, -1, axis=1)
    return -J * np.sum(x * neighbour_sum) / 2  # division by 2 to avoid double counting


def move(x, M, E, L, t, lookup_table):
    """
    Inputs:
        x: Spin configuration
        M: Magnetization of x
        E: Energy of x
        L: describes the lattice size L*L
        t: index for selecting the temperature from beta
        lookup_table: dictionary with acceptance probabilities for energy changes

    Returns:
        Updated x, M and E after one Monte Carlo move
    """
    # Pick one site at random
    i, j = np.random.randint(0, L), np.random.randint(0, L)

    # Compute the local magnetic field at site (i,j) due to nearest-neighbours
    h_ij = nn_sum(x, i, j, L)
    
    delta_E = 2 * J * x[i, j] * h_ij 
    
    # Use a look-up table
    R = lookup_table[delta_E][t]

    eta = np.random.rand()
    if R > eta or delta_E <= 0:
        # flip the spin
        x[i, j] *= -1
        # update the magnetisation and energy
        M += 2 * x[i, j]  
        E += delta_E  

    return x, M, E

def compute_error(data):
    """
    from http://dx.doi.org/10.1119/1.3247985
    
    Computes error taking autocorrelation into account.

    Inputs:
        data: data points

    Returns:
        the corrected error accounting for autocorrelation.
    
    """
    data = np.asarray(data)
    N = len(data)

    max_lag = N // 10  # limit

    autocorr = acf(data, nlags=max_lag, fft=True) 
    tau = np.sum(autocorr[1:])   
    if tau < 0:
        tau = 0  # Prevent negative effective sample size
        
    var = np.var(data, ddof=1)  # Sample variance

    # Compute effective number of independent samples
    N_eff = N / (1 + 2 * tau)
    corrected_error = np.sqrt(var / N_eff)

    return corrected_error


def critical_temp(beta, arr):
    """    
    Estimates the interval in which the critical temperature lies.
    
    Inputs:
        beta: Array of inverse temperatures
        arr: Array of corresponding values (chi or cv).

    Returns:
        The estimated interval in which the critical temperature lies.
    """
    
    index = np.argmax(arr)

    if index > 0:
        beta_lower = beta[index - 1]
    else:
        beta_lower = None  
    
    if index < len(beta) - 1:
        beta_higher = beta[index + 1]
    else:
        beta_higher = None  # No next value if at end   
        
    if beta_lower is not None and beta_higher is not None:
        print(f"The critical temperature lies between: {1 / beta_higher:.5f} and {1 / beta_lower:.5f}")
        return index  #return the index of the peak. I need this for the last exercise, where I need to choose a temperature lower than T_C.
    else:
        print("The interval for the critical temperature couldn't be determined.")
        
        
def plot_results(M_arr, M_err, E_arr, E_err, chi_arr, cv_arr, L):
    
    """
    Plots all quatities for a given lattice size L.
    """
    
    plot_data = {
        "M_L{}.png".format(L): (M_arr, M_err, "Magnetization per spin"),
        "chi_L{}.png".format(L): (chi_arr, None, "Magnetic susceptibility"),
        "E_L{}.png".format(L): (E_arr, E_err, "Energy per spin"),
        "cv_L{}.png".format(L): (cv_arr, None, "Heat capacity"),
    }

    for filename, (data, err, ylabel) in plot_data.items():
        fig, ax = plt.subplots(figsize = (6,6))
        ax.errorbar(beta, data, yerr=err if err is not None else None, fmt='o-', markersize=3)
        ax.set_xlabel(r'$\beta$')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        plt.savefig(filename)
        plt.close(fig) 
    
    
def plot_results_combined(L_values, beta, results_dict):
    """
    Plots quantities for different lattice sizes L on the same graph for comparison.
    """
    plot_data = {
        "Energy per spin": (0, 1),  # (E_arr, E_err)
        "Magnetization per spin": (2, 3),  # (M_arr, M_err)
        "Magnetic susceptibility": (4, None),  # (chi_arr, None)
        "Heat capacity": (5, None),  # (cv_arr, None)
    }

    for ylabel, (data_idx, err_idx) in plot_data.items():
        plt.figure(figsize=(6, 6))
        
        for L in L_values:
            data = results_dict[L][data_idx] 
            err = results_dict[L][err_idx] if err_idx is not None else None
            plt.errorbar(beta, data, yerr=err, label=f"L={L}", fmt='o-', markersize=3)
        
        plt.xlabel(r'$\beta$')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        filename = f"{ylabel}.png"
        plt.savefig(filename)
        plt.show()


def simulate(L):
    """
    Performs a MC simulation for a square lattice of size LxL and computes absolute magnetization,
    energy, susceptibility, and specific heat over a range of inverse temperatures.

    Inputs:
    L: size of the lattice

    Returns:
        index_peak_chi and index_peak_cv : the index corresponding to peaks of
        the susceptibility and heat capacity plots.
    """
    
    print(f"Running simulation for L = {L}")
    N = L ** 2
    Nsubsweep = 10*N # number of subsweeps

    M_arr = np.zeros_like(beta) # average absolute magnetizations
    E_arr = np.zeros_like(beta) # average energies
    M_err = np.zeros_like(beta) # standard deviations of the magnetizations
    E_err = np.zeros_like(beta) # standard deviations of the energies
    chi_arr = np.zeros_like(beta) # magnetic susceptibilities
    cv_arr = np.zeros_like(beta) # heat capacities

    # calculate the relevant physical quantities for different temperatures
    for t in range(beta.size):
        print('Running for inverse temperature =', beta[t])
        # Generate a random initial configuration
        x = np.random.choice([-1, 1], size=(L, L))
        # compute its magnetisation and energy
        M = np.sum(x)
        E = total_energy(x)

        # thermalisation 
        for _ in range(Nthermalization):
            x, M, E = move(x, M, E, L, t, lookup_table)
        
        # measurement of M and E
        print('Sampling M and E ...')
        M_data = np.zeros(Nsample)
        E_data = np.zeros(Nsample)

        M_data[0] = np.abs(M)/N  # absolute magnetization per spin
        E_data[0] = E/N  # energy per spin

        for n in tqdm(range(1, Nsample)):
            for _ in range(Nsubsweep):
                x, M, E = move(x, M, E, L, t, lookup_table)

            M_data[n] = np.abs(M)/N # absolute magnetization per spin
            E_data[n] = E/N # energy per spin

        # Compute binned means and errors
        M_arr[t] = np.mean(M_data) # average magnetization
        E_arr[t] = np.mean(E_data) # average energy
        M_err[t] = compute_error(M_data) # error considering autocorrelation
        E_err[t] = compute_error(E_data)
        
        # Use the fluctuation dissipation to compute the specific heat and susceptibility from the M and E data
        chi_arr[t] = beta[t] * N * (np.mean((M_data)**2) - np.mean(M_data)**2)
        cv_arr[t] = beta[t]**2 * N**2 * (np.mean((E_data)**2) - np.mean(E_data)**2)
    
    results_dict[L] = (E_arr, E_err, M_arr, M_err, chi_arr, cv_arr)

    print("Using the susceptibility plot:")
    index_peak_chi = critical_temp(beta, chi_arr) # calculate the interval in which T_C lies and get the index of the peak of the chi plot
    print("Using the heat capacity plot:")
    index_peak_cv  = critical_temp(beta, cv_arr)  # calculate the interval in which T_C lies and get the index of the peak of the cv plot
    
    plot_results(M_arr, M_err,E_arr, E_err, chi_arr, cv_arr, L)
    print("-" * 30)
    
    if index_peak_chi == index_peak_cv: # check if the peaks of the chi and cv plots agree.
        return index_peak_chi # if yes then return the index of the peak
    else:
        print("The computation of the critical temperature is not consistent.")
        return None



def M_time(L, beta_index, steps=5000):
    """
    Tracks M over time at a fixed beta such that T < T_C.
     
    Inputs:
    L: lattice size
    beta_index: index of the inverse temperature in a predefined beta array.
    steps : Number of time steps to track the magnetization.
    
    Returns: None
    """
    
    N = L**2
    Nsubsweep = 10*N 
    beta_sim = beta[beta_index]  # Select beta for T < T_C to simulate M
    print(f"Tracking M over time for L={L} at β={beta_sim:.3f}")

    # Initialize system with random spins
    x = np.random.choice([-1, 1], size=(L, L))
    M = np.sum(x)
    E = total_energy(x)

    # Thermalization
    for _ in range(Nthermalization):
        x, M, E = move(x, M, E, L, beta_index, lookup_table)

    # Tracking M over time
    M_time = np.zeros(steps)

    for step in tqdm(range(steps)):
        for _ in range(Nsubsweep):  # Perform multiple Monte Carlo moves per time step
            x, M, E = move(x, M, E, L,  beta_index, lookup_table)
        
        M_time[step] = M / N  # Store magnetization per spin

    # Plot M vs. time
    plt.figure(figsize=(6,6))
    plt.plot(range(0, steps, 10), M_time[::10], 'o', markersize=3, alpha=0.8)  # Plot every 10th step   
    plt.xlabel("Simulation Time")
    plt.ylabel("Magnetization per spin")
    plt.title(f"Magnetization vs. Time for L={L}, β={beta_sim:.3f}")
    plt.grid(True)
    plt.savefig("M_t.png")
    
    
def main():
    for L in L_values:
        index_peak = simulate(L)
        
    plot_results_combined(L_values, beta, results_dict)
        
    if index_peak != None:
        M_time(10, index_peak + 1)

if __name__ == "__main__":
    main()