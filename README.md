# 2D Ising Model Simulation

This project investigates the thermodynamic behavior of the 2D Ising model on a square lattice using the Metropolis-Hastings Monte Carlo algorithm with single-spin flip dynamics.

## Overview

- **Model:** 2D square lattice with periodic boundary conditions  
- **Algorithm:** Metropolis-Hastings  
- **Lattice sizes:** $ L = 10, 15, 20 $
- **Observables:**  
  - Magnetization $\langle|M|\rangle $ 
  - Energy $\langle E \rangle $
  - Magnetic susceptibility $\chi$  
  - Heat capacity $C_V$  
- **Goal:** Estimate the critical temperature $T_C and analyze finite-size effects.

## Key Results

- Estimated critical temperature $T_C \in [2.045\,K,\ 3\,K] $
- Sharper phase transition observed with increasing system size
- Sign flips of magnetization at $ T < T_C $ due to finite-size effects

## Features

- Fluctuation-dissipation relations used for $ \chi $ and $ C_V $
- Precomputed Boltzmann factors for efficient simulation
- Time evolution of magnetization visualized

## Future Work

- Implement cluster algorithms (e.g., Wolff or Swendsen–Wang) to overcome critical slowing down.
