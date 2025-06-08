# MICRO-515 EvoRob final project - Co-Evolution of Brain & Morphology in a Multilegged AntRobot

This project explores how jointly evolving a robot’s body and controller affects locomotion in MuJoCo’s Ant environment.

## Key Features
- **Variable Morphology**  
  - Cylindrical torso length ∈ [0.26 m, 2.25 m]  
  - Up to 4 legs per side, each with two actuated segments ∈ [0.15 m, 0.35 m]  
- **Genotype Encoding (1 028 genes)**  
  - **1 003 genes**: MLP controller (43 inputs → 17 hidden → 16 outputs)  
  - **8 genes**: binary leg‐presence switches  
  - **16 genes**: leg‐segment lengths  
  - **1 gene**: torso length

## Methods
- **CMA-ES (single‐objective)**  
  - Maximize average forward speed over 5 parallel trials  
  - Population = 40, Generations = 100  
- **NSGA-II (multi‐objective)**  
  - Objectives: maximize speed/distance & minimize control cost  
  - Population = 40, Generations = 140  
- **EA Hyperparameters**  
  - Crossover Cr = 0.07, Mutation F = 0.05  

## Results
- Evolved robots often exhibited jerky, unbalanced gaits, early termination, or circular spins.  
- NSGA-II runs sometimes “optimized” away all legs to minimize control cost; CMA-ES runs frequently plateaued with negligible speed gains.

## Conclusions & Next Steps
- Co‐optimizing morphology and controller in a high‐dimensional, discrete search space led to unstable solutions.  
- Future directions: smoother morphology encodings, controller pre‐training, refined objectives, and larger populations.

📂 [Videos & Plots](https://github.com/federock02/EPFL-MICRO515-EvolutionaryRobotics-FinalProject/blob/f18972f192bcc637e201659d41876789f78d4b6d/videos&plots)  |  🔗 [Full Report](#https://github.com/federock02/EPFL-MICRO515-EvolutionaryRobotics-FinalProject/blob/main/EvoRob-FinalProjectReport.pdf)

## How to Run

**Requirements:**
* Python >= 3.8

To install the required pip packages

````
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
````

To check if your installation was successful run the following command:

````
source .venv/bin/activate
python TestScript.py
````

To run a training, after modifying the hyperparameters in `exercise3.py`, run

````
python3 exercise3.py
````