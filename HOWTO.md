# Setup and Run the Simulation Models

# Mesa Simulation

## Prerequisites

Ensure the following are installed on your system:

### Python

- Version 3.7 or later

### Required Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `mesa`
- `matplotlib`
- `seaborn`
- `joblib`

## Running the Simulation

### Navigate to the project directory:
```bash
cd /path/to/your/files
```

### Start the simulation:
Execute the `simulation.py` script:
```bash
python simulation.py
```
- This launches an interactive Mesa server accessible at [http://127.0.0.1:8521](http://127.0.0.1:8521).
- Adjust the frames per second (FPS) in the GUI to set the simulation speed, and then click 'Start' to begin.
- Use the GUI to adjust parameters, explore different scenarios and visualize the simulation.

### Collect simulation data:
- The simulation saves output data as `merged_data.csv` in the current directory.

## Outputs

- Trained models are saved in the `models/` directory.
- Evaluation results (metrics and classification reports) are saved in the `results/` directory.


# Netlogo Simulation
