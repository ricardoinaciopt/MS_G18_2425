# 2024-25 M&S Project - Group 18

## Authors
- Ricardo Inácio
- Tomás Maciel
- Carlota Silva

# Simulating Biased Wealth Accumulation in a Society for Testing Mitigation Models and Fairness

## Overview

The experiment simulates a society divided into two groups:
- **Group A** (privileged): Individuals in this group have better access to wealth accumulation opportunities.
- **Group B** (unprivileged): Individuals in this group have fewer opportunities to accumulate wealth.

Additionally, gender-based income inequality is introduced, where women in both groups have reduced wealth growth. The primary goal of this simulation is to demonstrate how societal biases lead to wealth disparities, which in turn cause biased predictions when using machine learning models to predict class membership.

The simulation generates wealth data for individuals in both groups, which is then used to train a machine learning model. The performance of this model is evaluated, showing poorer performance in predicting Group B membership compared to Group A, reflecting societal biases.

## Key Classes and Attributes (MESA Simulation)

### `PersonAgent`
Represents an individual in the simulation. Each individual belongs to either Group A or Group B and has attributes that define their wealth, opportunities, sex, and life events.

#### Attributes:
- `group`: The group to which the agent belongs (A or B).
- `wealth`: Initial wealth assigned to the agent. Group A typically starts with higher wealth than Group B.
- `opportunities`: A boolean indicating whether the agent has opportunities to increase wealth. Group A has a higher probability of having opportunities.
- `sex`: The agent’s gender, either Male ("M") or Female ("F"). Females earn less due to gender-based income inequality.
- `age`: The agent's current age, which increases over time.
- `age_of_death`: The expected age at which the agent will die. This value is affected by the group and the agent’s wealth.
- `diseases`: The number of diseases an agent has contracted, which impacts their life expectancy.
- `disease_probability`: The probability that the agent contracts a disease, which is higher for Group B.
- `job`: A boolean indicating if the agent has a job. Agents can lose their jobs with a probability that is higher for Group B.
- `career_years`: The number of years the agent has been working. This increases the wealth accumulation rate after certain milestones.
- `has_car`: A boolean indicating whether the agent owns a car.
- `has_house`: A boolean indicating whether the agent owns a house.
- `reproduction_chance`: The chance that the agent reproduces when they interact with a compatible agent.
- `child_possibility`: The number of children the agent can have during its lifetime.

#### Methods:
- `step()`: Simulates the agent’s life, including wealth accumulation, job loss, disease acquisition, and reproduction.
- `move()`: Moves the agent to a new position in the simulation grid.

### `SocietyModel`
Represents the entire society. It initializes the grid with agents, runs the simulation, and collects data on wealth accumulation and group dynamics.

#### Attributes:
- `num_agents_a`: The number of agents in Group A.
- `num_agents_b`: The number of agents in Group B.
- `group_a_wealth_rate`: The rate of wealth accumulation for agents in Group A.
- `group_b_wealth_rate`: The rate of wealth accumulation for agents in Group B.
- `grid`: The spatial grid where agents are located.
- `schedule`: The schedule controlling agent activation in the simulation.
- `max_steps`: The maximum number of simulation steps.
- `datacollector`: Collects data on agent wealth and group dynamics throughout the simulation.

#### Methods:
- `create_agents()`: Creates agents in both groups with specified attributes, including wealth, opportunities, sex, and age of death.
- `step()`: Advances the simulation by one step, during which all agents perform their actions.
- `average_wealth()`: Returns the average wealth of all agents in the simulation.
- `group_average_wealth(group)`: Returns the average wealth of agents in the specified group (A or B).
- `train_model_on_collected_data()`: Trains a RandomForest model on the wealth data collected from the simulation and evaluates the model's performance.

## Simulation Process

1. **Initialization**:
   - The simulation begins with a specified number of agents in Group A and Group B. Agents are placed on a grid with random positions.
   - Group A agents start with more wealth and better opportunities for wealth accumulation. Group B agents start with less wealth and fewer opportunities.

2. **Wealth Accumulation**:
   - Each agent accumulates wealth based on their opportunities and sex. Men accumulate wealth faster than women, and Group A agents accumulate wealth faster than Group B.
   - Agents may buy cars and houses if they accumulate sufficient wealth, which impacts their reproduction chances and job security.

3. **Life Events**:
   - Agents age over time, contract diseases, lose jobs, and reproduce based on probabilities influenced by their group and wealth.
   - Agents die when they reach their age of death, and new agents are born if reproduction conditions are met.

4. **Wealth Transfer**:
   - Agents transfer wealth to other agents during interactions, simulating market-like dynamics.

5. **Data Collection**:
   - Data on agent wealth, group, sex, and life events are collected throughout the simulation using the `datacollector`.

6. **Machine Learning Model**:
   - After the simulation completes, a RandomForest classifier is trained on the collected wealth data to predict the group (A or B) an agent belongs to.
   - The performance of the model is evaluated using metrics like accuracy, precision, recall, F1-score, and ROC AUC.

## Fairness Analysis

The project includes a `fairness` folder containing Jupyter notebooks that analyze the fairness of the trained models. These notebooks evaluate the models using various fairness metrics, such as Equal Opportunity Difference (EOD) and Disparate Misclassification Rate (DMR).

## NetLogo Simulation

A similar experiment is conducted in NetLogo by importing the respective file. The NetLogo simulation follows these steps:

1. **Start the simulation**:
   - Open NetLogo.
   - Click on `File` -> `Open` and select the `simulation.nlogo` file.
   - Click on `Setup` to initialize the simulation.
   - Use the sliders from the GUI to adjust parameters to explore different scenarios.
   - Adjust the `ticks` to set the simulation speed.
   - Click on `Go` to start the simulation.

2. **Collect simulation data**:
   - When the simulation ends, click on `Export` to save the output data as `simulation_results.csv` in the current directory.

3. **Model Training**:
   - You can train and test the models with the data collected from the NetLogo simulation by running the following command in the root directory:
   ```bash
   python model_pipeline.py simulation_results.csv
   ```

___

# Research Questions

## How bias could affect the model results?

If you train a model on wealth accumulation data from a society with privileged and unprivileged groups, and then use that model to predict an individual's class based on attributes related to biased wealth accumulation, the results would likely exhibit several problematic biases:

## Overall Results

The model would likely show high accuracy overall, but this accuracy would be misleading due to the underlying biases in the training data. The model would essentially learn and perpetuate the existing societal biases in wealth accumulation.

## Results for Privileged Group

For the privileged group, the model would likely show:

- High accuracy in predictions
- Low false negative rate (rarely misclassifying privileged individuals as unprivileged)
- High precision (most predictions of privileged status would be correct)

This is because the model would learn to associate attributes correlated with wealth accumulation to the privileged class, reinforcing existing advantages.

## Results for Unprivileged Group

For the unprivileged group, the model would likely exhibit:

- Lower accuracy compared to the privileged group
- Higher false positive rate (more frequently misclassifying unprivileged individuals as privileged)
- Lower recall (failing to identify many truly unprivileged individuals)

The model would struggle to accurately classify unprivileged individuals, as their attributes related to wealth accumulation may not fit the patterns learned from the biased training data.

## Impact of Class Imbalance

If the number of elements in the minority (unprivileged) class is equal to or lower than the majority class, additional issues arise:

- **Equal class sizes**: The model may show somewhat balanced performance between classes, but still exhibit the biases mentioned above.
- **Lower minority class size**: The model would likely show:
  - Even higher overall accuracy, masking poor performance on the minority class
  - Increased bias towards the majority (privileged) class
  - Very low recall for the minority class
  - Potential for completely ignoring the minority class in extreme cases

These issues stem from the model having less data to learn patterns for the minority class, exacerbating the existing biases in wealth accumulation[1][2].

## Implications

This scenario highlights several critical issues in machine learning and fairness:

1. **Perpetuation of historical biases**: The model would reinforce existing societal inequalities by learning from biased historical data[3].
2. **Misleading performance metrics**: Overall accuracy would be a poor indicator of the model's fairness or real-world utility.
3. **Disparate impact**: The model's predictions would likely have disproportionately negative effects on the unprivileged group[4].
4. **Need for bias mitigation**: Techniques such as data resampling, algorithmic fairness constraints, or causal modeling may be necessary to create a more equitable model[5].

#### Citations:

[1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5546759/

[2] https://towardsdatascience.com/unveiling-the-power-of-bias-adjustment-enhancing-predictive-precision-in-imbalanced-datasets-ecad1836fc58?gi=a08f9b712988

[3] https://textbook.coleridgeinitiative.org/chap-bias.html

[4] https://www.mdpi.com/2227-7080/8/4/68

[5] https://www.sciencedirect.com/science/article/pii/S0167268122000580

[6] https://towardsdatascience.com/how-biased-is-your-regression-model-4ef6c1495b77

[7] https://www.sciencedirect.com/science/article/pii/S0010027723000823

[8] https://journals.sagepub.com/doi/10.1177/23328584241258741

