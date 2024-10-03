import mesa
import os
import json
import numpy as np
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import Slider
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from joblib import dump
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import warnings

warnings.filterwarnings("ignore")


class PersonAgent(mesa.Agent):
    def __init__(
        self,
        unique_id,
        model,
        group,
        initial_wealth,
        opportunities,
        sex,
        age_of_death,
        taxes_rate,
    ):
        super().__init__(unique_id, model)
        self.model = model
        self.group = group
        self.wealth = initial_wealth
        self.opportunities = opportunities
        self.career_years = 0
        self.wealth_growth_rate = (
            self.model.group_a_wealth_rate
            if group == "A"
            else self.model.group_b_wealth_rate
        )
        self.job = True
        self.sex = sex
        self.age = 0
        self.age_of_death = age_of_death
        self.diseases = 0
        self.diesease_probability = 0.01 if self.group == "A" else 0.05
        self.has_car = False
        self.has_house = False
        self.job_loss_probability = 0.05 if self.group == "A" else 0.15
        self.reproduction_chance = 0.03
        self.child_possibility = 1
        self.taxes_rate = taxes_rate

    def step(self):
        # Age the agent each step
        self.age += 1

        if self.wealth > (0.5 * self.model.average_wealth()):
            self.age_of_death += 0.2

        # simulates the agent getting diseases which will reduce its lifetime
        if np.random.uniform(0, 1) < self.diesease_probability:
            self.diseases += 1

        if self.diseases > 0:
            self.age_of_death -= 0.2 * self.diseases

        # Check if agent has reached age of death
        if self.age >= self.age_of_death:
            self.model.schedule.remove(self)
            self.model.grid.remove_agent(self)
            return  # End their life cycle if they reach the age of death

        if self.job and self.age >= 18:
            self.career_years += 1

            # Adjust wealth growth rate based on career years
            if self.career_years == int((self.model.max_steps * (2 / 10))):
                self.wealth_growth_rate *= 1.04
            elif self.career_years == int((self.model.max_steps * (5 / 10))):
                self.wealth_growth_rate *= 1.08
            elif self.career_years == int((self.model.max_steps * (8 / 10))):
                self.wealth_growth_rate *= 1.2

            # simulate losing the job
            if np.random.uniform(0, 1) < self.job_loss_probability:
                self.job = False
                return

            # Wealth accumulation based on opportunities and gender-based income inequality
            if self.opportunities:
                if self.sex == "F":
                    self.wealth += np.random.uniform(
                        0, self.wealth_growth_rate * 1.5 * (2 / 3)
                    )  # Women get 2/3 of the wealth
                else:
                    self.wealth += np.random.uniform(0, self.wealth_growth_rate * 1.5)
            else:
                if self.sex == "F":
                    self.wealth += np.random.uniform(
                        0, self.wealth_growth_rate * (2 / 3)
                    )
                else:
                    self.wealth += np.random.uniform(0, self.wealth_growth_rate)

        if self.wealth < 0:
            self.wealth = 0

        # simulate paying taxes
        if self.age >= 18:
            self.wealth -= self.taxes_rate * self.wealth

        # simulate buying car or house
        if self.wealth > (0.7 * self.model.average_wealth()) and not self.has_car:
            self.has_car = True
            self.wealth *= 0.7
            self.reproduction_chance *= 3
            self.job_loss_probability /= 2

        if self.wealth > (0.9 * self.model.average_wealth()) and not self.has_house:
            self.has_house = True
            self.wealth *= 0.3
            self.child_possibility *= 3
            self.job_loss_probability /= 4

        # Wealth transfer if interacting with another agent
        other_agent = self.random.choice(self.model.schedule.agents)
        if self.wealth > other_agent.wealth:
            wealth_transfer = np.random.uniform(0, 0.15)
            self.wealth += wealth_transfer
            other_agent.wealth -= wealth_transfer

        # Move to a new position
        self.move()

        # Check if touching another agent for reproduction
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        for other in cellmates:
            if (
                other is not self
                and self.group == other.group
                and self.sex != other.sex
                and np.random.uniform(0, 1) < self.reproduction_chance
                and self.age >= 18
                and other.age >= 18
            ):
                age_of_death = (
                    self.model.age_of_death_a
                    if self.group == "A"
                    else self.model.age_of_death_b
                )
                # number of children it can have
                for _ in range(1, self.child_possibility):
                    self.model.create_agent(
                        self.group,
                        np.random.uniform(1, 10),
                        np.random.choice([True, False]),
                        np.random.choice(["M", "F"]),
                        age_of_death,
                        self.taxes_rate,
                    )

    def move(self):
        possible_moves = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_position = self.random.choice(possible_moves)
        self.model.grid.move_agent(self, new_position)


class SocietyModel(mesa.Model):
    def __init__(
        self,
        num_agents_a,
        num_agents_b,
        group_a_wealth_rate,
        group_b_wealth_rate,
        max_steps,
        age_of_death_a,
        age_of_death_b,
        taxes_rate,
    ):
        super().__init__()
        # Initialize model variables from sliders
        self.num_agents_a = num_agents_a
        self.num_agents_b = num_agents_b
        self.group_a_wealth_rate = group_a_wealth_rate
        self.group_b_wealth_rate = group_b_wealth_rate
        self.grid = mesa.space.MultiGrid(30, 30, False)
        self.schedule = mesa.time.RandomActivation(self)
        self.next_id = 0
        self.max_steps = max_steps
        self.current_step = 0
        self.age_of_death_a = age_of_death_a
        self.age_of_death_b = age_of_death_b
        self.taxes_rate = taxes_rate
        self.create_agents()

        # Collect data
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Average Wealth": self.average_wealth,
                "Group A Average Wealth": lambda m: m.group_average_wealth("A"),
                "Group B Average Wealth": lambda m: m.group_average_wealth("B"),
                "Group A Wealth Rate": "group_a_wealth_rate",
                "Group B Wealth Rate": "group_b_wealth_rate",
                "Taxes Rate": "taxes_rate",
            },
            agent_reporters={
                "Wealth": "wealth",
                "Group": "group",
                "Opportunities": "opportunities",
                "Career Years": "career_years",
                "Sex": "sex",
                "Job": "job",
                "Age": "age",
                "Diseases": "diseases",
                "Has Car": "has_car",
                "Has House": "has_house",
                "Job Loss Probability": "job_loss_probability",
                "Reproduction Chance": "reproduction_chance",
                "Child Possibility": "child_possibility",
            },
        )

    def create_agents(self):
        for _ in range(self.num_agents_a):
            initial_wealth = np.random.uniform(5, 10)
            opportunities = np.random.choice([True, False], p=[0.8, 0.2])
            sex = np.random.choice(["M", "F"])
            self.create_agent(
                "A",
                initial_wealth,
                opportunities,
                sex,
                self.age_of_death_a,
                self.taxes_rate,
            )

        for _ in range(self.num_agents_b):
            initial_wealth = np.random.uniform(1, 5)
            opportunities = np.random.choice([True, False], p=[0.3, 0.7])
            sex = np.random.choice(["M", "F"], p=[0.6, 0.4])
            self.create_agent(
                "B",
                initial_wealth,
                opportunities,
                sex,
                self.age_of_death_b,
                self.taxes_rate,
            )

    def create_agent(
        self, group, initial_wealth, opportunities, sex, age_of_death, taxes_rate
    ):
        agent = PersonAgent(
            self.next_id,
            self,
            group,
            initial_wealth,
            opportunities,
            sex,
            age_of_death,
            taxes_rate,
        )
        self.next_id += 1
        self.schedule.add(agent)

        empty_cells = list(self.grid.empties)
        if empty_cells:
            random_position = self.random.choice(empty_cells)
            self.grid.place_agent(agent, random_position)

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.current_step += 1

        if self.current_step == self.max_steps:
            self.running = False
            self.train_model_on_collected_data()

        # represent advancements in society by increasing the wealth rate of discriminated class (B)
        if self.current_step == int(self.max_steps * (3 / 5)):
            self.group_b_wealth_rate = self.group_a_wealth_rate * 0.8

    def average_wealth(self):
        if not self.schedule.agents:
            return 0
        return sum(agent.wealth for agent in self.schedule.agents) / len(
            self.schedule.agents
        )

    def group_average_wealth(self, group):
        group_agents = [agent for agent in self.schedule.agents if agent.group == group]
        if not group_agents:
            return 0
        return sum(agent.wealth for agent in group_agents) / len(group_agents)

    def train_model_on_collected_data(self):
        agent_data = self.datacollector.get_agent_vars_dataframe().reset_index()
        agent_data["Group"] = agent_data["Group"].apply(lambda x: 1 if x == "A" else 0)

        model_data = self.datacollector.get_model_vars_dataframe()

        merged_data = agent_data.merge(
            model_data, left_on="Step", right_index=True, how="left"
        )
        merged_data["Sex"] = merged_data["Sex"].map({"M": 1, "F": 0})
        merged_data.to_csv("merged_data.csv")

        X = merged_data[
            [
                "Wealth",
                # "Opportunities",
                "Career Years",
                "Sex",
                "Job",
                "Diseases",
                "Has Car",
                "Has House",
                # "Job Loss Probability",
                # "Reproduction Chance",
                "Child Possibility",
            ]
        ]
        y = merged_data["Group"]

        models = {
            "RandomForest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False),
            "LightGBM": LGBMClassifier(random_state=42),
        }

        param_grids = {
            "RandomForest": {
                "n_estimators": [100, 200, 500],
                "max_depth": [3, 6, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "XGBoost": {
                "n_estimators": [100, 200, 500],
                "learning_rate": [0.01, 0.1, 0.3],
                "max_depth": [3, 6, 10, None],
                "gamma": [0, 0.1, 0.3],
            },
            "LightGBM": {
                "n_estimators": [100, 200, 500],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 6, 10, None],
                "num_leaves": [31, 50, 100],
            },
        }

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # compute Equal Opportunity Difference and Disparate Misclassification Rate
        def compute_eod(y_true, y_pred, group):
            # True Positive Rate (sensitivity) for each group
            tpr_group1 = recall_score(y_true[group == 0], y_pred[group == 0])
            tpr_group2 = recall_score(y_true[group == 1], y_pred[group == 1])
            return tpr_group2 - tpr_group1  # EOD is the difference in TPRs

        def compute_dmr(y_true, y_pred, group):
            # Misclassification rate (False Negative Rate + False Positive Rate) for each group
            fnr_group1 = 1 - recall_score(y_true[group == 0], y_pred[group == 0])
            fnr_group2 = 1 - recall_score(y_true[group == 1], y_pred[group == 1])
            fpr_group1 = 1 - precision_score(y_true[group == 0], y_pred[group == 0])
            fpr_group2 = 1 - precision_score(y_true[group == 1], y_pred[group == 1])
            return (fnr_group2 + fpr_group2) - (
                fnr_group1 + fpr_group1
            )  # DMR is the difference in FN + FP rates

        metrics = {
            "Model": [],
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "F1 Score": [],
            "ROC AUC": [],
            "EOD": [],
            "DMR": [],
        }

        classification_reports = {}

        for model_name, model in models.items():
            random_search = RandomizedSearchCV(
                model,
                param_distributions=param_grids[model_name],
                random_state=42,
                n_jobs=-1,
            )
            random_search.fit(X_train, y_train)

            best_model = random_search.best_estimator_

            os.makedirs("models", exist_ok=True)
            dump(best_model, f"models/{model_name}.pkl")

            # Predictions
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]

            # Confusion Matrix
            confusion_mat = confusion_matrix(y_test, y_pred)
            print(f"{model_name} Confusion Matrix:\n{confusion_mat}")

            # Classification Report
            classification_reports[model_name] = classification_report(
                y_test, y_pred, output_dict=True
            )
            print(
                f"{model_name} Classification Report:\n{classification_report(y_test, y_pred, output_dict=False)}"
            )

            # Performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            print(
                f"Performance Metrics for {model_name}:\n\tAccuracy: {accuracy:.4f}\n\tPrecision: {precision:.4f}\n\tRecall: {recall:.4f}\n\tF1: {f1:.4f}\n\tROC AUC: {roc_auc:.4f}"
            )

            # Fairness metrics
            eod = compute_eod(y_test, y_pred, y_test)
            dmr = compute_dmr(y_test, y_pred, y_test)

            print(
                f"Fairness Metrics:\n\tEqual Opportunity Difference: {eod:.3f}.\n\tMisclassification Rate: {dmr:.3f}"
            )

            # Append metrics for comparison
            metrics["Model"].append(model_name)
            metrics["Accuracy"].append(accuracy)
            metrics["Precision"].append(precision)
            metrics["Recall"].append(recall)
            metrics["F1 Score"].append(f1)
            metrics["ROC AUC"].append(roc_auc)
            metrics["EOD"].append(eod)
            metrics["DMR"].append(dmr)

            results = {}
            for i, m in enumerate(metrics["Model"]):
                results[m] = {
                    "accuracy": metrics["Accuracy"][i],
                    "precision": metrics["Precision"][i],
                    "recall": metrics["Recall"][i],
                    "f1": metrics["F1 Score"][i],
                    "roc_auc": metrics["ROC AUC"][i],
                    "eod": metrics["EOD"][i],
                    "dmr": metrics["DMR"][i],
                }

        # save results
        os.makedirs("results", exist_ok=True)
        with open("results/metrics.json", "w") as f:
            json.dump(results, f, indent=4)

        with open(f"results/classification_reports.json", "w") as f:
            json.dump(classification_reports, f, indent=4)

        # Plot the metrics for comparison
        # plt.figure(figsize=(10, 6))
        # for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]:
        #     sns.barplot(x=metrics["Model"], y=metrics[metric])
        #     plt.title(f"Model Comparison: {metric}")
        #     plt.show()


# Visualization components
def agent_portrayal(agent):
    portrayal = {
        "Layer": 0,
        "Color": "blue" if agent.group == "A" else "red",
        "scale": 1.5 + agent.wealth * 0.01,
    }

    if agent.sex == "F":
        portrayal["Shape"] = (
            "icons/woman_A.png" if agent.group == "A" else "icons/woman_B.png"
        )
    elif agent.sex == "M":
        portrayal["Shape"] = (
            "icons/man_A.png" if agent.group == "A" else "icons/man_B.png"
        )

    return portrayal


canvas_element = CanvasGrid(agent_portrayal, 30, 30, 600, 600)
chart_element = ChartModule(
    [
        {"Label": "Average Wealth", "Color": "black"},
        {"Label": "Group A Average Wealth", "Color": "blue"},
        {"Label": "Group B Average Wealth", "Color": "red"},
    ],
    2,
    6,
)

model_params = {
    "num_agents_a": Slider("Number of Group A Agents", 200, 100, 500),
    "num_agents_b": Slider("Number of Group B Agents", 200, 100, 500),
    "age_of_death_a": Slider("Age of Group A Death", 90, 50, 100, 5),
    "age_of_death_b": Slider("Age of Group B Death", 80, 50, 100, 5),
    "group_a_wealth_rate": Slider("Group A Wealth Growth Rate", 0.6, 0.01, 1.0, 0.01),
    "group_b_wealth_rate": Slider("Group B Wealth Growth Rate", 0.2, 0.01, 1.0, 0.01),
    "taxes_rate": Slider("Taxes Rate", 0.15, 0.1, 0.5, 0.05),
    "max_steps": Slider("Maximum Steps", 100, 10, 500),
}

server = ModularServer(
    SocietyModel,
    [canvas_element, chart_element],
    "Wealth Distribution Model",
    model_params,
)
server.port = 8521
server.launch()
