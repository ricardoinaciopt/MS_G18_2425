import mesa
import numpy as np
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import Slider
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.metrics import ConfusionMatrixDisplay


class PersonAgent(mesa.Agent):
    def __init__(self, unique_id, model, group, initial_wealth, opportunities):
        super().__init__(unique_id, model)
        self.group = group
        self.wealth = initial_wealth
        self.opportunities = opportunities
        self.career_years = 0
        self.wealth_growth_rate = (
            self.model.group_a_wealth_rate
            if group == "A"
            else self.model.group_b_wealth_rate
        )

    def step(self):
        if self.opportunities:
            self.wealth += np.random.uniform(0, self.wealth_growth_rate * 1.5)
        else:
            self.wealth += np.random.uniform(0, self.wealth_growth_rate)

        other_agent = self.random.choice(self.model.schedule.agents)
        if self.wealth > other_agent.wealth:
            wealth_transfer = np.random.uniform(0, 0.15)
            self.wealth += wealth_transfer
            other_agent.wealth -= wealth_transfer


class SocietyModel(mesa.Model):
    def __init__(
        self,
        num_agents_a,
        num_agents_b,
        group_a_wealth_rate,
        group_b_wealth_rate,
        max_steps,
    ):
        super().__init__()
        self.num_agents_a = num_agents_a
        self.num_agents_b = num_agents_b
        self.group_a_wealth_rate = group_a_wealth_rate
        self.group_b_wealth_rate = group_b_wealth_rate
        self.grid = mesa.space.MultiGrid(30, 30, False)
        self.schedule = mesa.time.RandomActivation(self)
        self.next_id = 0
        self.max_steps = max_steps
        self.current_step = 0
        self.create_agents()

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Average Wealth": self.average_wealth,
                "Group A Average Wealth": lambda m: m.group_average_wealth("A"),
                "Group B Average Wealth": lambda m: m.group_average_wealth("B"),
            },
            agent_reporters={
                "Wealth": "wealth",
                "Group": "group",
                "Opportunities": "opportunities",
            },
        )

    def create_agents(self):
        for _ in range(self.num_agents_a):
            initial_wealth = np.random.uniform(5, 10)
            opportunities = np.random.choice([True, False], p=[0.8, 0.2])
            self.create_agent("A", initial_wealth, opportunities)

        for _ in range(self.num_agents_b):
            initial_wealth = np.random.uniform(1, 5)
            opportunities = np.random.choice([True, False], p=[0.3, 0.7])
            self.create_agent("B", initial_wealth, opportunities)

    def create_agent(self, group, initial_wealth, opportunities):
        agent = PersonAgent(self.next_id, self, group, initial_wealth, opportunities)
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
            self.train_model_on_collected_data()

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
        merged_data.to_csv("merged_data.csv")

        X = merged_data[
            [
                "Wealth",
                # "Average Wealth",
                # "Group A Average Wealth",
                # "Group B Average Wealth",
            ]
        ]
        y = merged_data["Group"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        dump(model, "model.joblib")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of being Class 1

        # Confusion Matrix
        confusion_mat = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{confusion_mat}")
        print(classification_report(y_test, y_pred))

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

        # Visualization of Confusion Matrix
        ConfusionMatrixDisplay(confusion_matrix=confusion_mat).plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.show()

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(
            fpr, tpr, color="blue", lw=2, label="ROC Curve (area = %0.2f)" % roc_auc
        )
        plt.plot([0, 1], [0, 1], color="red", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()


# Visualization components
def agent_portrayal(agent):
    return {
        "Shape": "circle",
        "Filled": "true",
        "r": 0.5 + agent.wealth / 20,
        "Layer": 0,
        "Color": "blue" if agent.group == "A" else "red",
    }


grid = CanvasGrid(agent_portrayal, 30, 30, 500, 500)
chart = ChartModule(
    [
        {"Label": "Average Wealth", "Color": "Black"},
        {"Label": "Group A Average Wealth", "Color": "Blue"},
        {"Label": "Group B Average Wealth", "Color": "Red"},
    ],
    data_collector_name="datacollector",
)

model_params = {
    "num_agents_a": Slider("Number of Group A Agents", 50, 1, 100, 1),
    "num_agents_b": Slider("Number of Group B Agents", 50, 1, 100, 1),
    "group_a_wealth_rate": Slider("Group A Wealth Rate", 0.2, 0.01, 0.5, 0.01),
    "group_b_wealth_rate": Slider("Group B Wealth Rate", 0.1, 0.01, 0.5, 0.01),
    "max_steps": Slider("Max Steps", 100, 10, 500, 10),
}

server = ModularServer(
    SocietyModel,
    [grid, chart],
    "Interactive Society Model with Biased Wealth Accumulation",
    model_params,
)

server.port = 8521
server.launch()
