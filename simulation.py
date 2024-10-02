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
    def __init__(
        self, unique_id, model, group, initial_wealth, opportunities, sex, age_of_death
    ):
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
        self.job = True
        self.sex = sex
        self.age = 0
        self.age_of_death = age_of_death
        self.diseases = 0
        self.diesease_probability = 0.01 if self.group == "A" else 0.1
        self.has_car = False
        self.has_house = False
        self.job_loss_probability = 0.05 if self.group == "A" else 0.15
        self.reproduction_chance = 0.05
        self.child_possibility = 1

    def step(self):
        # Age the agent each step
        self.age += 1

        if self.wealth > (0.5 * self.model.average_wealth()):
            self.age_of_death += 0.05

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

        if self.job:
            self.career_years += 1

            # Adjust wealth growth rate based on career years
            if self.career_years > 5:
                self.wealth_growth_rate += 0.01
            elif self.career_years > 10:
                self.wealth_growth_rate += 0.03
            elif self.career_years > 20:
                self.wealth_growth_rate += 0.06

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

        # simulate buying car or house
        if self.wealth > (0.6 * self.model.average_wealth()) and not self.has_car:
            self.has_car = True
            self.wealth *= 0.7
            self.reproduction_chance *= 2
            self.job_loss_probability /= 2

        if self.wealth > (0.8 * self.model.average_wealth()) and not self.has_house:
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
            ):
                age_of_death = 90 if self.group == "A" else 70
                # number of children it can have
                for _ in range(1, self.child_possibility):
                    self.model.create_agent(
                        self.group,
                        np.random.uniform(1, 10),
                        np.random.choice([True, False]),
                        np.random.choice(["M", "F"]),
                        age_of_death,
                    )

    def move(self):
        possible_moves = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_position = self.random.choice(possible_moves)
        self.model.grid.move_agent(self, new_position)

#fazer função auxiliar q,a  meio dos max steps, iguala o wealth de todos os agentes,e chamala no step do model da sociedade
class SocietyModel(mesa.Model):
    def __init__(
        self,
        num_agents_a,
        num_agents_b,
        group_a_wealth_rate,
        group_b_wealth_rate,
        max_steps,
        age_of_death_a=0.9,
        age_of_death_b=0.8,
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
        self.age_of_death_a = age_of_death_a * max_steps
        self.age_of_death_b = age_of_death_b * max_steps
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

    def change_wealth_rate_group(self, group, new_rate):
        agents_group = [agent for agent in self.schedule.agents if agent.group == group]

        for agent in agents_group:
            agent.wealth_growth_rate = new_rate

    def equalize_wealth_rate(self, new_rate):
        self.change_wealth_rate_group("A", new_rate)
        self.change_wealth_rate_group("B", new_rate)

    def create_agents(self):
        for _ in range(self.num_agents_a):
            initial_wealth = np.random.uniform(5, 10)
            opportunities = np.random.choice([True, False], p=[0.8, 0.2])
            sex = np.random.choice(["M", "F"])
            self.create_agent(
                "A", initial_wealth, opportunities, sex, self.age_of_death_a
            )

        for _ in range(self.num_agents_b):
            initial_wealth = np.random.uniform(1, 5)
            opportunities = np.random.choice([True, False], p=[0.3, 0.7])
            sex = np.random.choice(["M", "F"])
            self.create_agent(
                "B", initial_wealth, opportunities, sex, self.age_of_death_b
            )

    def create_agent(self, group, initial_wealth, opportunities, sex, age_of_death):
        agent = PersonAgent(
            self.next_id, self, group, initial_wealth, opportunities, sex, age_of_death
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

        if self.current_step == self.max_steps / 2:
            self.equalize_wealth_rate(0.25)

        wealth_rate_sum = 0
        for agent in self.schedule.agents:
            wealth_rate_sum += agent.wealth_growth_rate

        print(wealth_rate_sum/len(self.schedule.agents))

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
                # "Job",
                "Diseases",
                "Has Car",
                "Has House",
                # "Job Loss Probability",
                "Reproduction Chance",
                # "Child Possibility",
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

        plot = False
        if plot:
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
    portrayal = {
        "Layer": 0,
        "Color": "blue" if agent.group == "A" else "red",
        "scale": 1.5 + agent.wealth / 10,
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
    "num_agents_a": Slider("Number of Group A Agents", 80, 1, 100, 1),
    "num_agents_b": Slider("Number of Group B Agents", 60, 1, 100, 1),
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
