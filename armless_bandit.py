import numpy as np
import pandas as pd

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

class GradientBanditsAgent:
    def __init__(self, n_actions, alpha):
        self.n_actions = n_actions
        self.alpha = alpha
        self.H = np.zeros(n_actions)
        self.reward_avg = 0
        self.t = 0

    def select_action(self):
        probs = softmax(self.H)
        return np.random.choice(self.n_actions, p=probs)

    def update(self, action, reward):
        self.t += 1
        self.reward_avg += (reward - self.reward_avg) / self.t
        probs = softmax(self.H)
        self.H -= self.alpha * (reward - self.reward_avg) * probs
        self.H[action] += self.alpha * (reward - self.reward_avg)



df = pd.read_csv("diabetes_prediction_dataset.csv")

def arm_always_1(row): return 1
def arm_always_0(row): return 0
def arm_age_over_50(row): return 1 if row["age"] > 50 else 0
def arm_bmi_over_30(row): return 1 if row["bmi"] > 30 else 0

arms = [arm_always_1, arm_always_0, arm_age_over_50, arm_bmi_over_30]
n_arms = len(arms)

agent = GradientBanditsAgent(n_actions=n_arms, alpha=0.1)

correct = 0
selected_arms = []

for _, row in df.iterrows():
    arm = agent.select_action()
    pred = arms[arm](row)
    reward = 1 if pred == row["diabetes"] else 0
    agent.update(arm, reward)
    correct += reward
    selected_arms.append(arm)

accuracy = correct / len(df)

print("Точність :", accuracy)
