import pandas as pd
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

class GradientBanditsAgent:
    def __init__(self, n_actions, alpha=0.1):
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



df = pd.read_csv("cybersecurity_intrusion_data.csv")

def rule_high_reputation(row):   
    return 1 if row['ip_reputation_score'] > 0.4 else 0

def rule_failed_logins(row):      
    return 1 if row['failed_logins'] >= 2 else 0

arms = [rule_high_reputation, rule_failed_logins]


def ab_test(df):
    predictions = []
    for _, row in df.iterrows():
        pred_A = rule_high_reputation(row)
        pred_B = rule_failed_logins(row)
    final_pred = df.apply(rule_high_reputation, axis=1)
    accuracy = np.mean(final_pred == df["attack_detected"])
    return accuracy


agent = GradientBanditsAgent(n_actions=len(arms), alpha=0.1)
correct = 0

for _, row in df.iterrows():
    arm = agent.select_action()
    pred = arms[arm](row)
    reward = 1 if pred == row["attack_detected"] else 0
    agent.update(arm, reward)
    correct += reward

bandit_accuracy = correct / len(df)
ab_accuracy = ab_test(df)

print("Точність A/B тестування:", ab_accuracy)
print("Точність Gradient Bandit:", bandit_accuracy)
print("Ймовірності вибору стратегій:", softmax(agent.H))
