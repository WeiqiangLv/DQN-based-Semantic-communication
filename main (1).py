import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, Counter
import random
from scipy.stats import norm
from scipy.optimize import minimize

# -------------------------------
# Reproducibility
# -------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -------------------------------
# Utils
# -------------------------------
def db2lin(x_db: float) -> float:
    return 10.0 ** (x_db / 10.0)

def lin2db(x_lin: float) -> float:
    x_lin = max(x_lin, 1e-12)
    return 10.0 * np.log10(x_lin)

# -------------------------------
# Deep Q-Network
# -------------------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------------
# DQN Agent
# -------------------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        # decay once per EPISODE (slower, clearer learning curve)
        self.epsilon_decay_episode = 0.96
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # start target as a copy of online
        self.update_target_model()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # epsilon-greedy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return int(q_values.argmax(dim=1).item())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        mse = nn.MSELoss()

        for state, action, reward, next_state, done in minibatch:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            next_t  = torch.as_tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Current Q(s, ·)
            q = self.model(state_t)

            # Target Q value for the taken action
            with torch.no_grad():
                next_q_max = self.target_model(next_t).max(dim=1)[0].item()
                target_val = reward if done else reward + self.gamma * next_q_max

            # Build a detached target vector
            q_target = q.clone().detach()
            q_target[0, action] = target_val

            # Forward again to get fresh graph for loss
            q_pred = self.model(state_t)
            loss = mse(q_pred, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def decay_epsilon_episode(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_episode)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# -------------------------------
# Reward model pieces
# -------------------------------
def reward_function(T, BER, SA, OH, alpha1, alpha2, alpha3, alpha4, beta1, beta2, gamma):
    # Increasing reward with T, (1-BER), and SA^gamma; quadratic penalty on OH
    return (alpha1 * np.log(1 + beta1 * T) +
            alpha2 * (1 - np.exp(-beta2 * (1 - BER))) +
            alpha3 * (SA ** gamma) -
            alpha4 * (OH ** 2))

def throughput_constraint(T, SNR):
    # T <= log2(1+SNR)  ->  log2(1+SNR) - T >= 0
    return np.log2(1 + SNR) - T

def ber_bound(M, SNR):
    # Per-bit SNR: SNR_b = SNR / log2(M)
    # BER ≈ Q(sqrt(2*SNR_b))
    # BPSK (M=2) matches; QPSK (M=4) gives Q(sqrt(SNR))
    log2M = np.log2(M)
    SNR_b = SNR / log2M
    return norm.sf(np.sqrt(2.0 * SNR_b))

def semantic_min(T, lambda_param):
    # Minimal achievable SA given T (toy linkage)
    return 1.0 - np.exp(-lambda_param * T)

# -------------------------------
# Action mapping: (modulation, CP)
# -------------------------------
MODS = [("BPSK", 2), ("QPSK", 4)]
CP_OPTIONS = [1/32, 1/16, 1/8, 1/4]  # 0.03125..0.25
ACTION_SIZE = len(MODS) * len(CP_OPTIONS)  # 8

def decode_action(action):
    mod_idx = action // len(CP_OPTIONS)
    cp_idx = action % len(CP_OPTIONS)
    mod_label, M = MODS[mod_idx]
    cp = CP_OPTIONS[cp_idx]
    return mod_label, M, cp

# -------------------------------
# Optimization problem (reward oracle)
# -------------------------------
def optimize_params(state, action):
    # Use effective SNR (state[1]) and semantic importance (state[2])
    SNR, semantic_importance = float(state[1]), float(state[2])

    mod_label, M, cp = decode_action(action)

    # Adaptive parameters
    alpha3 = k1 * semantic_importance
    gamma_exp = 0.5 + 1.5 * semantic_importance

    # Objective (negative for minimizer)
    def objective(x):
        T, BER, SA, OH = x
        return -reward_function(T, BER, SA, OH,
                                alpha1, alpha2, alpha3, alpha4,
                                beta1, beta2, gamma_exp)

    # Modulation-specific BER cap
    ber_cap = ber_bound(M, SNR)

    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda x: throughput_constraint(x[0], SNR)},
        {'type': 'ineq', 'fun': lambda x: ber_cap - x[1]},  # BER <= cap
        {'type': 'ineq', 'fun': lambda x: x[2] - semantic_min(x[0], lambda_param)},  # SA >= min
        {'type': 'eq',   'fun': lambda x: x[3] - cp},  # OH == CP (bind action)
    ]

    # Bounds (OH fixed to cp)
    bounds = [(0, 1), (0, 1), (0, 1), (cp, cp)]

    # Feasible initial guess
    T0 = float(min(np.log2(1 + SNR), 0.95))
    BER0 = float(max(0.0, min(0.5, ber_cap * 0.9)))
    SA0 = float(semantic_min(T0, lambda_param))
    x0 = [T0, BER0, SA0, cp]

    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        # Fallback: reward at feasible x0
        return -objective(x0)

    return -result.fun  # maximize reward

# -------------------------------
# AWGN/Fading Environment
# -------------------------------
class ToyEnv:
    # A tiny environment with slowly-varying SNR (in dB) and delay-spread (tau).
    # If CP < tau, the effective SNR is penalized (ISI). Semantic importance drifts slowly.
    # State = [t_norm, SNR_lin_eff, semantic_importance, tau, jammer, reserved]
    def __init__(self, max_steps=500,
                 mu_snr_db=12.0, rho_snr=0.98, sigma_snr_db=1.0,
                 tau_rho=0.95, tau_sigma=0.01,
                 isi_penalty_db_per_frac=40.0):
        self.max_steps = max_steps
        self.mu_snr_db = mu_snr_db
        self.rho_snr = rho_snr
        self.sigma_snr_db = sigma_snr_db
        self.tau_rho = tau_rho
        self.tau_sigma = tau_sigma
        self.isi_penalty_db_per_frac = isi_penalty_db_per_frac
        self.reset()

    def reset(self):
        self.step_idx = 0
        self.snr_db = np.random.uniform(8.0, 18.0)  # initial SNR in dB
        self.tau = np.random.uniform(1/32, 1/4)     # needed CP fraction
        self.semantic_importance = np.random.rand()
        return self._make_state(self.snr_db, self.tau, self.semantic_importance)

    def _make_state(self, snr_db_eff, tau, sem_imp):
        t_norm = self.step_idx / self.max_steps
        snr_lin_eff = db2lin(snr_db_eff)
        jammer = 0.0  # placeholder
        reserved = 0.0
        return np.array([t_norm, snr_lin_eff, sem_imp, tau, jammer, reserved], dtype=np.float32)

    def step(self, action):
        mod_label, M, cp = decode_action(action)

        # ISI penalty if CP < tau
        deficit = max(0.0, self.tau - cp)
        penalty_db = self.isi_penalty_db_per_frac * deficit  # e.g., 0.125 deficit -> ~5 dB
        snr_eff_db = self.snr_db - penalty_db

        # Reward via optimizer uses effective SNR
        state_for_reward = self._make_state(snr_eff_db, self.tau, self.semantic_importance)
        reward = optimize_params(state_for_reward, action)

        # Evolve channel SNR (AR(1) toward mu) with small noise; keep effect of penalty persistent a bit
        noise_db = np.random.normal(0.0, self.sigma_snr_db)
        self.snr_db = (self.rho_snr * self.snr_db +
                       (1.0 - self.rho_snr) * self.mu_snr_db +
                       noise_db - 0.3 * penalty_db)

        # Evolve tau (needed CP) slowly
        self.tau = np.clip(self.tau_rho * self.tau +
                           (1.0 - self.tau_rho) * np.random.uniform(1/32, 1/4) +
                           np.random.normal(0.0, self.tau_sigma),
                           1/32, 1/4)

        # Evolve semantic importance slowly
        self.semantic_importance = float(np.clip(
            0.98 * self.semantic_importance + 0.02 * np.random.rand() + np.random.normal(0.0, 0.02),
            0.0, 1.0))

        self.step_idx += 1
        done = (self.step_idx >= self.max_steps)
        next_state = self._make_state(self.snr_db, self.tau, self.semantic_importance)
        info = {"snr_db_eff": snr_eff_db, "penalty_db": penalty_db, "cp": cp, "tau": self.tau, "mod": mod_label}
        return next_state, reward, done, info

# -------------------------------
# Main training loop
# -------------------------------
def train_dqn():
    state_size = 6
    action_size = ACTION_SIZE
    agent = DQNAgent(state_size, action_size)
    batch_size = 32

    # episodes = 1000  # original
    episodes = 25      # short demo run

    env = ToyEnv(max_steps=500)
    for e in range(episodes):
        state = env.reset()
        total_reward = 0.0
        action_hist = Counter()

        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.replay(batch_size)

            state = next_state
            total_reward += reward
            action_hist[action] += 1

            if done:
                agent.decay_epsilon_episode()
                print("Episode: {}/{}, Total Reward: {:.4f}, Epsilon: {:.3f}, Actions: {}".format(
                    e+1, episodes, total_reward, agent.epsilon, dict(action_hist)))
                break

        if (e + 1) % 10 == 0:
            agent.update_target_model()

# -------------------------------
# Global parameters
# -------------------------------
alpha1, alpha2, alpha4 = 1, 1, 1
beta1, beta2 = 1, 1
k1 = 1
lambda_param = 1

if __name__ == "__main__":
    train_dqn()
