Semantic-Aware OFDM Demo DQN
This small demo accompanies the idea in the paper abstract below. It trains a Deep Q-Network (DQN) to choose (modulation × cyclic-prefix) actions for a synthetic OFDM link while a constrained nonlinear optimizer computes a per-step reward over Throughput (T), BER, Semantic Accuracy (SA), and Overhead (OH).

What this demo does
Action space (8 actions): {BPSK, QPSK} × {CP ∈ {1/32, 1/16, 1/8, 1/4}}.
CP is enforced as equality in the optimizer so the action truly binds OH.
State: 6-D random vector in [0,1); only SNR = state[1] and semantic_importance = state[2] are used.
Reward:
\( r = \alpha_1 \log(1+\beta_1 T) + \alpha_2 (1-e^{-\beta_2 (1-\text{BER})}) + \alpha_3 \text{SA}^{\gamma} - \alpha_4 \text{OH}^2 \).
Constraints: T ≤ log2(1+SNR), BER ≤ BER_cap(M,SNR), SA ≥ 1-exp(-λT), and OH == CP.
Modulation-aware BER cap: BPSK/QPSK via per-bit SNR \( \text{SNR}_b = \text{SNR}/\log_2 M \), giving
BER_cap = Q(√(2·SNR_b)). (So BPSK uses Q(√(2·SNR)); QPSK uses Q(√(SNR)).)
Training: DQN with target network and experience replay.
Episodes default to 25 (original 1000 is commented in the code for quick runs).
⚠️ Note: This is a small environment: states and transitions are random. It’s meant as a minimal, inspectable skeleton , not a full physical simulator.

Quickstart (Install & Run Together)
Linux / macOS (CPU)
python -m venv .venv && source .venv/bin/activate \
&& python -m pip install --upgrade pip \
&& pip install numpy scipy torch \
&& python main.py
Windows (PowerShell, CPU)
python -m venv .venv; .\.venv\Scripts\Activate.ps1;
python -m pip install --upgrade pip;
pip install numpy scipy torch;
python .\main.py
If you need a specific PyTorch build (CUDA/CPU), follow the official selector at pytorch.org and replace the pip install torch line accordingly.

Files
main.py — single-file demo (DQN + SLSQP; episodes=25; action=(mod,CP))
(optional) requirements.txt (you can create it):
numpy
scipy
torch
Configuration knobs (inside main.py)
Episodes:
# episodes = 1000  # original
episodes = 25      # short demo
Action map (top of file):
MODS = [("BPSK", 2), ("QPSK", 4)]
CP_OPTIONS = [1/32, 1/16, 1/8, 1/4]
