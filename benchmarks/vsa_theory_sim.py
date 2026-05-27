import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys

# Ensure astrum_verum is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from astrum_verum.vsa import core

def run_simulation(N=10000, D=10000, num_facts=1000):
    print(f"Running VSA Simulation with D={D}, Vocabulary Size N={N}")
    rng = np.random.default_rng(42)
    
    # 1. Create a vocabulary of N random bipolar vectors
    print("Generating vocabulary...")
    vocab = core.random_atoms(N, D, rng)
    
    # 2. Simulate noise distribution (random vectors dot product with vocab)
    print("Simulating noise...")
    random_unbound = core.random_atoms(1000, D, rng)
    # Cosine similarity is just (a @ b) / D for bipolar vectors
    noise_cosines = (random_unbound @ vocab.T) / D
    noise_flat = noise_cosines.flatten()
    
    # 3. Simulate true facts (S + R + O bundled)
    print(f"Simulating {num_facts} facts...")
    signal_cosines = []
    
    # Pre-select roles (just random vectors)
    role_s = core.random_atoms(1, D, rng)[0]
    role_r = core.random_atoms(1, D, rng)[0]
    role_o = core.random_atoms(1, D, rng)[0]
    
    # We will pick 3 random components for each fact
    for i in range(num_facts):
        s_idx, r_idx, o_idx = rng.choice(N, size=3, replace=False)
        s_vec = vocab[s_idx]
        r_vec = vocab[r_idx]
        o_vec = vocab[o_idx]
        
        # Bind with roles
        b_s = core.bind(role_s, s_vec)
        b_r = core.bind(role_r, r_vec)
        b_o = core.bind(role_o, o_vec)
        
        # Bundle to create the fact
        fact = core.bundle(np.stack([b_s, b_r, b_o]), rng)
        
        # Unbind object
        unbound_o = core.unbind(fact, role_o)
        
        # Measure similarity with the TRUE target
        sim_true = (unbound_o @ o_vec) / D
        signal_cosines.append(sim_true)
        
    signal_cosines = np.array(signal_cosines)
    
    # --- Plotting ---
    print("Generating plot...")
    plt.figure(figsize=(10, 6))
    
    # Plot Noise
    # Subsample noise for faster plotting
    plt.hist(np.random.choice(noise_flat, size=min(len(noise_flat), 100000), replace=False), 
             bins=100, density=True, alpha=0.7, color='red', label='Noise (Irrelevant Facts)')
    
    # Plot Signal
    plt.hist(signal_cosines, bins=50, density=True, alpha=0.7, color='green', label='Signal (Correct Decodes)')
    
    # Lines for thresholds
    plt.axvline(x=0.35, color='black', linestyle='--', linewidth=2, label='Threshold = 0.35')
    
    plt.title('VSA Clean-up: Signal vs Noise Distributions (D=10,000, 3-term bundle)', fontsize=14)
    plt.xlabel('Cosine Similarity (Score)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../vsa_distributions.png'))
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    
    # --- Benchmarking Early Exit ---
    print("\n--- Benchmarking Clean-up (N=10,000) ---")
    
    # Simulating the query loop:
    # Let's say we have K=5 candidates. 
    # For a full scan, we do clean-up 5 times.
    # For early exit, we do it 1 time (since the true fact is usually first).
    
    dummy_unbound = core.random_atoms(5, D, rng)
    
    # Full Top-K (K=5)
    start = time.perf_counter()
    for i in range(5):
        # clean-up: dot product with entire vocabulary N=10,000
        sims = (dummy_unbound[i] @ vocab.T) / D
        best_idx = np.argmax(sims)
    end = time.perf_counter()
    full_scan_time = (end - start) * 1000 # ms
    
    # Early Exit (assuming first fact is the true one and triggers threshold)
    start = time.perf_counter()
    for i in range(1):
        sims = (dummy_unbound[i] @ vocab.T) / D
        best_idx = np.argmax(sims)
    end = time.perf_counter()
    early_exit_time = (end - start) * 1000 # ms
    
    print(f"Full Scan (K=5) Time: {full_scan_time:.2f} ms")
    print(f"Early Exit Time:      {early_exit_time:.2f} ms")
    print(f"Speedup:              {full_scan_time / early_exit_time:.2f}x")
    
if __name__ == "__main__":
    run_simulation()
