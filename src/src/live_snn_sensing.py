#!/usr/bin/env python3
"""
Live Spectrum Sensing using SNN + RTL-SDR
Terminal-based version with Cognitive Radio Decision
"""

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import subprocess
import time
from collections import deque

# ====================== MODEL ======================
class SpectrumSensingNet(nn.Module):
    def __init__(self, input_size=128, hidden=64, output=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=surrogate.fast_sigmoid(slope=25))
        self.fc2 = nn.Linear(hidden, output)
        self.lif2 = snn.Leaky(beta=0.95, spike_grad=surrogate.fast_sigmoid(slope=25))

    def forward(self, x, num_steps):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x[:, :, step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec)

# ====================== HELPER FUNCTIONS ======================
def rate_encode(signal, n_timesteps=50):
    magnitude = np.abs(signal)
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    spike_train = np.zeros((len(magnitude), n_timesteps))
    for i, rate in enumerate(magnitude):
        spikes = np.random.rand(n_timesteps) < rate
        spike_train[i] = spikes.astype(float)
    return spike_train

def extract_features(iq):
    magnitude = np.abs(iq)
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()
    return magnitude

def capture_iq(freq_hz=945e6, num_samples=128, sample_rate=2.048e6, gain=40):
    """Capture IQ samples using rtl_sdr command-line tool"""
    cmd = [
        "rtl_sdr",
        "-f", str(int(freq_hz)),
        "-s", str(int(sample_rate)),
        "-n", str(num_samples * 2),
        "-g", str(gain),
        "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True, timeout=3)
        data = np.frombuffer(result.stdout, dtype=np.uint8)
        iq = (data[0::2].astype(np.float32) - 127.5) + 1j * (data[1::2].astype(np.float32) - 127.5)
        return iq[:num_samples]
    except Exception as e:
        print(f"Capture Error: {e}")
        return np.zeros(num_samples, dtype=np.complex64)

# ====================== MAIN ======================
def main():
    print("Loading SNN model...")
    model = SpectrumSensingNet()
    
    # Load trained weights (make sure the file is in the correct path)
    model.load_state_dict(torch.load("models/spectrum_snn_real_fading_weights.pth", map_location="cpu"))
    model.eval()
    print("Model loaded successfully!\n")

    # ========== CHANGE FREQUENCY HERE ==========
    center_freq = 945e6          # Examples: 945e6, 900e6, 98.3e6, 100.1e6
    # ===========================================

    num_steps = 50
    threshold = 0.70
    history = deque(maxlen=5)

    print(f"Listening at {center_freq/1e6:.3f} MHz")
    print(f"Decision Threshold : {threshold}")
    print("Press Ctrl+C to stop\n")
    print(f"{'Time':<10} {'Conf':<8} {'Avg':<8} {'Decision':<12} {'Cognitive Radio Action'}")
    print("-" * 75)

    try:
        while True:
            iq = capture_iq(freq_hz=center_freq)
            feat = extract_features(iq)
            spikes = rate_encode(feat, n_timesteps=num_steps)
            x = torch.FloatTensor(spikes).unsqueeze(0)

            with torch.no_grad():
                spk_out = model(x, num_steps)
                conf = spk_out.sum(0).softmax(dim=1)[0][1].item()

            history.append(conf)
            avg_conf = np.mean(history)

            if avg_conf > threshold:
                decision = "OCCUPIED"
                action = "Primary User active → Secondary User must stay silent"
            else:
                decision = "VACANT"
                action = "Channel free → Secondary User can transmit"

            print(f"{time.strftime('%H:%M:%S'):<10} {conf:<8.3f} {avg_conf:<8.3f} {decision:<12} {action}")
            time.sleep(0.4)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
