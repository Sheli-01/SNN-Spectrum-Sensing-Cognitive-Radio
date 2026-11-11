# SNN-based Spectrum Sensing for Cognitive Radio Networks

**Spiking Neural Network based Real-time Spectrum Sensing using Raspberry Pi and RTL-SDR**

This project implements a low-power Spiking Neural Network (SNN) for binary spectrum sensing (Occupied / Vacant) in Cognitive Radio systems. The system is deployed on a Raspberry Pi 4 interfaced with an RTL-SDR Blog V3 and features a real-time Streamlit dashboard with live spectrum, waterfall display, and intelligent vacant channel selection.

---

## Project Highlights

- LIF-based Spiking Neural Network (128 → 64 → 2)
- Rate coding for spike generation
- Real-time spectrum sensing using RTL-SDR Blog V3
- Streamlit dashboard with Live Spectrum + Waterfall
- Antenna Profile selection
- Automatic **Find Best Vacant Channel** feature
- Cognitive Radio decision support (Secondary User can transmit / must stay silent)
- Trained on synthetic IQ data + real multipath dataset

---

## System Architecture
Antenna → RTL-SDR Blog V3 → IQ Capture → Magnitude + Normalization
→ Rate Coding → LIF-SNN (128→64→2) → Occupied/Vacant Decision
→ Cognitive Radio Layer → Streamlit Dashboard


---

## Hardware Requirements

- Raspberry Pi 4 (4GB/8GB recommended)
- RTL-SDR Blog V3
- Suitable Antenna (Dipole recommended)
- 5V 3A Power Supply
- MicroSD Card (32GB+)

## Software Requirements

- Raspberry Pi OS (64-bit)
- Python 3.9+
- PyTorch
- snnTorch
- Streamlit
- NumPy, SciPy, Matplotlib
- rtl-sdr tools

---

## Repository Structure
├── models/                  # Trained SNN weights (.pth files)
├── src/
│   ├── live_snn_sensing.py  # Terminal-based live sensing
│   └── snn_dashboard.py     # Streamlit dashboard
├── notebooks/               # Training notebooks (Phase 1 & Phase 2)
├── images/                  # Block diagrams and dashboard screenshots
├── docs/                    # Project report and presentation
├── hardware/                # Hardware setup guide
├── requirements.txt
└── README.md



---

## How to Run

### 1. Install Dependencies

python -m venv snn_env
source snn_env/bin/activate        # On Windows: snn_env\Scripts\activate
pip install -r requirements.txt

### 2. Run Terminal Live Sensing

python src/live_snn_sensing.py

### 3. Run Streamlit Dashboard
   streamlit run src/snn_dashboard.py

   Then open: http://localhost:8501


### Key Features of Dashboard
1. Real-time Occupied / Vacant decision
2. Confidence score
3. Signal Power (dB) and Estimated SNR
4. Live Spectrum plot
5. Live Waterfall display
6. Antenna Profile selection (FM / Cellular / Wideband)
7. Find Best Vacant Channel (automatic scanning & switching)
8. Cognitive Radio action message
9. Model Details
Neuron Model: Leaky Integrate-and-Fire (LIF)
Architecture: 128 → 64 → 2
Encoding: Rate Coding
β (Leak factor): 0.95
Time steps: 50
Training: Supervised (CrossEntropyLoss + Adam)


## Frequency Bands Used

| Purpose                        | Frequency Range                          |
|--------------------------------|------------------------------------------|
| Live Hardware Testing          | 88–108 MHz (FM) and 700–960 MHz (Cellular) |
| Mendeley Dataset (Offline)     | 2.4 GHz                                  |
| RTL-SDR Blog V3 Limit          | ~500 kHz to 1.766 GHz                    |

---

## Results Summary

- **Synthetic Data**: High accuracy (up to ~100% under controlled conditions)
- **Real Multipath Dataset (Hybrid CNN+SNN)**: **83.54%** Validation Accuracy
- **Live Hardware**: Successfully detects Occupied and Vacant channels on FM and low-band cellular frequencies

---

## Team

- A L Gowripriya (1BI23ET001)
- Jeevika A N (1BI23ET021)
- N Thanushree (1BI23ET035)
- Y G S Shreeraksha (1BI23ET054)

**Guide**: Prof. Ranjana K  
**Institution**: Bangalore Institute of Technology

---

## Future Work

- Multi-class modulation recognition on hardware
- STDP / unsupervised learning
- Cooperative spectrum sensing
- Deployment on neuromorphic hardware
- Energy consumption measurement

---

## License

This project is intended for academic and educational purposes.





