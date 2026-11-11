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
