# ⚡ Smart Energy Optimizer

AI-powered appliance scheduling system that uses Reinforcement Learning and IBM Watson ML to minimize electricity bills through Real-Time Pricing (RTP) optimization.

## Features
- 🤖 **Q-Learning Agent** - Trained RL model for optimal scheduling
- ☁️ **IBM Watson ML** - Cloud-based predictions
- 📊 **Real-Time Pricing** - Dynamic price-aware optimization
- 💰 **Cost Savings** - Shift flexible appliances to off-peak hours

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
| File | Description |
|------|-------------|
| `app.py` | Streamlit dashboard |
| `agent.py` | Q-Learning agent (Double Q-Learning) |
| `rtp_model.py` | RTP price generator |
| `utils.py` | Utilities & state discretization |
| `ibm_integration.py` | IBM Watson ML connector |
| `train.py` | Training script |

## Architecture
```
User Input → State Discretization → IBM Watson / Local Agent → Optimal Slot → Dashboard
```

## License
MIT
