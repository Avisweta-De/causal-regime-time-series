# Contributing to Causal Regime Detection

Thank you for your interest in contributing! This document provides guidelines to make the process smooth for everyone.

---

## 🗺️ Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/causal-regime-time-series.git
   cd causal-regime-time-series
   ```
3. **Set up** a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   # source venv/bin/activate    # macOS/Linux
   pip install -r requirements.txt
   ```
4. **Create a feature branch** off `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## 🧪 Running Tests

Before submitting a PR, run the smoke tests to ensure nothing is broken:

```bash
python -m pytest tests/ -v
```

---

## 📝 Code Style

- Follow **PEP 8** conventions
- Use **type hints** for all function parameters and return types (see existing `src/` modules as reference)
- Write **docstrings** in NumPy style (Parameters / Returns sections)
- Keep functions focused: one function = one responsibility

---

## 📬 Submitting a Pull Request

1. Commit your changes with a descriptive message:
   ```bash
   git commit -m "feat: add Kalman filter regime detector"
   ```
2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
3. Open a **Pull Request** against the `main` branch
4. Fill in the PR description explaining:
   - **What** you changed
   - **Why** (motivation / linked issue)
   - **How** to test it

---

## 💡 Areas for Contribution

| Area | Ideas |
|:---|:---|
| 🤖 **New Detectors** | Kalman filter, Hidden Markov Model (HMM via `hmmlearn`) |
| 📊 **More Assets** | Bonds (TLT), crypto (BTC-USD), sector ETFs |
| 📈 **Strategies** | Kelly criterion, volatility targeting, mean-reversion |
| 🧪 **Tests** | Expand `tests/` with unit tests for each module |
| 📚 **Docs** | Improve notebooks, add inline comments to complex logic |
| 🌐 **Dashboard** | Streamlit or Dash web UI for live regime tracking |
| 🔌 **API** | FastAPI endpoint for real-time regime predictions |

---

## 🐛 Reporting Bugs

Please open an [Issue](https://github.com/Avisweta-De/causal-regime-time-series/issues) and include:
- Python version and OS
- Exact error message and traceback
- Minimal code to reproduce

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
