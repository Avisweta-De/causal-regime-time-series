# 🚀 Deployment Guide - Streamlit Cloud

This guide walks you through deploying the Causal Regime Trading Strategy dashboard to Streamlit Cloud (free tier available).

## Quick Start (3 Steps)

### Step 1: Push Code to GitHub ✅
Your code is already in GitHub! If you haven't pushed the latest changes:
```bash
git add .
git commit -m "Add Streamlit app for interactive dashboard"
git push origin main
```

### Step 2: Create Streamlit Cloud Account 
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign up with your GitHub account
3. Click "Deploy an app"

### Step 3: Configure Deployment
1. Select your repository: `causal-regime-time-series`
2. Main file path: `app.py`
3. Branch: `main`
4. Click "Deploy"

**That's it!** 🎉 Your app will be live in ~2 minutes.

---

## Detailed Deployment Steps

### Prerequisites
- ✅ Git repository on GitHub (done)
- ✅ `app.py` in root directory (done)
- ✅ `requirements.txt` with dependencies (updated)
- ✅ `.streamlit/config.toml` configuration (created)

### Streamlit Cloud Automatic Deployment

**Why Streamlit Cloud?**
- ✅ Free tier (unlimited public apps)
- ✅ One-click GitHub integration
- ✅ Auto-deploys on every GitHub push
- ✅ HTTPS by default
- ✅ Shareable public URL
- ✅ Built-in analytics

**Deployment Process:**

1. **Go to Streamlit Cloud**
   - URL: https://share.streamlit.io
   - Sign in with GitHub

2. **Create New App**
   - Click "New app"
   - Select your repository
   - Branch: `main`
   - File path: `app.py`

3. **Advanced Settings (Optional)**
   - Python version: 3.11
   - Client timeout: 30 seconds
   - Secrets (if you add API keys later)

4. **Deploy**
   - Click "Deploy"
   - Wait 1-2 minutes for build to complete
   - Your app gets a public URL: `https://<app-name>.streamlit.app`

### Access Your Deployed App
Once deployed, your app will be available at:
```
https://causal-regime-trading.streamlit.app
```
(Note: Actual URL depends on GitHub repo name)

### Share Your App
```
Direct link: https://causal-regime-trading.streamlit.app
Markdown: [View Dashboard](https://causal-regime-trading.streamlit.app)
Update README.md with the link!
```

---

## Alternative Deployments

### Heroku Deployment (Paid after free tier)
```bash
# Install Heroku CLI, then:
heroku login
heroku create causal-regime-trading
git push heroku main
```

### AWS/GCP/Azure
- Use Docker containerization
- Deploy to Lambda, Cloud Run, or similar
- More complex but more scalable

### Local Development
```bash
streamlit run app.py
```
Opens on `http://localhost:8501`

---

## Troubleshooting

### App Takes Too Long to Load
- **Cause**: Data download on first run
- **Solution**: Streamlit caches data automatically after first load
- First load: 30-60 seconds, subsequent loads: <2 seconds

### "ModuleNotFoundError" Errors
- **Check**: requirements.txt has all dependencies
- **Solution**: Update requirements and push to GitHub (re-deploys automatically)

### App Crashes with "API Limit"
- **Cause**: Yahoo Finance daily limit (~2000 requests/hour)
- **Solution**: Add caching (already implemented with `@st.cache_data`)

### Can't See Changes After Push
- **Solution**: Streamlit Cloud auto-deploys within 1-2 minutes
- Manual redeploy: Click "Rerun" in Streamlit Cloud dashboard

---

## Performance Tips

✅ **Already Implemented:**
- Data caching with `@st.cache_data`
- Efficient date range filtering
- Lightweight Streamlit components

### Further Optimization
If you get "over quota" on Yahoo Finance:
```python
# Use local cached data instead
price_data = pd.read_csv('data/processed/market_with_regimes.csv', index_col=0, parse_dates=True)
```

---

## Update Your README.md

Add this section to your README.md to link to the deployed app:

```markdown
## 🌐 Live Dashboard

**[View Interactive Dashboard](https://causal-regime-trading.streamlit.app)** 

Try the real-time dashboard with:
- Market regime detection (Bull/Neutral/Crisis)
- Causality heatmaps
- Walk-forward backtesting results
- Customizable date ranges and assets
```

---

## Security & Privacy

**Streamlit Cloud Default Security:**
- ✅ HTTPS encryption (automatic)
- ✅ No data stored on servers
- ✅ All computation runs in-session
- ✅ Public = anyone can view, private = GitHub members only

**Make Repo Public:**
If you want others to see your app, make your GitHub repo public:
- GitHub → Settings → Change repository visibility → Public

---

## Getting Your Share Link

Once deployed, share with investors/employers:

**Best Formats:**
1. **Direct Link**: `https://causal-regime-trading.streamlit.app`
2. **QR Code**: Streamlit generates QR in share menu
3. **Embedded**: `<iframe src="https://causal-regime-trading.streamlit.app"></iframe>`
4. **Portfolio**: Add link to your portfolio website

---

## Next Steps

1. ✅ Deploy to Streamlit Cloud (this guide)
2. Add to portfolio website
3. Share with potential employers/investors
4. Collect feedback and iterate
5. Add real trading capabilities (optional, requires broker API)

---

## Support

- 📚 Streamlit Docs: https://docs.streamlit.io
- 🐛 Issues: Check Streamlit Cloud logs
- 💬 Community: https://discuss.streamlit.io

---

**Happy Deploying!** 🚀
