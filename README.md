# 🏀 NBA Player Scouting Dashboard

A live analytics dashboard for advanced player scouting using **nba_api** and **Streamlit**.

This tool lets you:
- Analyze any NBA player’s recent and career performance
- View advanced box score trends (PTS, REB, AST, 3PM)
- See opponent defensive and pace context
- Generate simple per-game projections (PTS, REB, AST, PRA)
- Compare windows (L5, L10, L20, Season, Career)
- Review past games vs selected opponents

---

## 🚀 Features
- Dynamic data from the official NBA stats API (`nba_api`)
- Fully interactive dashboard (Streamlit)
- Component-level projections using shot types & rebound splits
- Built-in opponent adjusters using team defensive rating and pace proxy
- Lightweight and deployable anywhere

---

## 🧰 Requirements
Python 3.10+
Dependencies (installed automatically on deployment):
```bash
pip install -r requirements.txt
