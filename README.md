# 🏀 NBA Finals Predictor

A Python project that predicts the NBA Finals winner 
using real time data from the NBA API and a composite 
scoring model validated across 25 years of historical data.

## How It Works
1. Pulls current season stats for all 30 NBA teams
2. Engineers offensive and defensive features
3. Ranks teams across 8 statistical categories
4. Validates model against 25 years of Finals results
5. Predicts the current season Finals winner

## Results
- 2025-26 Predicted Winner: Oklahoma City Thunder
- Historical Accuracy: 32% (2000-2025)
- 10x better than random guessing (3.3%)

## Technologies Used
- Python
- nba_api
- pandas
- matplotlib

## Stats Used
Offensive: Win %, Points per game, Plus/Minus, Turnovers per game
Defensive: Defensive Rating, Steals per game, Defensive Rebounds

## How To Run
1. Install dependencies:
   pip install nba_api pandas matplotlib

2. Run the script:
   python main.py
