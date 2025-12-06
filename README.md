# F1 Winner Prediction 🏎️

This project predicts the winner of Formula 1 races using historical race data and machine learning.

## Overview

- Fetches past F1 race results from the Ergast API (2023 and 2024 seasons).  
- Adds manually announced grid data for upcoming races (e.g., Abu Dhabi 2025).  
- Encodes driver and constructor names into numerical features.  
- Uses a Gradient Boosting Classifier to predict the probability of each driver winning.  
- Outputs predicted winning probabilities for the upcoming race.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/f1-winner-prediction.git
cd f1-winner-prediction
pip install -r requirements.txt



