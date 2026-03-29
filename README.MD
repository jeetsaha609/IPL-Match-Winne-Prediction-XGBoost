🏏 IPL Match Winner Prediction using Machine Learning

📌 Project Overview

This project focuses on predicting the winner of IPL matches using historical match and ball-by-ball data. It combines feature engineering, time-aware validation, and machine learning modeling to build a robust prediction system.

Unlike basic models, this project emphasizes:

- Real-world feature design
- Avoiding data leakage
- Time-series consistency
- Model interpretability

---

📊 Dataset Used

- matches.csv → Match-level data (teams, toss, venue, result)
- deliveries.csv → Ball-by-ball data (runs, wickets, players)

---

⚙️ Key Features Engineered

🔥 Team Performance

- "team1_strength", "team2_strength" → Average runs (historical)
- "team1_batsman_strength", "team2_batsman_strength" → Top batsman performance
- "team1_bowler_strength", "team2_bowler_strength" → Best bowler impact

📈 Recent Form

- "team1_form", "team2_form" → Weighted recent performance (last 5 matches)

🤝 Head-to-Head (H2H)

- Historical wins between two teams
- "h2h_diff" → Advantage metric

🏟️ Venue Advantage

- Team win % at a specific venue
- "venue_diff" → Venue bias

🎯 Toss Impact

- Toss win percentage
- Toss advantage indicator

⚖️ Final Engineered Features

- "form_diff"
- "strength_diff"
- "h2h_diff"
- "venue_diff"
- "batting_diff"
- "bowling_diff"
- "toss_advantage"

---

🧠 Model Used

🚀 XGBoost Classifier

- "n_estimators = 300"
- "learning_rate = 0.05"
- "max_depth = 5"
- Subsampling for generalization

📊 Why XGBoost?

- Handles non-linear relationships
- Works well with tabular data
- Provides feature importance

---

🧪 Train-Test Strategy

✔ Time-based split (NO shuffle)

train_test_split(..., shuffle=False)

➡ Ensures:

- No future data leakage
- Real-world simulation

---

📈 Evaluation Metrics

- ✅ Accuracy Score
- ✅ Classification Report (Precision, Recall, F1-score)
- ✅ Feature Importance Visualization

---

📊 Sample Output

- Feature Importance Plot
- Predicted Wins Distribution
- Model Accuracy Score

---

⚠️ Key Learning Highlights

✔ Avoided data leakage using time-based filtering
✔ Built domain-specific features (not generic ML)
✔ Used historical-only data for feature creation
✔ Combined player + team + venue intelligence

---

🚀 Future Improvements

- Add player-level embeddings
- Use advanced models (LightGBM, CatBoost)
- Hyperparameter tuning (Optuna)
- Deploy as a web app (Streamlit / Flask)

---
Model Accuracy: ~50% (evaluated on test split)
Note: Limited test data size may affect accuracy reliability

💡 Conclusion

This project is not just about prediction — it demonstrates:

- Strong understanding of feature engineering
- Proper handling of time-series data
- Ability to build interview-level ML systems

---

👨‍💻 Author

Jeet Saha
Aspiring Data Scientist | Machine Learning Enthusiast

---

⭐ If you found this project useful, feel free to star the repository!
