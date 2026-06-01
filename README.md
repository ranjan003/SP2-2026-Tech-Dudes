# Sona Power Predict - 2026

**College Name:** Sona College of Technology

**Team Name:** Tech-Dudes

### Team Members

Naveen Prasana S - Year 3, Computer Science and Design (CSD)

Ranjan M - Year 3, Computer Science and Design (CSD)

Siva Sankar C - Year 3, Computer Science and Design (CSD)



---

### Libraries Used in Model

Based on the `mymodelfile.py` submission, the following Python libraries are utilized for data manipulation and mathematical operations.

**pandas** is used for loading CSV files, performing groupby aggregations, handling time-series operations, and constructing the innings-level dataset used for training and evaluation.

**numpy** is utilized for numerical computations, array operations, prediction clipping, statistical calculations, and performance evaluation metrics.

**scikit-learn** provides the machine learning framework required for model development, including HuberRegressor for robust regression, HistGradientBoostingRegressor for ensemble-based experimentation, and StandardScaler for feature normalization.

**scipy** supports optimization-related computations and mathematical utilities used during model calibration and parameter tuning.

**difflib** is used for fuzzy player-name matching and entity resolution, enabling consistent mapping of player records across different datasets.

---

### Model

The prediction framework is based on HuberRegressor, a robust linear regression technique designed to reduce the influence of outliers while maintaining stable predictive performance. Historical IPL powerplay data was analyzed to identify scoring patterns across multiple seasons, venues, teams, batters, and bowlers.

Extensive experimentation was conducted using tree-based ensemble models. Evaluation results indicated that these approaches struggled to adapt to the continuous increase in IPL scoring rates observed over recent seasons. Since powerplay scoring averages have increased significantly over time, models relying purely on historical distributions frequently produced conservative predictions.

The final model incorporates historical scoring trends, player statistics, venue characteristics, batting strength, bowling strength, and matchup-specific information. Residual-based learning enables the model to adapt to evolving scoring environments while maintaining consistency across different seasons. This approach improves predictive reliability and provides balanced powerplay score estimation for modern IPL matches.

---

### License

This project is licensed under the MIT License.
