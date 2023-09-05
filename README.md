# Store-Sales-and-A-B-Test-Analysis.
This repository features A/B test analysis for a fast-food restaurant's menu launch. It covers data preparation, exploratory analysis, statistical tests, modeling, and visualizations. Promotion A, with high sales and consistent growth, is recommended for a nationwide launch. Ongoing monitoring is advised.
Store Sales and A/B Test Analysis
Overview
This repository contains code, data, and documentation for an A/B testing analysis to identify an optimal promotional strategy for a fast food restaurant's new menu item launch.

The repository includes a Jupyter notebook walking through the data preparation, exploratory analysis, statistical testing, predictive modeling, and evaluation of results. Interactive visualizations are generated using Plotly and Bokeh.

The analysis compares performance metrics across three promotional strategies tested in a sample of restaurants over a four week period. Key techniques include ANOVA testing, linear regression, and time series forecasting.

Based on having the highest total sales and most consistent weekly growth, Promotion A is identified as the best performing strategy. The recommendation is to launch the new menu item nationwide using Promotion A.

Ongoing monitoring is advised to ensure continued positive performance once implemented. Adjustments to the promotional mix may be required based on the data insights.

This repository provides a template for a comprehensive A/B testing analysis covering the full process from data to visualization to modeling and recommendations. The documented methodology and interpreted results offer guidance for data-driven decision making.

Store Sales Analysis
Business Context
The Store Sales Analysis focuses on understanding the sales performance of a chain of stores. Key objectives include assessing the impact of promotions, analyzing market sizes, and identifying weekly sales trends. Time series forecasting is also conducted to predict future sales.

Data Overview
Data Source: dataset.csv
Columns:
MarketID, LocationID: Store identifiers
Promotion: A, B, or C
Week: Weeks 1 to 4
SalesInThousands: Weekly sales figures
Analysis Approach
Data Preparation and Exploration:

Loading the dataset.
Creating new columns for total sales and average weekly sales.
Dummy encoding for promotions.
Adding a "Market Size" column based on Market IDs.
Exploratory Data Analysis (EDA):

Analyzing overall metrics, including total sales and average weekly sales by promotion.
Conducting a statistical test to compare the impact of different promotions.
Investigating total sales by market size.
Data Visualization:

Creating visualizations using Seaborn and Plotly Express.
Visualizing weekly sales by promotion using a Seaborn boxplot.
Visualizing total sales by market size using a Seaborn bar plot.
Week-over-Week Trends Analysis:

Analyzing week-over-week sales trends for different promotions.
Plotting weekly sales trends using Seaborn and Matplotlib.
Interactive Plots with Bokeh:

Creating interactive plots using Bokeh, including a bar chart, box plot, and line chart.
Visualizing store size vs. sales with heatmaps and size categories.
Time Series Forecasting:

Implementing time series forecasting using linear regression.
Evaluating the forecasting model with metrics like RMSE and R-squared.
Forecasting future sales based on the trained model.
A/B Test Analysis
Business Context
The A/B Test Analysis focuses on comparing three promotional strategies (A, B, C) for a fast food restaurant's new menu item launch. The goal is to identify the most effective promotional strategy based on weekly sales data.

Data Overview
Data Source: data/
Contents:
analysis.ipynb: Jupyter notebook with the complete analysis.
data/: Folder containing raw input data CSV files.
outputs/: Folder with generated plots and results.
requirements.txt: Package dependencies.
Key Results
The A/B Test Analysis revealed the following key findings:

Promotion A had the highest total sales.
Weekly sales increased steadily for Promotion A.
Statistical tests confirmed a significant difference in performance among the promotions.
A linear model predicted a 15% sales lift for Promotion A.
Forecasting predicted a continued positive trajectory under Promotion A.
Recommendations
Based on the A/B Test Analysis, the following recommendations are provided:

Launch the new menu item nationally using Promotion A.
Continue monitoring key performance indicators (KPIs) to ensure no drop-off after the launch.
Be prepared to adjust promotional strategies quickly if data indicates a decline.
Next Steps
The next steps for both analyses include:

Finalizing a detailed launch plan for the national roll-out.
Developing a dashboard to track performance metrics in real-time.
Optimizing the marketing mix based on ongoing results.
Author
Your Name
