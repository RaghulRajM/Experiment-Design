# Experiment-Design

Inspired by Matt Dancho's tutorial, "A/B Testing with Machine Learning - A Step-by-Step Tutorial," I have implemented Machine Learning techniques for A/B Testing. The goal is to enhance the understanding of the effects of digital marketing efforts using A/B testing, a crucial strategy in the era of digital marketing tools like Google Analytics, Google Adwords, and Facebook Ads.

A/B testing is a powerful tool for businesses, allowing them to assess the impact of changes in various aspects of digital marketing, such as landing pages, popup forms, and repo titles. The competitive advantage lies in the ability to make informed decisions based on customer behavior and preferences.

Traditional statistical inference approaches to A/B testing, though effective, fall short in capturing the complexity of customer behavior. Machine Learning emerges as a superior alternative, capable of generating insights from intricate systems where customers follow diverse paths, exhibit varied site engagement times, and come from different backgrounds.

The repo provides a comprehensive overview and step-by-step walkthrough of implementing Machine Learning for A/B Testing using three different algorithms: Linear Regression, Decision Trees, and XGBoost. The emphasis is on understanding A/B Testing, recognizing why Machine Learning surpasses traditional statistical inference, and gaining practical insights for implementation.

### Key Takeaways:

Understanding A/B Testing: A/B Testing involves running parallel tests on a Treatment Group (exposed to changes) and a Control Group (no change) to compare conversion rates through statistical inference.

### Challenges with Traditional Approaches: 

The limitations of traditional approaches become evident in the dynamic real-world scenarios where users have diverse characteristics, spend varying time on the website, and access it through different channels.

### Why Machine Learning: 

Machine Learning addresses the complexities by modeling entire systems, considering ongoing events, user features, and more. The flexibility to combine multiple approaches enhances the depth of insights.

### Step-by-Step Walkthrough: 

The repo guides readers through the implementation of Machine Learning for A/B Testing using the R programming language. It features a real-world experiment named "Free Trial Screener" conducted by Udacity, aiming to set clear expectations for students and reduce frustration among those with limited time commitment.

### Project Goal: 

The analysis investigates the features contributing to enrollments and assesses the impact of the new "Setting Expectations" form on enrollments. The distinction between the experiment group (Experiencing the form) and the control group (Not seeing the form) allows for a detailed evaluation.

This implementation not only aligns with best practices but also offers a practical approach to A/B Testing with Machine Learning, empowering business professionals to make data-driven decisions and advance their careers in the realms of Data Science and Machine Learning.

### Data source:

-  https://www.kaggle.com/datasets/tammyrotem/control-data
-  https://www.kaggle.com/datasets/tammyrotem/experiment-data

### Business Insights - Machine Learning Advantages

Utilizing Machine Learning for A/B Testing yields several crucial advantages:

Understanding Complex Systems: Machine Learning revealed that Pageviews and Clicks drive the system, insights not easily discernible through Statistical Inference.

Direction and Magnitude of Experiments: Machine Learning provided insights into the experiment's impact. For instance, Linear Regression highlighted a drop of -17.6 Enrollments Per Day when Experiment equals 1. Statistical Inference would only indicate the presence or absence of an effect.

### Udacity's Strategic Considerations:

To maximize enrollments, focus on increasing Page Views, identified as the most critical feature in two out of three models.

Introducing a popup form may decrease enrollments if time commitment alerts are vital. Decision Tree rules and Linear Regression both indicate a negative impact when Experiment is less than or equal to 0.5 or equals 1, respectively. Decisions should align with Udacity’s overarching goals, determined collaboratively by business and marketing teams.

### Key Considerations: Cross-Validation and Model Performance Enhancement

#### Enhancing Model Performance

Data Aggregation Issue: The current aggregated data restricts a comprehensive understanding of customer behavior. Analyzing unaggregated data is recommended for a nuanced understanding of individual customer enrollment probabilities.

Customer-Centric Features: Lack of features related to customer characteristics hinders modeling. Inclusion of pertinent features is pivotal for enhancing model performance and gaining insights into complex purchasing behavior.

### Cross-Validation Necessity for Model Tuning

Cross-Validation Significance: Cross-validation is imperative to prevent models from overfitting to the test dataset. 

Model Performance Results: Decision Tree and XGBoost parameters were chosen using 5-Fold Cross Validation, with results indicating the average Mean Absolute Error (MAE). Notably, the baseline Linear Regression model demonstrated comparable performance to XGBoost, suggesting simplicity in the dataset. Anticipated improvements, incorporating model-boosting recommendations, might position XGBoost as the preferred model as system complexity increases.
