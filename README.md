# DS 4320 Project 2: Borrower Risk Assessment and Loan Default Prediction

### Executive Summary
This repository presents a data science project focused on analyzing borrower financial profiles and predicting loan default risk using applicant-level data. The project includes a dataset sourced from Kaggle and transformed into a secondary dataset D1 using MongoDB Atlas as the document store. The workflow covers data acquisition, transformation,insertion into MongoDB, feature engineering, and implementation of a Random Forest classification model that  generates a default risk score between 0 and 1 for every borrower. Supporting materials such as a data dictionary, schema guidelines, bias analysis, background research, and a press release are also included to provide context and ensure transparency, reproducibility, and accessibility for both technical and non-technical audiences.

### Project Information

| Spec           | Value |
|----------------|-------|
| Name           | Leah Kim |
| NetID          | sxk2eh |
| DOI            | [https://doi.org/10.5281/zenodo.19355541](https://doi.org/10.5281/zenodo.19355541) |
| Press Release  | [Identifying Default Risk Earlier Could Help Lenders Make Better Loan Decisions](press-release/press_release.md)  |
| Pipeline       | [data_creation.ipynb](https://github.com/leahhkim/project2-loan-default/blob/main/data_creation/data_creation.ipynb), [data_creation.md](https://github.com/leahhkim/project1-sports-outcomes/blob/main/data_creation/data_creation.md)
 | License        | [MIT](LICENSE) |

## Problem Definition
### General and Specific Problem
* **General Problem:** "Predicting loan default risk."
* **Specific Problem:** "It is often difficult for lenders to know in advance which borrowers are more likely to miss payments or default on a loan. I want to explore how applicant financial information and past repayment behavior can be used to predict default risk and develop a default risk score that estimates how likely a borrower is to default."

### Motivation
Loan default risk is an important problem because lenders have to decide whether a borrower is likely to repay a loan before approving it. This can be difficult because borrowers may look similar at first, but their payment behavior and financial history can lead to very different outcomes. I chose this project because it is a real-world problem where data can help reveal patterns that are not obvious right away. By studying loan default risk, I can explore how borrower information may be used to support better lending decisions and reduce uncertainty.

### Rationale
I refined the general problem by focusing on a borrower’s default risk score rather than only a yes-or-no default prediction. Predicting loan default can be difficult, even when a lender has access to financial background and repayment history. Instead of simply classifying borrowers as likely to default or not, a default risk score provides a more useful way to show how likely a borrower is to default based on past behavior and financial characteristics. This allows the project to better capture the uncertainty of lending decisions while still making it possible to analyze which factors may contribute most to default risk.

### Press Release Headline and Link (click to view)
[Identifying Default Risk Earlier Could Help Lenders Make Better Loan Decisions](press-release/press_release.md) 

## Domain Exposition

### Terminology

| Term / KPI | Meaning in This Project | Why It Matters |
|---|---|---|
| Loan Default | When a borrower fails to repay the loan as agreed | This is the main outcome the project is trying to predict |
| Default Risk | The likelihood that a borrower will fail to repay a loan | This is the central idea of the project |
| Default Risk Score | A score or probability showing how likely a borrower is to default | This gives a more useful result than only predicting yes or no |
| Repayment History | A record of whether the borrower has made past payments on time | This is one of the most important indicators of future default |
| Credit Limit | The maximum amount of credit available to the borrower | This helps describe the borrower’s financial situation and may relate to default risk |

### Domain
This project lives in the domain of finance, credit risk analysis, and predictive modeling. Credit risk analysis uses borrower financial data and repayment behavior to better understand patterns that may be associated with loan default. In this project, borrower information such as credit limit, bill amounts, payment amounts, and repayment history is used to explore which factors may influence whether a borrower defaults. It also connects to data-driven forecasting, since the goal is to estimate a borrower’s likelihood of default based on past financial behavior.

### Background Reading

[Link to Background Readings](https://myuva-my.sharepoint.com/:f:/g/personal/sxk2eh_virginia_edu/IgAwZ-htJfbOR5hhCWcNN3BLAd14KyBpf62C_qiHLEDdrAs?e=q9NlzY)


### Readings Summary

| Title | Brief Description | Link to File |
|---|---|---|
| Understanding Default Risk: Definition, Types, and Ways to Measure It | Explains what default risk means and why it matters to lenders and financial institutions. This reading helps establish the core domain concept behind the project. | [Open file](https://myuva-my.sharepoint.com/:b:/g/personal/sxk2eh_virginia_edu/IQCGF0PzzgTZSacRgZcYbPrkAX_IpRK3Ii55XkiGsKsingA?e=RePjB2) |
| Understanding Credit Risk: Definitions, Ratings, and Key Examples | Provides background on credit risk as the broader concept that includes the possibility of borrower nonpayment. This helps connect loan default to the larger field of lending and financial risk. | [Open file](https://myuva-my.sharepoint.com/:b:/g/personal/sxk2eh_virginia_edu/IQAuJELNYj5RQZUfSH2mCTEZAZxPah3FFhWMMyoIheqZSpI?e=cqA5fa) |
| What is Payment History? | Describes how payment history affects credit scores and why missed or late payments are important indicators of borrower risk. This is relevant because repayment behavior is closely tied to default risk. | [Open file](https://myuva-my.sharepoint.com/:b:/g/personal/sxk2eh_virginia_edu/IQAMHQsil5sYSI--KYdxatztAfHec4Siut6J70IygJU0anM?e=ug4RBI) |
| What Is Payment History and How Does It Impact Your Credit? | Explains payment history in simple terms and shows how it influences a borrower’s credit profile. This helps support the idea that past repayment behavior may be useful in predicting future default. | [Open file](https://myuva-my.sharepoint.com/:b:/g/personal/sxk2eh_virginia_edu/IQATxp0nNDMMTKMVxUjsjp5QATF4_XwPppb1RRa8jXXTIRg?e=Rwcw3H) |
| What Affects Your Credit Scores? | Explains the main factors that influence credit scores, including payment history, credit usage, and account activity. This reading helps show which borrower characteristics may relate to loan default risk. | [Open file](https://myuva-my.sharepoint.com/:b:/g/personal/sxk2eh_virginia_edu/IQBF_cUnvsAAS6Fsw3mIYYLvARuwo05VbuK9pehIML7Wefc?e=SrrKhA) |


## Data Creation
### Raw Data Acquisition Process

To build the dataset for this project, I sourced a publicly available dataset from Kaggle titled "Loan Default" by Nikhil1e9. The dataset includes borrower-level financial information such as income, loan amount, credit history, and past repayment behavior, all of which are relevant to understanding and predicting loan default risk. I chose this dataset because it captures the kinds of variables a real lender would have access to when evaluating a borrower, making it a strong fit for my goal of developing a default risk score.
After downloading the dataset as a CSV file from Kaggle, I reviewed the available fields to identify which variables were most useful for my specific problem. Since I am interested in predicting how likely a borrower is to default rather than just whether they will, I paid particular attention to features that reflect financial stability and repayment patterns over time. The dataset was then loaded into my analysis environment where it serves as the foundation for all further processing, feature engineering, and modeling in this project.

### Code to Create Data

| File | Description | Link |
|------|-------------|------|
| `data/Loan_default.csv` | Raw loan applicant dataset downloaded from Kaggle containing 255,347 borrower records with financial and repayment history features | [Loan_default.csv](https://github.com/leahhkim/project2-loan-default/blob/main/data/Loan_default.csv) |
| `data_creation/data_creation.ipynb` | Python notebook that reads the raw CSV and inserts all 255,347 records into MongoDB Atlas under the `loan_default` collection | [data_creation.ipynb](https://github.com/leahhkim/project2-loan-default/blob/main/data_creation/data_creation.ipynb) |

### Rationale for Critical Decisions

Several important decisions were made when creating and preparing the dataset that could introduce or mitigate uncertainty. First, I selected the Kaggle loan default dataset and chose to focus on borrower-level financial and demographic features, since these variables are most relevant for predicting default risk. This involved deciding which features to include, which may introduce uncertainty because excluding certain variables such as borrower location or credit history details could limit the model's ability to capture all factors that influence whether a borrower defaults. Another key decision was how to structure the data in MongoDB, including choosing to store each loan application as its own document with all features at the top level rather than using nested structures. These design choices can affect how easily the data can be queried and may influence how features are prepared for modeling. Additionally, decisions were made about how to handle the raw data before insertion, such as preserving original string values for categorical fields like employment type and education level rather than encoding them at the storage stage, which pushes that uncertainty into the analysis pipeline instead. To mitigate these effects, I focused on using consistent data preparation methods, dropping and reinserting the collection on every run to ensure clean data, and documenting all assumptions so that the impact of these decisions remains transparent and reproducible.

### Bias Identification

Bias in this dataset could be introduced through several aspects of how the data was originally collected and structured. Since this is a secondary dataset compiled and published on Kaggle, it inherits any biases present in the original lending records it was derived from. For example, the borrowers represented in the dataset may not reflect the full population of loan applicants if certain demographics, income groups, or geographic regions were more likely to be included than others, which would be an example of selection bias. Additionally, features like employment type and education level may reflect societal inequalities that already exist in lending practices, meaning any model trained on this data could reinforce those existing biases rather than correct for them. Since the dataset was created for analytical purposes rather than as a direct export of real lending records, the decisions made by the dataset creator when generating or sampling the data could also introduce bias that is difficult to detect or quantify.

### Bias Mitigation

To mitigate the biases identified in the data collection process, several strategies can be applied during analysis. First, the distribution of sensitive features such as employment type, education level, and marital status should be examined to check whether certain groups are over or underrepresented in the dataset, and if so, resampling techniques like oversampling minority groups or undersampling majority groups can help balance the data before modeling. Second, since the dataset has a natural class imbalance between defaulters and non-defaulters, metrics like precision, recall, and F1-score should be used instead of raw accuracy to avoid a model that simply predicts the majority class. Finally, after building a model, feature importance analysis can be used to check whether the model is relying too heavily on features that may serve as proxies for sensitive demographic characteristics, and those features can be removed or adjusted to reduce the risk of the model producing biased predictions.

## Metadata

### Implicit Schema

The `loan_default` collection stores one document per loan application.
All documents should follow these structural conventions:

- Every document must contain all 26 fields listed below
- Field names must use PascalCase for original fields (e.g. `LoanAmount` not `loan_amount`) and snake_case for derived D1 fields (e.g. `credit_tier`, `risk_flag`)
- Boolean-like fields (`HasMortgage`, `HasDependents`, `HasCoSigner`) must be stored as `"Yes"` or `"No"` strings, not booleans
- The `Default` and `risk_flag` fields must be stored as integers — `0` or `1`
- Numeric fields (`Income`, `LoanAmount`, etc.) must not be stored as strings
- All fields are stored at the top level — there is no nested structure or embedded documents in this collection
- No additional or renamed fields should be added without updating this schema guide
- The `LoanID` field must be unique across all documents

### Data Summary

| Database Component | Description |
|--------------------|-------------|
| Database Name | `sxk2eh` |
| Collection Name | `loan_default` |
| Source Dataset | `Loan_default.csv` |
| Document Level | Each document represents one borrower and one loan application |
| Contents Summary | The collection contains borrower-level financial, demographic, and loan-related information used to study loan default risk |
| Original Fields | LoanID, Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio, Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner, Default |
| Derived Fields | credit_tier, income_bracket, debt_burden, loan_size, default_rate_by_employment, default_rate_by_purpose, default_rate_by_education, risk_flag, default_risk_score|
| Target Variable | `Default`, indicating whether the borrower defaulted on the loan |
| Total Documents | `255,347` |

### Data Dictionary

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `LoanID` | String | Unique identifier for each loan application | `L-10001` |
| `Age` | Integer | Borrower age in years | `35` |
| `Income` | Integer | Annual income in U.S. dollars | `65000` |
| `LoanAmount` | Integer | Total loan amount in U.S. dollars | `15000` |
| `CreditScore` | Integer | Borrower credit score | `720` |
| `MonthsEmployed` | Integer | Number of months the borrower has been at their current employer | `48` |
| `NumCreditLines` | Integer | Number of open credit lines the borrower has | `5` |
| `InterestRate` | Float | Loan interest rate as a percentage | `11.5` |
| `LoanTerm` | Integer | Loan duration in months | `36` |
| `DTIRatio` | Float | Debt-to-income ratio | `0.32` |
| `Education` | String | Highest education level attained by the borrower | `Bachelor's` |
| `EmploymentType` | String | Borrower’s type of employment | `Full-time` |
| `MaritalStatus` | String | Marital status of the borrower | `Married` |
| `HasMortgage` | String | Whether the borrower currently has a mortgage | `No` |
| `HasDependents` | String | Whether the borrower has dependents | `Yes` |
| `LoanPurpose` | String | Stated purpose of the loan | `Home Improvement` |
| `HasCoSigner` | String | Whether the borrower has a co-signer | `No` |
| `Default` | Integer | Repayment outcome where `0` means no default and `1` means default | `1` |
| `credit_tier` | String | Category created from `CreditScore`, such as Poor, Fair, Good, or Excellent | `Good` |
| `income_bracket` | String | Category created from `Income`, such as Low, Medium, or High | `Medium` |
| `debt_burden` | String | Category created from `DTIRatio`, such as Low, Moderate, or High | `High` |
| `loan_size` | String | Category created from `LoanAmount`, such as Small, Medium, or Large | `Medium` |
| `default_rate_by_employment` | Float | Average default rate for borrowers with the same employment type | `0.18` |
| `default_rate_by_purpose` | Float | Average default rate for borrowers with the same loan purpose | `0.24` |
| `default_rate_by_education` | Float | Average default rate for borrowers with the same education level | `0.15` |
| `risk_flag` | Integer | Indicator where `1` means the borrower meets the project’s high-risk rule and `0` otherwise | `1` |


### Data Dictionary Quantification of Uncertainty

| Feature | Mean | Std Dev | IQR | Outlier Count | Outlier % | Interpretation |
|---------|------|---------|-----|---------------|-----------|----------------|
| `Age` | 43.50 | 14.99 | 25.00 | 0 | 0.00% | Evenly distributed with no outliers, low uncertainty |
| `Income` | 82,499.30 | 38,963.01 | 67,393.50 | 0 | 0.00% | High spread but no outliers, moderate uncertainty due to self-reporting |
| `LoanAmount` | 127,578.87 | 70,840.71 | 122,829.00 | 0 | 0.00% | Very wide spread indicating high variability across borrowers |
| `CreditScore` | 574.26 | 158.90 | 275.00 | 0 | 0.00% | Wide range across the dataset, no outliers detected |
| `MonthsEmployed` | 59.54 | 34.64 | 60.00 | 0 | 0.00% | High std dev relative to mean suggests significant variability |
| `NumCreditLines` | 2.50 | 1.12 | 1.00 | 0 | 0.00% | Low variability, tight distribution around the mean |
| `InterestRate` | 13.49 | 6.64 | 11.48 | 0 | 0.00% | Moderate spread, no outliers detected |
| `LoanTerm` | 36.03 | 16.97 | 24.00 | 0 | 0.00% | Discrete fixed values (12, 24, 36, 48, 60), low uncertainty |
| `DTIRatio` | 0.50 | 0.23 | 0.40 | 0 | 0.00% | Centered near 0.5, no outliers, moderate spread |
| `Default` | 0.12 | 0.32 | 0.00 | 29,653 | 11.61% | Binary variable — 11.61% flagged as outliers reflects class imbalance |
| `default_rate_by_employment` | 0.1161 | 0.0146 | 0.0050 | 127,480 | 49.92% | High outlier rate because most borrowers cluster around the mean rate with a few employment types pulling away |
| `default_rate_by_purpose` | 0.1161 | 0.0072 | 0.0009 | 102,584 | 40.17% | Very small IQR means most values are tightly packed, making small deviations appear as outliers |
| `default_rate_by_education` | 0.1161 | 0.0093 | 0.0201 | 0 | 0.00% | No outliers, education level default rates are evenly distributed |
| `risk_flag` | 0.1124 | 0.3158 | 0.0000 | 28,689 | 11.24% | Binary variable — 11.24% of borrowers meet all three high risk criteria simultaneously |

