# Bankruptcy Prediction using data from 10-K reports
We propose a multimodal approach designed to handle sparse data in the task of bankruptcy prediction. <br>

Run inside data folder: <br><br>
`unzip sec_fins_sparse.zip` <br>

|        Financial Dataset Info               |        |
|----------------------------------|-------------|
| Number of features                       | 9700       |
|Number of samples | 39365 |
| Number of Bankrupt cases                 | 194       |
| Mean of non-NaN number of features per sample | 208.24      |
| Std of non-NaN number of features per sample   | 94.12       |
|Training samples | 27555|
|Validation samples | 3944 |
|Test samples | 7866|
|Training <em>bankrupt</em> samples | 136|
|Validation <em>bankrupt</em> samples | 19 |
|Test <em>bankrupt </em> samples | 39|


<br><br>

|         Density Percentage of 20 most dense features               |        |
|----------------------------------|-------------|
| IncomeTaxExpenseBenefit                        |    0.885330  |
| CashAndCashEquivalentsAtCarryingValue          |    0.922317|
| StockholdersEquity                             |    0.864347|
| Assets                                         |    0.861222|
| LiabilitiesAndStockholdersEquity               |    0.855023|
| NetIncomeLoss                                  |    0.845828|
| NetCashProvidedByUsedInInvestingActivities     |    0.813540|
| NetCashProvidedByUsedInFinancingActivities    |    0.813540|
| NetCashProvidedByUsedInOperatingActivities     |    0.811482|
| RetainedEarningsAccumulatedDeficit             |    0.801702|
| EarningsPerShareBasic                          |    0.801651|
| WeightedAverageNumberOfSharesOutstandingBasic  |    0.795351|
| DeferredIncomeTaxExpenseBenefit                  |  0.792455|
| WeightedAverageNumberOfDilutedSharesOutstanding  |  0.787705|
| ComprehensiveIncomeNetOfTax                      |  0.765934|
| PropertyPlantAndEquipmentNet                     |  0.762733|
| ShareBasedCompensation                           |  0.753461|
| AccumulatedOtherComprehensiveIncomeLossNetOfTax  |  0.752140|
| CommonStockSharesAuthorized                      |  0.726839|

<br>

Usage: <br>

`python main.py --model_type XGBoost --modality financial --n_estimators 10 --max_depth 3`