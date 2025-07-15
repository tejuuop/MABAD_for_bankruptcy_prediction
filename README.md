# Bankruptcy Prediction using data from 10-K reports
We propose a multimodal approach designed to handle sparse data in the task of bankruptcy prediction. <br>

## ⚠️ Key Challenges Recap

| Challenge                       | Description                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| 📊 Severe Class Imbalance       | Next-year bankruptcies are rare (~0.45%), making supervised learning fragile |
| 🧮 Tabular Data with Missing Values | Missing-by-design or structural NaNs must be handled appropriately         |
| 📝 Sparse Text Modalities       | Two sources of text (e.g., MD&A and Auditor comments) are high-dimensional and sparse |
| 🔄 Multimodal Fusion            | Combining heterogeneous sources (tabular + multiple texts) is nontrivial     |
| 🧪 Limited Labeled Data         | Labels may be unreliable, noisy, or unavailable for certain years or firms   |
| 📈 Need for Robust Generalization | Overfitting on the majority class is easy; detecting subtle minority patterns is hard |



|        Stratified Data Split              |        |
|----------------------------------|-------------|
|Training samples | 27328|
|Validation samples | 3912 |
|Test samples | 7801|
|Training <em>bankrupt</em> samples | 134|
|Validation <em>bankrupt</em> samples | 19 |
|Test <em>bankrupt </em> samples | 39|

<br>

|        Time-based Data Split              |        |
|----------------------------------|-------------|
|Training samples (from 1999-12-31 to 2017-12-31) | 29752|
|Validation samples (from 2018-01-01 to 2018-12-31) | 2908 |
|Test samples (from 2019-01-01 to 2021-10-31)| 6381|
|Training <em>bankrupt</em> samples | 126|
|Validation <em>bankrupt</em> samples | 19 |
|Test <em>bankrupt </em> samples | 47|

<br>

|        Financial Dataset Info              |        |
|----------------------------------|-------------|
| Number of features                       | 9700       |
| Mean of number of non-NaN  of features per sample | 210.37      |
| Std of  number of non-NaN features per sample   | 93.93      |



<br>

Usage: <br>

1) git clone

2) Move `/home/andreasideras/research_ideas/BankruptcyPredictionAD/data/` `mda_auditor.csv`, `numerical.csv`, `subset_mda_auditor.csv` and `subset_numerical.csv` to `/data` folder.  Subset files are just smaller dense datasets for quick experimentation (use --subset True) (<span style="color:red">INTERNAL-REMOVE IT</span>). You can use either a stratified or a time-split on the bankrupt cases setting --split from values {'stratified','time'}.


3) You can run any modality by giving the argument --modality a value from {financial, mda_auditor, multimodal}, for example:


    `python train.py --model_type XGBoost --modality multimodal --n_estimators 100 --max_depth 10 --subset False --tfidf_dim 1000`

    This will train the model and save the validation set performance and hyperparameters used in the *hyperparameters* folder. It saves a model only when there was an improvement on the validation performance based on the *config.VAL_METRIC_TO_OPTIMIZE* metric.

4) Test the (best) saved model on the test set:
    
    `python test.py --model_type XGBoost --modality multimodal --subset False`

<br>

To implement your own model, you need to extend the models.base_model.BaseModel class. If some sort of preprocessing is required (e.g., tf-idf extraction - check preprocessing.tfidf_preprocessor.TFIDFpreprocessor), create your own preproccessor by extending the preprocessing.base_preprocessor.BasePreprocessor class. If not, just use the preprocessing.pass_preprocessor.PASSpreprocessor. 

<br><br>

|         Density Percentage of the Top 40 Densest Features               |        |
|----------------------------------|-------------|
|CashAndCashEquivalentsAtCarryingValue  |                                     92.3%|
|IncomeTaxExpenseBenefit                                                 |    88.6%|
|StockholdersEquity                                                       |   86.5%|
|Assets                                                                  |    86.2%|
|LiabilitiesAndStockholdersEquity                                        |    85.6%|
|NetIncomeLoss                                                           |    84.7%|
|NetCashProvidedByUsedInFinancingActivities                              |    81.4%|
|NetCashProvidedByUsedInInvestingActivities                              |    81.4%|
|NetCashProvidedByUsedInOperatingActivities                              |    81.2%|
|RetainedEarningsAccumulatedDeficit                                      |    80.3%|
|EarningsPerShareBasic                                                   |    80.2%|
|WeightedAverageNumberOfSharesOutstandingBasic                           |    79.6%|
|EarningsPerShareDiluted                                                 |    79.5%|
|DeferredIncomeTaxExpenseBenefit                                          |   79.3%|
|WeightedAverageNumberOfDilutedSharesOutstanding                         |    78.8%|
|ComprehensiveIncomeNetOfTax                                             |    76.6%|
|PropertyPlantAndEquipmentNet                                             |   76.4%|
|ShareBasedCompensation                                                  |    75.4%|
|AccumulatedOtherComprehensiveIncomeLossNetOfTax                          |   75.3%|
|CommonStockSharesAuthorized                                             |    72.8%|
|InterestExpense                                                         |    71.1%|
|CommonStockValue                                                         |   69.2%|
|AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment  |  69.0%|
|CommonStockSharesIssued                                                  |   68.9%|
|CashAndCashEquivalentsPeriodIncreaseDecrease                            |    68.3%|
|CommonStockParOrStatedValuePerShare                                     |    67.2%|
|PaymentsToAcquirePropertyPlantAndEquipment                              |    67.1%|
|CurrentFederalTaxExpenseBenefit                                         |    66.6%|
|OperatingIncomeLoss                                                     |    65.9%|
|Liabilities                                                             |    65.8%|
|UnrecognizedTaxBenefits                                                 |    65.0%|
|PropertyPlantAndEquipmentGross                                          |    64.7%|
|Goodwill                                                               |     64.6%|
|DeferredFederalIncomeTaxExpenseBenefit                                  |    64.4%|
|CurrentStateAndLocalTaxExpenseBenefit                                   |    64.4%|
|CurrentIncomeTaxExpenseBenefit                                          |    63.8%|
|EffectiveIncomeTaxRateReconciliationAtFederalStatutoryIncomeTaxRate       |  61.7%|
|LiabilitiesCurrent                                                      |    60.0%|
|AssetsCurrent                                                          |     60.0%|
|CommonStockSharesOutstanding                                         |       59.8%|


