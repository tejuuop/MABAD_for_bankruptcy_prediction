import pandas as pd
import numpy as np
from models.mabad import MABAD

def harmonize_financials(raw_sec_data, uid):
    """
    Forces the raw SEC data to perfectly match the model's expected training format.
    """
    print("Harmonizing SEC data to match model blueprint...")
    
    # 1. Read ONLY the header (first row) of your training data to get the exact blueprint
    blueprint_df = pd.read_csv('data/numerical.csv', nrows=0)
    expected_columns = blueprint_df.columns.tolist()
    
    # 2. Create a completely blank dataframe filled with 0.0s matching the blueprint
    harmonized_df = pd.DataFrame(0.0, index=[0], columns=expected_columns)
    harmonized_df['uid'] = uid
    harmonized_df.set_index('uid', inplace=True)
    
    # 3. Map your raw SEC data into the correct columns
    for sec_tag, value in raw_sec_data.items():
        if sec_tag in harmonized_df.columns:
            harmonized_df.at[uid, sec_tag] = float(value)
        else:
            # If the model didn't see this SEC tag during training, ignore it!
            print(f"  -> Ignoring unused SEC tag: {sec_tag}")
            
    return harmonized_df

def test_new_company():
    # 1. Initialize Model
    kwargs = {
        'model_type': 'MABAD',
        'modality': 'multisource',
        'tfidf_dim': 768, 
        'device': 'cuda:0', 
        'load_tfidfs': False,
        'standardize': False # Handled by the preprocessor automatically
    }
    model = MABAD(**kwargs)
    model.load()
    model.preprocessor.load()

    new_uid = "NEW_DEMO_CORP__2025"
    
    # 2. Your messy, raw data from the SEC Downloader
    # (Notice how it might be missing hundreds of columns - that's okay now!)
    raw_sec_data = {
        'Assets': 2500000.0,
        'Liabilities': 2800000.0,
        'NetIncomeLoss': -150000.0,
        'SomeWeirdNewSECTag': 999.0 # The harmonizer will safely ignore this
    }
    
    # 3. Harmonize the numbers!
    financial_data = harmonize_financials(raw_sec_data, new_uid)
    
    # 4. Text Data (Copy-pasted from the PDF)
    text_data = pd.DataFrame([{
        'uid': new_uid,
        'item_7': "Management notes that supply chain issues have severely restricted cash flow...",
        'opinion_text': "We issue a going concern warning due to the inability to secure further credit."
    }]).set_index('uid')

    X_new = (financial_data, text_data)

    # 5. Make the Prediction
    print("\nRunning live inference...")
    anomaly_distance = model.predict_proba(X_new)[0]
    is_bankrupt = model.predict(X_new)[0]
    
    print("\n" + "="*40)
    print("      LIVE INFERENCE RESULTS")
    print("="*40)
    print(f"Anomaly Distance Score: {anomaly_distance:.4f}")
    if is_bankrupt == 1:
        print("Model Decision: 🚨 HIGH BANKRUPTCY RISK 🚨")
    else:
        print("Model Decision: ✅ HEALTHY")
    print("="*40)

if __name__ == '__main__':
    test_new_company()