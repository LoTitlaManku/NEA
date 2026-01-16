
import os
import pandas as pd
import numpy as np

ledger_folder = "saved_predictions"
for filename in os.listdir(ledger_folder):
    try:
        ticker = filename.split("_")[0]
        filepath = os.path.join(ledger_folder, filename)
        ledger = pd.read_csv(filepath)

        ledger['Target_Date'] = pd.to_datetime(ledger['Target_Date'], format='ISO8601')
        ledger['Target_Date'] = ledger['Target_Date'].dt.strftime('%Y-%m-%d %H:%M')
        ledger[["Actual_Price", "Is_Correct"]] = ledger[["Actual_Price", "Is_Correct"]].replace("Invalid date", np.nan)

        ledger.to_csv(filepath, index=False)


    except Exception as e:
        print(f"Error - {type(e).__name__} {e}")


