import joblib
import numpy as np
import warnings
import pandas as pd
import requests

warnings.filterwarnings('ignore')


url = "http://localhost:6789/api/pipeline_schedules/4/pipeline_runs/e40a25717d114dc490126c815b472d4c"

try:
    response = requests.post(url=url)
    response.raise_for_status()  # Raise HTTP error if status is not 200-299
    result = response.json()  # Convert response to JSON
except requests.exceptions.RequestException as e:
    result = {"error": str(e)}

print(result)