# Machine Learning Pipeline

An End-to-End Machine Learning pipeline deployed as a REST API.  \
API developed with **FastAPI**

### Data 
Medical Insurance Payout  \
https://www.kaggle.com/datasets/harshsingh2209/medical-insurance-payout

### Process:
* POST training data on `train/`
* Data gets preprocessed and split
* Model pipeline is created
* Random Forest Regressor model gets trained
* Evaluated
* POST test sample on `test/`

### Setup:
1. Clone the repository
2. Create virtualenv and run `pip install -r requirements.txt`
3. To launch server, run `uvicorn app:app --reload`
4. Test with Python requests or Postman