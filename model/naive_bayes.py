from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ])
