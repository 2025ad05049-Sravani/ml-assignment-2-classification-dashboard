from sklearn.tree import DecisionTreeClassifier

def build_model():
    return DecisionTreeClassifier(max_depth=5, random_state=42)
