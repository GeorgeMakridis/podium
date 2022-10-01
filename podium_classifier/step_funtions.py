from catboost import CatBoostClassifier
from Config import model_filename


def ml_based(iot):
    from_file = CatBoostClassifier()
    model = from_file.load_model(model_filename, format="cbm")
    y_preds = model.predict(iot)
    return y_preds


def preprocess_data(iot):
    """ preprocess iot data if needed"""

    return iot



