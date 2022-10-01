# imports
from flask import Flask, request
from podium_classifier.Iot import Iot
from step_funtions import preprocess_data, ml_based

app = Flask(__name__)


def classify(iot):
    # iot.label = UNKNOWN
    # STEP 1 (pre-process data)
    iot = preprocess_data(iot)
    # STEP 2 (ML method)
    iot.label = ml_based(iot)
    print(iot.label)

    return iot


@app.route('/podium_classifier/', methods=['POST'])
def podium_classifier():
    content = request.get_json(silent=True)
    print(content)
    iot = Iot(**content)
    labeld_iot = classify(iot)

    import json
    json_str = json.dumps(labeld_iot)
    print(json_str)
    return json_str

@app.route('/')
def index():
    return 'Welcome to PoDiuM for maritime defect diagnosis'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9999)
