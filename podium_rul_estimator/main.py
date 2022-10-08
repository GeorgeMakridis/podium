# imports
from flask import Flask, request
from podium_classifier.Iot import Iot
from flask_restplus import Api, Resource, fields
import numpy as np
from step_funtions import preprocess_data, deepar_prediction

app = Flask(__name__)
api = Api(app)
a_timeseries = api.model('RulDeepar',{'timeseries':fields.Float('The timeseires.')})

@api.route('/rul_deepar')
class RulDeepar(Resource):

    @api.expect(a_timeseries)
    def post(sel):
        timseries = api.payload
        iot = Iot(**timseries)
        preproxessed_iot = preprocess_data(iot)
        result = deepar_prediction(preproxessed_iot)


        import json
        json_str = json.dumps(result)
        print(json_str)
        return json_str

# @app.route('/podium_rul_weibull/', methods=['POST'])
# def podium_rul_weibull():
#     content = request.get_json(silent=True)
#     print(content)
#     iot = Iot(**content)
#     labeld_iot = weibull_prediction(iot)
#
#     import json
#     json_str = json.dumps(labeld_iot)
#     print(json_str)
#     return json_str

@app.route('/')
def index():
    return 'Welcome to PoDiuM for maritime defect diagnosis'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9999)
