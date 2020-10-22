from flask_restful import Resource
from flask import request

# import all neccesary ml functions
from ml_code.model_data.yodlee_encoder import df_encoder

# create the flow for data input and running it through the ml functions/models
class Foo(Resource):
    def get(self, id):
        data = request.form['data']
        return {'data': data, 'id': id}

# hello