from flask import Flask
from flask_cors import CORS
from flask_restful import Api

from resources.foo import Foo

app = Flask(__name__)
CORS(app)
api = Api(app)

# Define api routes with the associated imported class
api.add_resource(Foo, '/Foo', '/Foo/<string:id>')

if __name__ == '__main__':
  app.run(host='localhost', port=5000, debug=True, threaded=True)
