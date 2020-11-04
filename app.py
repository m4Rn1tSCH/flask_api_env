from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Api

from resources.foo import Foo

# constructor for app
app = Flask(__name__)
CORS(app)
api = Api(app)

"""RUN EITHER AS API WITH ADDED RESOURCES OR RUN WITH ROUTES"""
"""
Define api routes with the associated imported class
if more resources are added; unique decorator required
"""
api.add_resource(Foo, '/Foo', '/Foo/<string:id>')
# api.add_resource(Foo, '/Foo', '/Foo/json_api_test')

"""
All objects are received by this end of the app
This is directly routed and running in the app
All objects will be converted to a python dict {"key":"value"}
"""
# GET requests will be blocked
@app.route('/json_example', methods=['POST']) 
def json_example():

    '''
    This functions returns a JSON object that is saved elsewhere like a py dictionary and can either use POST OR GET method.
    
    '''
    # show an incoming JSON object that is structured like a dictionary
    req_data = request.get_json()

    merchant = req_data['merchant']
    counterparty = req_data['amount']
    # two keys are needed because of the nested object
    python_version = req_data['version_info']['python']
    # an index is needed because of the array
    timestamp = req_data['examples'][0]
    boolean_test = req_data['boolean_test']

    # no f literals to avoid compatibility problems with py <3.6
    return '''
           The merchant value is: {}
           The counterparty value is: {}
           The Python version is: {}
           The timestamp is: {}
           The boolean value is: {}'''.format(language, framework, python_version, example, boolean_test)

@app.route('/json_test', methods=['POST']) # GET requests will be blocked
def json_test():

    # show an incoming JSON object that is structured like a py dictionary
    req_data = request.get_json()
    key = req_data['key']
    key2 = req_data['key2']
    key3 = req_data['test']

    return '''
           The first value is: {}
           The second test value is: {}
           The third value is: {}
           '''.format(key, key2, key3)

# uses post by standard
@app.route('/query_example')
def query_example():

    '''
    The query passes data to the flask app, string + URL passes data without taking action
    Query also works with GET method 
    '''
    # is looking for that arg in the URL and retrieves the value
    # .get avoids 400 error when the arg is no in the URL and keeps the system running
    # returns None if key does not exist
    arg1 = request.args.get('arg1')
    arg2 = request.args.get('arg2')
    arg3 = request.args.get('arg3')
    # http://127.0.0.1:5000/query_example?
    # framework=test&language=Python&arg2=HELLO&arg3=TESTING
    # URL passed will return values of keys that are found
    return '''
            <h1>The arg1 value is: {}</h1>
            <h1>The arg2 value is: {}</h1>
            <h1>The arg3 value is: {}</h1>
            '''.format(arg1, arg2, arg3)

@app.route('/form_example', methods=['GET', 'POST']) #allow both GET and POST requests
def form_example():

    '''
    GET: form will be generated
    POST: process incoming data
    Performs a POST request to the same route that generated the form
    Use as a form in the web browser or pass data to it in "body" and as "form-data"
    will pull keys from passed form data and display corresponding value
    Keys are read from the name attributes on form input
    '''
     # this block is only entered when the form is submitted
    if request.method == 'POST':
        # look for key and return None when key not found
        merchant = request.form.get('merchant')
        # key referenced directly but returns 400 error when not found
        amount = request.form.get('amount')
        
        return '''
                <h1>The merchant value is: {}</h1>
                <h1>The amount value is: {}</h1>
                '''.format(merchant, amount)

    # This decorator displays a form with two two fields to type data
    return '''<form method="GET">
                  merchant: <input type="text" name="merchant"><br>
                  amount: <input type="text" name="amount"><br>
                  <input type="submit" value="Submit"><br>
              </form>'''

@app.route('/transfer_data', methods=['POST'])
def transfer_data():

    '''
    This function converts a dictionary into a JSON object and transfers it
    The response objects is the visible as JSON and can contain py dict data
    '''
    # test_data passed; converted to a dict
    # a = ['test1', 'test2', 'test3', 'test4', 'test5']
    # b  = ['value1', 'value2', 'value3' , 'value3', 'value4', 'value5']
    # d = dict([a, b] for a, b in zip(a, b))

    response = jsonify({"test_key1" : "test_value1",
                        "test_key2" : "test_value2"},
                       {"test2" : "value2"})
    response.status_code = 200
    return response




if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True, threaded=True)
