from flask import Flask, request,jsonify
from flask_pymongo import PyMongo
import numpy as np
import os
import recognition
import datetime
import base64

clValues = {0:0.02,1:0.02,2:0.02,3:0.02,4:0.01}

app = Flask(__name__)
if(os.environ['MODE']=='production'):
    app.config['MONGO_URI'] = os.environ['MONGODB_URI'];
    app.config['DEBUG'] = False
else:
    app.config['DEBUG'] = True
    app.config['MONGO_DBNAME'] = 'bottles'

mongo = PyMongo(app)


@app.route('/predict',methods=['POST'])
def new_record():
    data = request.json.get('data').encode()
    dataArray = np.fromstring(bytes(base64.decodestring(data)), dtype=np.uint8)
    dataArray.resize(1,100,200,3)
    prob,cl= recognition.predict(dataArray)
    print(prob,cl)
    if(prob<0.95):
        photo = mongo.db.images.insert({'data':data})
        return jsonify({'valid':'False'})
    else:
        record = mongo.db.record.insert({'id':request.json.get('id'),'cl':str(cl),'value':clValues[cl],'prob':str(prob)})
        return jsonify({'value':str(clValues[cl]),'valid':'True'})

@app.errorhandler(404)
def page_not_found(error):
	return 'Page not found :('

@app.errorhandler(500)
def internal_server_error(error):
    print(error)
    return 'Something went wrongish '

@app.errorhandler(Exception)
def unhandled_exception(e):
    print(e)
    return 'Something went wrongish '

if __name__ == '__main__':
    app.run()
