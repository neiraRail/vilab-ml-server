from flask import Flask, jsonify, request
from flask_pymongo import PyMongo
#from flask_cors import CORS
from os import environ as env
from models.model import Model

mongohost = env.get("MONGO_HOST", "localhost")
app = Flask(__name__)
#CORS(app)
app.config["MONGO_URI"] = f"mongodb://{mongohost}:27017/mydatabase"
mongo = PyMongo(app)

@app.route('/model', methods=['GET'])
def get_models():
    nodos = mongo.db.models.find({}, {'_id': 0})
    return jsonify(list(nodos))

def validate_data(data):
    try:
        data = list(data)
        return True
    except:
        return False

@app.route('/model/predict/<id>', methods=['POST'])
def predict(id):
    data = request.json
    if not validate_data(data):
        return "data is not valid", 400

    # load model
    # predict using model
    return {
        "count": len(list(data))
    }

@app.route('/model', methods=['GET'])
def create_model():
    data = request.json
    try:
        job = Model(data['path'], data['tipo'])
        job_id = mongo.db.models.insert_one(vars(job)).inserted_id
    except KeyError as e:
        return {'error': f'Falta el campo {e}'}, 400
    except BaseException as e:
        return {'error': f'Error desconocido: {e}'}, 500
    
    return {'message': 'Modelo creado exitosamente', 'job_id': str(job_id)}, 201


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8082)