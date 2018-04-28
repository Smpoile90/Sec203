from flask import Flask,jsonify
from keras import Sequential
import scrape
import myMongo
import pickle
import pandas

app = Flask(__name__)

with open('myModel.bin', 'rb') as file:
    config = pickle.load(file)

model = Sequential.from_config(config)


##Evaluate is used on newly scraped data
def evaluate(data):
    data = pandas.DataFrame(data, index=[6])
    data = data.values
    prediction, probability = model.predict_classes(data), model.predict(data)
    probability = float(probability[0][0]) * 100
    prediction = int(prediction[0][0])
    return prediction, probability


@app.route('/<string:name>')
def evaluateName(name):
    ##First query the db
    try:
        x = myMongo.queryUname(name)
    except: x = None
    ##If db returns nothing
    if x is None:
        data = scrape.getName(name)
        if data is None:
            data = {'name':'Does not exist'}
            return jsonify(data)
        data.pop('name')
        image = data.pop('image')
        botvalue, probability = evaluate(data)
        data['bot'], data['probability'], data['name'], data['image'] = botvalue, probability, name,image
        try:
            myMongo.insertOrUpdate(data)
        except:
            print('DB conn failed')

        data.pop('_id')

    else:
        data = x

    return jsonify(data)

@app.route('/listBots')
def listBots():
    x = myMongo.listBots()
    print(x)
    return jsonify(x)


app.run(debug=True)


