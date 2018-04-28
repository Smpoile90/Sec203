import numpy
import pandas
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

train_path = 'train.csv'  # ~\Documents\SEC203DAT\Train1
test_path = 'test.csv'  # ~\Documents\SEC203DATTest1

OUTPUT_PATH= 'myModel.bin'

def getData():
    # train data
    dataframe = pandas.read_csv(train_path)
    dataframe.pop('name')
    dataset = dataframe.values
    train_x = dataset[:, 1:7]
    train_y = dataset[:, 0]
    # test data
    dataframe = pandas.read_csv(test_path)
    dataframe.pop('name')
    dataset = dataframe.values
    test_x = dataset[:, 1:7]
    test_y = dataset[:, 0]

    return train_x, train_y, test_x, test_y

    # baseline model

def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(24, input_dim=6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(12, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model,**kwargs):
    model.fit(kwargs['x'], kwargs['y'], epochs=20, batch_size=50, verbose=1)
    return model

##Used for testing the success of different model over various runs
def evaluate(**kwargs):
    seed = 7
    numpy.random.seed(seed)
    # evaluate model with standardized dataset
    estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(estimator, kwargs['x'], kwargs['y'], cv=kfold)
    print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

def main():
    train_x, train_y, test_x, test_y = getData()
    model = train_model(create_baseline(),x=train_x,y=train_y)

    ##Get the config and save it to the OUTPUTH PATH
    config = model.get_config()
    with open(OUTPUT_PATH, 'wb') as file:
         pickle.dump(config, file)


if __name__ == "__main__":
    main()


#
# model = train_model(create_baseline())
# config = model.get_config()
#
# with open('myModel', 'wb') as file:
#     pickle.dump(config, file)
#
#
# with open('myModel', 'rb') as file:
#      config =pickle.load(file)
#
# model = Sequential.from_config(config)
# print(model.summary())
#
#
# model = train_model(create_baseline())
#
# data = pandas.DataFrame(tes, index=[6])
# data = data.values
# prediction, probability = model.predict_classes(data), model.predict(data)
# probability = probability * 100
# print(prediction)
# print(probability)



