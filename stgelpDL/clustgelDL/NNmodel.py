import numpy as np
import tensorflow as tf

from predictor.utility import  msg2log

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 64
EPOCHS=20
DROP_PROB_0=0.2
DROP_PROB_1=0.4

def create_model(n_input:int, n_output:int, f: object=None)->tf.keras.Sequential:
    model = tf.keras.Sequential([

        tf.keras.layers.Flatten(input_shape=(n_input, 1)),
        tf.keras.layers.Dense(n_input/2, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='sigmoid'),
        tf.keras.layers.Dense(n_output, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  loss=tf.keras.losses.KLDivergence(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),tf.keras.metrics.MeanSquaredError(),
                           tf.keras.metrics.MeanAbsoluteError()])

    model.summary()
    if f is not None:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model

def create_LSTMmodel(n_input:int, n_output:int, f: object=None)->tf.keras.Sequential:

    model = tf.keras.Sequential([

        # tf.keras.layers.Flatten(input_shape=(n_input, 1)),
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(n_input, 1)),
        # tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Dense(64, activation='relu'),
        # tf.keras.layers.Dropout(0.4),
        # tf.keras.layers.Dense(32, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.Dense(16, activation='sigmoid'),
        tf.keras.layers.Dense(n_output, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  loss=tf.keras.losses.KLDivergence(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),tf.keras.metrics.MeanSquaredError(),
                           tf.keras.metrics.MeanAbsoluteError()])

    model.summary()
    if f is not None:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model

def createTfDatasets(X,y, validationRatio:float=0.1, f:object = None)->(tf.data.Dataset.from_tensor_slices,tf.data.Dataset.from_tensor_slices):

    n,m=X.shape
    n_train=int(n *(1.0 -validationRatio))

    train_dataset = tf.data.Dataset.from_tensor_slices((X[:n_train,:], y[:n_train]))
    test_dataset = tf.data.Dataset.from_tensor_slices((X[n_train:,:], y[n_train:]))

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return train_dataset, test_dataset

def createTfDatasetsLSTM(X,y, validationRatio:float=0.1, f:object = None)->(tf.data.Dataset.from_tensor_slices,tf.data.Dataset.from_tensor_slices):

    n,m=X.shape
    n_train=int(n *(1.0 -validationRatio))
    X=X.reshape((X.shape[0],X.shape[1],1))
    train_dataset = tf.data.Dataset.from_tensor_slices((X[:n_train,:,:], y[:n_train]))
    test_dataset = tf.data.Dataset.from_tensor_slices((X[n_train:,:,:], y[n_train:]))

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return train_dataset, test_dataset

def fitModel(model:tf.keras.Sequential, train_dataset:tf.data.Dataset.from_tensor_slices,
             test_dataset:tf.data.Dataset.from_tensor_slices, n_epochs:int=4,f: object=None)->(object,list):
    history = model.fit(train_dataset, epochs=n_epochs,verbose=1,validation_data=test_dataset)
    msg2log(fitModel.__name__, "\n\nTraining loss values and menrics valuese\n{}".format(history.history), f)

    eval_history= model.evaluate(test_dataset,verbose=1)
    msg2log(fitModel.__name__, "\n\nEvaluation loss values\n{}".format(eval_history), f)

    return history, eval_history

def predictModel(model:tf.keras.Sequential, predict_dataset: np.array, predict_labels:list,f:object=None)->np.array:
    # predict_dataset=predict_dataset.reshape((predict_dataset.shape[0],predict_dataset.shape[1],1))

    y_pred = model.predict(predict_dataset)
    n,m =y_pred.shape
    msg2log(predictModel.__name__,"{}".format(y_pred),f)
    state_pred = [np.argmax(y_pred[1, :]) for i in range(n)]

    msg="Prediction\nTimestamp label       State"
    msg2log(None, msg,f)

    for i in range(n):
        msg="{}  {}".format(predict_labels[i], state_pred[i])
        msg2log(None, msg, f)

    return state_pred


if __name__=="__main__":
    pass

    # # DATA_URL='https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    # #
    # # path =tf.keras.utils.get_file('mnist.npz',DATA_URL)
    # # with np.load(path) as data:
    # #     train_examples =data['x_train']
    # #     train_labels = data['y_train']
    # #     test_examples = data['x_test']
    # #     test_labels = data['y_test']
    #
    # pass
    # train_examples =np.random.rand(100,288)
    # train_labels =np.random.rand(100)
    # test_examples =np.random.rand(20,288)
    # test_labels =np.random.rand(20)
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    # test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
    #
    # BATCH_SIZE = 32
    # SHUFFLE_BUFFER_SIZE = 64
    #
    # train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    # test_dataset = test_dataset.batch(BATCH_SIZE)
    #
    # model = tf.keras.Sequential([
    #     # tf.keras.layers.Flatten(input_shape=(28, 28)),
    #     tf.keras.layers.Flatten(input_shape=(288,1)),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dropout(0.3),
    #     tf.keras.layers.Dense(10, activation='softmax')
    # ])
    #
    # model.compile(optimizer=tf.keras.optimizers.RMSprop(),
    #                 loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #                 metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    #
    # model.fit(train_dataset, epochs=4)
    #
    # model.evaluate(test_dataset)
    #
    # pass
    #
    # model.summary()
    #
    # tf.keras.utils.plot_model(
    #     model, to_file='model.png', show_shapes=False, show_layer_names=True,
    #     rankdir='TB', expand_nested=False, dpi=96
    # )
    #
    # pass
    #
    # predict_examples =np.random.rand(2,288)
    # predict_dataset = train_dataset.batch(BATCH_SIZE)
    # y_pred=model.predict(predict_dataset)
    # c=np.argmax(y_pred[0,:])
    # pass



