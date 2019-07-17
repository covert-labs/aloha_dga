"""Train and test CNN classifier"""
import dga_classifier.data as data
import numpy as np
from keras.preprocessing import sequence
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv1D, Input, Dense, concatenate
from keras.optimizers import SGD
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D


def build_model(max_features, maxlen):
    '''
    Derived CNN model from Keegan Hines' Snowman
        https://github.com/keeganhines/snowman/
    '''
    text_input = Input(shape = (maxlen,), name='text_input')
    x = Embedding(input_dim=max_features, input_length=maxlen, output_dim=128)(text_input)

    conv_a = Conv1D(15,2, activation='relu')(x)
    conv_b = Conv1D(15,3, activation='relu')(x)
    conv_c = Conv1D(15,4, activation='relu')(x)
    conv_d = Conv1D(15,5, activation='relu')(x)
    conv_e = Conv1D(15,6, activation='relu')(x)

    pool_a = GlobalMaxPooling1D()(conv_a)
    pool_b = GlobalMaxPooling1D()(conv_b)
    pool_c = GlobalMaxPooling1D()(conv_c)
    pool_d = GlobalMaxPooling1D()(conv_d)
    pool_e = GlobalMaxPooling1D()(conv_e)

    flattened = concatenate(
        [pool_a, pool_b, pool_c, pool_d, pool_e])

    drop = Dropout(.2)(flattened)
    dense = Dense(1)(drop)
    out = Activation("sigmoid")(dense)
    model = Model(inputs=text_input, outputs=out)

    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )

    return model



def run(max_epoch=25, nfolds=10, batch_size=128):
    """Run train/test on logistic regression model"""
    indata = data.get_data()

    # Extract data and labels
    X = [x[1] for x in indata]
    labels = [x[0] for x in indata]

    # Generate a dictionary of valid characters
    valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}

    max_features = len(valid_chars) + 1
    maxlen = np.max([len(x) for x in X])

    # Convert characters to int and pad
    X = [[valid_chars[y] for y in x] for x in X]
    X = sequence.pad_sequences(X, maxlen=maxlen)

    # Convert labels to 0-1
    y = [0 if x == 'benign' else 1 for x in labels]

    final_data = []

    for fold in range(nfolds):
        print "fold %u/%u" % (fold+1, nfolds)
        X_train, X_test, y_train, y_test, _, label_test = train_test_split(X, y, labels, 
                                                                           test_size=0.2)

        print 'Build model...'
        model = build_model(max_features, maxlen)

        print "Train..."
        X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.05)
        best_iter = -1
        best_auc = 0.0
        out_data = {}

        for ep in range(max_epoch):
            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1)

            t_probs = model.predict(X_holdout)
            t_auc = sklearn.metrics.roc_auc_score(y_holdout, t_probs)

            print 'Epoch %d: auc = %f (best=%f)' % (ep, t_auc, best_auc)

            if t_auc > best_auc:
                best_auc = t_auc
                best_iter = ep

                probs = model.predict(X_test)

                out_data = {'y':y_test, 'labels': label_test, 'probs':probs, 'epochs': ep,
                            'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, probs > .5)}

                print sklearn.metrics.confusion_matrix(y_test, probs > .5)
            else:
                # No longer improving...break and calc statistics
                if (ep-best_iter) > 2:
                    break

        final_data.append(out_data)

    return final_data
