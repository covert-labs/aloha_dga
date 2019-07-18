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
from keras.layers.recurrent import LSTM


def build_model(max_features, maxlen, num_targets=1):
    '''
    [Deep Learning For Realtime Malware Detection (ShmooCon 2018)](https://www.youtube.com/watch?v=99hniQYB6VM)'s 
    LSTM + CNN (see 13:17 for architecture) by Domenic Puzio and Kate Highnam

    AND
    
    Derived CNN model from Keegan Hines' Snowman https://github.com/keeganhines/snowman/
    '''
    text_input = Input(shape = (maxlen,), name='text_input')
    x = Embedding(input_dim=max_features, input_length=maxlen, output_dim=128)(text_input)

    lstm = LSTM(128)(x)
    lstm = Dropout(0.5)(lstm)
    lstm = Dense(1)(lstm)

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
        [pool_a, pool_b, pool_c, pool_d, pool_e, lstm])

    drop = Dropout(.2)(flattened)

    outputs = []
    for name in data.get_labels():
        dense = Dense(1)(drop)
        out = Activation("sigmoid", name=name)(dense)
        outputs.append(out)
    model = Model(inputs=text_input, outputs=outputs)
    model.compile(
        loss=data.get_losses(),
        loss_weights=data.get_loss_weights(),
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

    malware_labels = data.get_malware_labels(labels)
    all_Ys = data.expand_labels(labels)

    final_data = []

    for fold in range(nfolds):
        print "fold %u/%u" % (fold+1, nfolds)
        train_test = train_test_split(X, labels, *all_Ys, test_size=0.2, stratify=labels)
        X_train, X_test, label_train, label_test, y_train, y_test = train_test[:6]
        dga_training_test = train_test[6:]

        all_Y_train = [y_train]
        for idx in range(0, len(dga_training_test), 2):
            all_Y_train.append(dga_training_test[idx])

        print 'Build model...'
        model = build_model(max_features, maxlen, num_targets=len(malware_labels) + 1)

        print "Train..."
        train_test = train_test_split(X_train, *all_Y_train, test_size=0.05, stratify=label_train)
        X_train, X_holdout, y_train, y_holdout = train_test[:4]
        dga_training_test = train_test[4:]
        all_Y_train = [y_train]
        for idx in range(0, len(dga_training_test), 2):
            all_Y_train.append(dga_training_test[idx])

        best_iter = -1
        best_auc = 0.0
        out_data = {}

        for ep in range(max_epoch):
            model.fit(X_train, data.y_list_to_dict(all_Y_train), batch_size=batch_size, epochs=1)

            t_probs = model.predict(X_holdout)[0]
            t_auc = sklearn.metrics.roc_auc_score(y_holdout, t_probs)

            print 'Epoch %d: auc = %f (best=%f)' % (ep, t_auc, best_auc)

            if t_auc > best_auc:
                best_auc = t_auc
                best_iter = ep

                probs = model.predict(X_test)[0]

                out_data = {'y':y_test, 'labels': label_test, 'probs':probs, 'epochs': ep,
                            'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, probs > .5)}

                print sklearn.metrics.confusion_matrix(y_test, probs > .5)
            else:
                # No longer improving...break and calc statistics
                if (ep-best_iter) > 2:
                    break

        final_data.append(out_data)

    return final_data
