"""Train and test LSTM classifier"""
import dga_classifier.data as data
import numpy as np
from keras.preprocessing import sequence
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import sklearn
from sklearn.model_selection import train_test_split


def build_model(max_features, maxlen, num_targets=1):
    """Build LSTM model"""

    text_input = Input(shape = (maxlen,), name='text_input')
    x = Embedding(input_dim=max_features, input_length=maxlen, output_dim=128)(text_input)
    x = LSTM(128)(x)
    drop = Dropout(0.5)(x)

    outputs = []
    for name in data.get_labels():
        dense = Dense(1)(drop)
        out = Activation("sigmoid", name=name)(dense)
        outputs.append(out)
    model = Model(inputs=text_input, outputs=outputs)
    model.compile(
        loss=data.get_losses(),
        loss_weights=data.get_loss_weights(),
        optimizer='rmsprop'
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

