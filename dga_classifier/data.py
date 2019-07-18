"""Generates data for train/test algorithms"""
from datetime import datetime
from StringIO import StringIO
from urllib import urlopen
from zipfile import ZipFile

import cPickle as pickle
import os
import random
import tldextract
import numpy as np

from dga_classifier.dga_generators import banjori, corebot, cryptolocker, \
    dircrypt, kraken, lockyv2, pykspa, qakbot, ramdo, ramnit, simda, \
    matsnu, suppobox, gozi

# Location of Alexa 1M
ALEXA_1M = 'http://s3.amazonaws.com/alexa-static/top-1m.csv.zip'

# Our ourput file containg all the training data
DATA_FILE = 'traindata.pkl'

def get_alexa(num, address=ALEXA_1M, filename='top-1m.csv'):
    """Grabs Alexa 1M"""
    url = urlopen(address)
    zipfile = ZipFile(StringIO(url.read()))
    return [tldextract.extract(x.split(',')[1]).domain for x in \
            zipfile.read(filename).split()[:num]]

def gen_malicious(num_per_dga=10000):
    """Generates num_per_dga of each DGA"""
    domains = []
    labels = []

    # We use some arbitrary seeds to create domains with banjori
    banjori_seeds = ['somestring', 'firetruck', 'bulldozer', 'airplane', 'racecar',
                     'apartment', 'laptop', 'laptopcomp', 'malwareisbad', 'crazytrain',
                     'thepolice', 'fivemonkeys', 'hockey', 'football', 'baseball',
                     'basketball', 'trackandfield', 'fieldhockey', 'softball', 'redferrari',
                     'blackcheverolet', 'yellowelcamino', 'blueporsche', 'redfordf150',
                     'purplebmw330i', 'subarulegacy', 'hondacivic', 'toyotaprius',
                     'sidewalk', 'pavement', 'stopsign', 'trafficlight', 'turnlane',
                     'passinglane', 'trafficjam', 'airport', 'runway', 'baggageclaim',
                     'passengerjet', 'delta1008', 'american765', 'united8765', 'southwest3456',
                     'albuquerque', 'sanfrancisco', 'sandiego', 'losangeles', 'newyork',
                     'atlanta', 'portland', 'seattle', 'washingtondc']

    segs_size = max(1, num_per_dga/len(banjori_seeds))
    for banjori_seed in banjori_seeds:
        domains += banjori.generate_domains(segs_size, banjori_seed)
        labels += ['banjori']*segs_size

    domains += corebot.generate_domains(num_per_dga)
    labels += ['corebot']*num_per_dga

    # Create different length domains using cryptolocker
    crypto_lengths = range(8, 32)
    segs_size = max(1, num_per_dga/len(crypto_lengths))
    for crypto_length in crypto_lengths:
        domains += cryptolocker.generate_domains(segs_size,
                                                 seed_num=random.randint(1, 1000000),
                                                 length=crypto_length)
        labels += ['cryptolocker']*segs_size

    domains += dircrypt.generate_domains(num_per_dga)
    labels += ['dircrypt']*num_per_dga

    # generate kraken and divide between configs
    kraken_to_gen = max(1, num_per_dga/2)
    domains += kraken.generate_domains(kraken_to_gen, datetime(2016, 1, 1), 'a', 3)
    labels += ['kraken']*kraken_to_gen
    domains += kraken.generate_domains(kraken_to_gen, datetime(2016, 1, 1), 'b', 3)
    labels += ['kraken']*kraken_to_gen

    # generate locky and divide between configs
    locky_gen = max(1, num_per_dga/11)
    for i in range(1, 12):
        domains += lockyv2.generate_domains(locky_gen, config=i)
        labels += ['locky']*locky_gen

    # Generate pyskpa domains
    domains += pykspa.generate_domains(num_per_dga, datetime(2016, 1, 1))
    labels += ['pykspa']*num_per_dga

    # Generate qakbot
    domains += qakbot.generate_domains(num_per_dga, tlds=[])
    labels += ['qakbot']*num_per_dga

    # ramdo divided over different lengths
    ramdo_lengths = range(8, 32)
    segs_size = max(1, num_per_dga/len(ramdo_lengths))
    for rammdo_length in ramdo_lengths:
        domains += ramdo.generate_domains(segs_size,
                                          seed_num=random.randint(1, 1000000),
                                          length=rammdo_length)
        labels += ['ramdo']*segs_size

    # ramnit
    domains += ramnit.generate_domains(num_per_dga, 0x123abc12)
    labels += ['ramnit']*num_per_dga

    # simda
    simda_lengths = range(8, 32)
    segs_size = max(1, num_per_dga/len(simda_lengths))
    for simda_length in range(len(simda_lengths)):
        domains += simda.generate_domains(segs_size,
                                          length=simda_length,
                                          tld=None,
                                          base=random.randint(2, 2**32))
        labels += ['simda']*segs_size

    # matsnu
    domains += matsnu.generate_domains(num_per_dga, include_tld=False)
    labels += ['matsnu']*num_per_dga

    # suppobox
    domains += suppobox.generate_domains(num_per_dga, include_tld=False)
    labels += ['suppobox']*num_per_dga

    # gozi
    domains += gozi.generate_domains(num_per_dga, include_tld=False)
    labels += ['gozi']*num_per_dga

    return domains, labels

def gen_data(force=False):
    """Grab all data for train/test and save

    force:If true overwrite, else skip if file
          already exists
    """
    if force or (not os.path.isfile(DATA_FILE)):
        domains, labels = gen_malicious(10000)

        # Get equal number of benign/malicious
        domains += get_alexa(len(domains))
        labels += ['benign']*len(domains)

        pickle.dump(zip(labels, domains), open(DATA_FILE, 'w'))

def get_data(force=False):
    """Returns data and labels"""
    gen_data(force)

    return pickle.load(open(DATA_FILE))

def get_malware_labels(labels):
    malware_labels = list(set(labels))
    malware_labels.remove('benign')
    malware_labels.sort()
    return malware_labels


def expand_labels(labels):
    '''
    This function takes the labels as returned from get_data()
    and it converts them into a list of lists of 0/1 labels per
    'benign' and per each malware family
    '''

    # Convert labels to 0-1
    y = [0 if label == 'benign' else 1 for label in labels]
    all_Ys = [y]
    for malw_label in get_malware_labels(labels):
        all_Ys.append([1 if label == malw_label else 0 for label in labels])
    return all_Ys

def get_labels():
    return [
        'main',
        'corebot',
        'dircrypt',
        'kraken',
        'pykspa',
        'qakbot',
        'ramnit',
        'locky',
        'banjori',
        'cryptolocker',
        'ramdo',
        'simda',
        'matsnu',
        'suppobox',
        'gozi',
    ]

def get_losses():
    return {
        'main': 'binary_crossentropy',
        'corebot': 'binary_crossentropy',
        'dircrypt': 'binary_crossentropy',
        'kraken': 'binary_crossentropy',
        'pykspa': 'binary_crossentropy',
        'qakbot': 'binary_crossentropy',
        'ramnit': 'binary_crossentropy',
        'locky': 'binary_crossentropy',
        'banjori': 'binary_crossentropy',
        'cryptolocker': 'binary_crossentropy',
        'ramdo': 'binary_crossentropy',
        'simda': 'binary_crossentropy',
        'matsnu': 'binary_crossentropy',
        'suppobox': 'binary_crossentropy',
        'gozi': 'binary_crossentropy',
    }

def get_loss_weights():
    return {
        'main': 1.0,
        'corebot': 0.1,
        'dircrypt': 0.1,
        'kraken': 0.1,
        'pykspa': 0.1,
        'qakbot': 0.1,
        'ramnit': 0.1,
        'locky': 0.1,
        'banjori': 0.1,
        'cryptolocker': 0.1,
        'ramdo': 0.1,
        'simda': 0.1,
        'matsnu': 0.1,
        'suppobox': 0.1,
        'gozi': 0.1,
    }

def y_list_to_dict(all_Ys):
    return dict([(label, np.array(y)) for label, y in zip(get_labels(), all_Ys)])