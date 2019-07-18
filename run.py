"""Run experiments and create figs"""
import itertools
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import numpy as np

import dga_classifier.bigram as bigram
import dga_classifier.lstm as lstm
import dga_classifier.cnn as cnn
import dga_classifier.cnn_lstm as cnn_lstm

import dga_classifier.aloha_bigram as aloha_bigram
import dga_classifier.aloha_lstm as aloha_lstm
import dga_classifier.aloha_cnn as aloha_cnn
import dga_classifier.aloha_cnn_lstm as aloha_cnn_lstm

from scipy import interp
from sklearn.metrics import roc_curve, auc

RESULT_FILE = 'results.pkl'

def run_experiments(nfolds=10):

    options = {
        'nfolds': nfolds,
        # enable for quick functional testing
        # 'max_epoch':2
    }

    """Runs all experiments"""
    print '========== aloha_cnn_lstm =========='
    aloha_cnn_lstm_results = aloha_cnn_lstm.run(**options)

    print '========== aloha_cnn =========='
    aloha_cnn_results = aloha_cnn.run(**options)

    print '========== aloha_bigram =========='
    aloha_bigram_results = aloha_bigram.run(**options)

    print '========== aloha_lstm =========='
    aloha_lstm_results = aloha_lstm.run(**options)

    print '========== cnn_lstm =========='
    cnn_lstm_results = cnn_lstm.run(**options)

    print '========== cnn =========='
    cnn_results = cnn.run(**options)

    print '========== bigram =========='
    bigram_results = bigram.run(**options)

    print '========== lstm =========='
    lstm_results = lstm.run(**options)

    return {
        'options': options,
        'model_results': {
            'aloha_bigram': aloha_bigram_results,
            'aloha_lstm': aloha_lstm_results,
            'aloha_cnn': aloha_cnn_results,
            'aloha_cnn_lstm': aloha_cnn_lstm_results,
            'bigram': bigram_results,
            'lstm': lstm_results,
            'cnn': cnn_results,
            'cnn_lstm': cnn_lstm_results,
        }
    }

def calc_macro_roc(fpr, tpr):
    """Calcs macro ROC on log scale"""
    # Create log scale domain
    all_fpr = sorted(itertools.chain(*fpr))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(tpr)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    return all_fpr, mean_tpr / len(tpr), auc(all_fpr, mean_tpr) / len(tpr)

def calculate_metrics(model_results):
    fpr = []
    tpr = []
    for model_result in model_results:
        t_fpr, t_tpr, _ = roc_curve(model_result['y'], model_result['probs'])
        fpr.append(t_fpr)
        tpr.append(t_tpr)
    model_fpr, model_tpr, model_auc = calc_macro_roc(fpr, tpr)
    return model_fpr, model_tpr, model_auc

def create_figs(nfolds=10, force=False):
    """Create figures"""
    # Generate results if needed
    if force or (not os.path.isfile(RESULT_FILE)):
        results = run_experiments(nfolds=nfolds)
        pickle.dump(results, open(RESULT_FILE, 'w'))
    else:
        results = pickle.load(open(RESULT_FILE))

    metrics = []
    for name, model_result in sorted(results['model_results'].items()):
        if model_result is not None:
            fpr, tpr, auc = calculate_metrics(model_result)
            print "AUC: %s %.4f"%(name, auc)
            metrics.append({
                'name': name,
                'fpr': fpr,
                'tpr': tpr,
                'auc': auc,
            })

    colors = {
        'lstm': 'green',
        'cnn': 'red',
        'bigram': 'blue',
        'cnn_lstm': 'orange',
    }

    # these control which models get grouped together when ROC images get created.
    metrics_filters = {
        'all': ('lstm', 'aloha_lstm','cnn', 'aloha_cnn','cnn_lstm', 'aloha_cnn_lstm','bigram', 'aloha_bigram',),
        'lstm': ('lstm', 'aloha_lstm',),
        'cnn': ('cnn', 'aloha_cnn',),
        'cnn_lstm': ('cnn_lstm', 'aloha_cnn_lstm',),
        'bigram': ('bigram', 'aloha_bigram',),
    }

    # Save figures
    from matplotlib import pyplot
    with pyplot.style.context('bmh'):
        for plot_name, metrics_filter in metrics_filters.items():
            fig1, plt = pyplot.subplots()

            metrics_to_plot = [rec for rec in metrics if rec['name'] in metrics_filter]
            for metrics_record in sorted(metrics_to_plot, key=lambda rec: rec['auc'], reverse=True):

                linestyle = '-'
                if 'aloha' in metrics_record['name']:
                    linestyle = ':'

                color = colors[metrics_record['name'].replace('aloha_', '')]

                plt.plot(
                    metrics_record['fpr'],
                    metrics_record['tpr'],
                    label='%s (AUC = %.4f)' % (metrics_record['name'].upper(), metrics_record['auc'], ),
                    rasterized=True,
                    linestyle=linestyle,
                    color=color,

                )

            plt.set_ylim([0.0, 1.05])
            plt.set_xlabel('False Positive Rate', fontsize=16)
            plt.set_ylabel('True Positive Rate', fontsize=16)
            plt.set_title('ROC - Binary Classification (%s)'%(metrics_record['name'].upper(),), fontsize=20)
            plt.legend(loc="best", fontsize=9)
            plt.tick_params(axis='both', labelsize=16)

            # create ROC curves at various zooms at linear scale
            plt.set_xscale('linear')
            for xmax in [0.2, 0.4, 0.6, 0.8, 1.0]:
                plt.set_xlim([0.0, xmax])
                fig1.savefig('results-linear-{plot_name}-{xmax}.png'.format(xmax=xmax, plot_name=plot_name), pad_inches=0.25, bbox_inches='tight')

            # create ROC curves at various zooms at log scale
            plt.set_xscale('log')
            for xmax in [0.5, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
                plt.set_xlim([0.0, xmax])
                fig1.savefig('results-logscale-{plot_name}-{xmax}.png'.format(xmax=xmax, plot_name=plot_name), pad_inches=0.25, bbox_inches='tight')

            for xmax in [1.05]:
                plt.set_xlim([0.000001, xmax])
                fig1.savefig('results-logscale-{plot_name}-0.000001-to-{xmax}.png'.format(xmax=xmax, plot_name=plot_name), pad_inches=0.25, bbox_inches='tight')


if __name__ == "__main__":
    create_figs(nfolds=1) # Run with 1 to make it fast
