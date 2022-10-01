import numpy as np
import os
from gluonts.dataset.common import ListDataset

from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.seasonal_naive import SeasonalNaiveEstimator
from gluonts.distribution.piecewise_linear import PiecewiseLinearOutput
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.wavenet import WaveNetEstimator

import pandas as pd

MIN_DATE = '2007-05-31'
MAX_DATE = '2019-09-30'


def create_dataframes_array(data):
    dataframes = []
    filenames = []
    for file in os.listdir(datasets):
        if file.endswith('.csv'):
            df = pd.read_csv(datasets / file, parse_dates=['key_as_string', 'priceStringDate'],
                             index_col='priceStringDate')
            df = df.drop(columns=['product', 'Unnamed: 0', 'key', 'key_as_string'])
            # df=df[df.index.year>2000]
            df_month = df.resample('M').sum()
            df_month = df_month[df_month.index > MIN_DATE]
            df_month = df_month[df_month.index < MAX_DATE]

            df_month['doc_count'] = df_month['doc_count'] + 1
            # df_month['exp'] = np.exp(df_month['doc_count'])
            df_month['pct'] = df_month['doc_count'].pct_change()
            df_month['log_ret'] = np.log(df_month['pct'] + 1)
            df_month.dropna(inplace=True)

            filenames.append(file)
            dataframes.append(df_month)
            return dataframes


dataframes = create_dataframes_array()

N = len(dataframes)
T = len(dataframes[0])
prediction_length = 4
freq = "1M"
starts = []
custom_dataset = np.zeros(shape=(N, T))

for i, df in enumerate(dataframes):
    custom_dataset[i, :] = df['pct'].to_numpy()
    starts.append(df.index[0])

custom_ds_metadata = {'num_series': 29,
                      'prediction_length': prediction_length,
                      'context_length': 12,
                      'freq': freq
                      }

def run_tests():
    NUMBER_OF_TEST = 1

    for k in range(NUMBER_OF_TEST):
        train_ds = ListDataset([{'target': x, 'start': starts[i]}
                                for i, x in enumerate(custom_dataset[:, k:-NUMBER_OF_TEST + k - prediction_length])],
                               freq=custom_ds_metadata['freq'])

        # test dataset: use the whole dataset, add "target" and "start" fields
        test_ds = ListDataset([{'target': x, 'start': starts[i]}
                               for i, x in enumerate(custom_dataset[:, k + 1:-NUMBER_OF_TEST + k + 1 - prediction_length])],
                              freq=custom_ds_metadata['freq'])

        estimator_dare = DeepAREstimator(
            prediction_length=custom_ds_metadata['prediction_length'],
            context_length=custom_ds_metadata['context_length'],
            freq=custom_ds_metadata['freq'],
            trainer=Trainer(epochs=100,
                            ),
            distr_output=PiecewiseLinearOutput(8)
        )
        estimator_sffe = SimpleFeedForwardEstimator(
            num_hidden_dimensions=[10],
            prediction_length=custom_ds_metadata['prediction_length'],
            context_length=custom_ds_metadata['context_length'],
            freq=custom_ds_metadata['freq'],
            trainer=Trainer(ctx="cpu", epochs=100,
                            num_batches_per_epoch=100
                            )
        )
        estimator_factor = DeepFactorEstimator(
            prediction_length=custom_ds_metadata['prediction_length'],
            context_length=custom_ds_metadata['context_length'],
            freq=custom_ds_metadata['freq'],
            num_hidden_local=8,
            trainer=Trainer(ctx="cpu",
                            epochs=100,
                            )
        )
        estimator_seasonal_naive = SeasonalNaiveEstimator(
            prediction_length=custom_ds_metadata['prediction_length'],
            freq=custom_ds_metadata['freq'],
        )
        estimator_Wave = WaveNetEstimator(
            freq=custom_ds_metadata['freq'],
            prediction_length=custom_ds_metadata['prediction_length'],
            trainer=Trainer(epochs=100)
        )

        predictor_dare = estimator_dare.train(train_ds)
        predictor_sffe = estimator_sffe.train(train_ds)
        predictor_factor = estimator_factor.train(train_ds)
        predictor_seasonal = estimator_seasonal_naive.train(train_ds)
        predictor_Wave = estimator_Wave.train(train_ds)

        predictors = [predictor_dare, predictor_sffe, predictor_factor, predictor_seasonal, predictor_Wave]
        predictors = {"Deep AR Estimator": predictor_dare, "Simple Feed Forward Estimator": predictor_sffe,
                      "Deep Factor Estimator": predictor_factor, "Seasonal Naive Estimator": predictor_seasonal,
                      "WaveNet Estimator": predictor_Wave}


    def make_forecasts(predictors, test_data, n_sampl):
        """Takes a list of predictors,gluonTS test data and number of samples
        and returns forecasts for each of them"""
        forecasts = []
        tss = []
        for predictor_name, predictor in predictors.items():
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=test_ds,
                predictor=predictor,
                num_samples=n_sampl
            )
            forecasts.append((list(forecast_it), predictor_name))
            tss = list(ts_it)
        return forecasts, tss


    forecasts, tss = make_forecasts(predictors, test_ds, 100)
