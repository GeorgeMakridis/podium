from catboost import CatBoostClassifier
import pandas as pd
from Config import FREQ, PREDICTION_LENGTH, NUMBER_OF_TEST, CONTEXT_LENGTH


def deepar_prediction(data):
    date_columns = ['TransactionDate']
    for date_column in date_columns:
        #     prices['year_'+date_column] = prices.index.year
        data['month'] = data[date_column].dt.month
        data['day_of_month'] = data[date_column].dt.day
        data['day_of_week'] = data[date_column].dt.dayofweek

    THRESHOLD_NUMBER_OF_TRANSACTIONS = 1

    df=data
    # df = data[data['TransactionDate'] < '2020-12-06 00:00:00']

    df = df[df['Debit_CreditIndicator'] == C_D]

    if len(df) > 0:

        df.set_index(['TransactionDate'], inplace=True)

        df = df.sort_index()

        start_date = df.index[0]
        end_date = df.index[-1]

        print(start_date)
        print(end_date)
        index = pd.date_range(start=start_date, end=end_date, freq=FREQ)

        # df_grouped = df.groupby(['skAcctKey', 'Master_Category'])
        df_grouped = df.groupby(['skAcctKey'])

        dataframes = []
        filenames = []

        for group in df_grouped.groups:
            if len(df_grouped.get_group(group)) > THRESHOLD_NUMBER_OF_TRANSACTIONS:
                df = df_grouped.get_group(group)
                df = df[['Amount', 'month', 'day_of_month', 'day_of_week']]
                df.fillna(0, inplace=True)
                df = df.sort_index()

                mockup_dataframe = pd.DataFrame(index=index)

                df_category = df.resample(FREQ).sum()
                mockup_dataframe = pd.merge(mockup_dataframe, df_category, how='left', left_index=True, right_index=True)

                mockup_dataframe = mockup_dataframe.resample(FREQ).sum()

                filenames.append(group)
                dataframes.append(mockup_dataframe)

        N = len(dataframes)
        T = len(dataframes[0])
        starts = []
        print('Number of Timeseries :' + str(N))
        print('Number of Samples in each ts :' + str(T))
        print('Freq : ' + FREQ)

        custom_dataset = dataframes

        custom_ds_metadata = {'num_series': N,
                              'prediction_length': PREDICTION_LENGTH,
                              'context_length': CONTEXT_LENGTH,
                              'freq': FREQ
                              }

        from gluonts.dataset import common

        test_ds = common.ListDataset([{'target': custom_dataset[i].Amount[
                                                  -NUMBER_OF_TEST - custom_ds_metadata['prediction_length'] -
                                                  custom_ds_metadata['context_length']:-NUMBER_OF_TEST -
                                                                                       custom_ds_metadata[
                                                                                           'prediction_length']],
                                        'start': custom_dataset[i].index[
                                            -NUMBER_OF_TEST - custom_ds_metadata['prediction_length'] -
                                            custom_ds_metadata['context_length']],
                                        'feat_dynamic_real': [custom_dataset[i].month[
                                                              -NUMBER_OF_TEST - custom_ds_metadata[
                                                                  'prediction_length'] - custom_ds_metadata[
                                                                  'context_length']:-NUMBER_OF_TEST],
                                                              custom_dataset[i].day_of_month[
                                                              -NUMBER_OF_TEST - custom_ds_metadata[
                                                                  'prediction_length'] - custom_ds_metadata[
                                                                  'context_length']:-NUMBER_OF_TEST],
                                                              custom_dataset[i].day_of_week[
                                                              -NUMBER_OF_TEST - custom_ds_metadata[
                                                                  'prediction_length'] - custom_ds_metadata[
                                                                  'context_length']:-NUMBER_OF_TEST]]}
                                       for i in range(N)],
                                      freq=custom_ds_metadata['freq'])

        from pathlib import Path

        # loads it back
        from gluonts.model.predictor import GluonPredictor

        predictor_deserialized = GluonPredictor.deserialize(Path("/tmp/"))

        forecasts = predictor_deserialized.predict(test_ds, num_samples=100)
        # for forecast in forecasts:
        #     print(forecast)
        fore = list(forecasts)

        print(fore)
        print(fore[0].quantile(0.95))
        return filenames[0],fore[0].quantile(0.95)


def preprocess_data(iot):
    """ preprocess iot data if needed"""

    return iot

from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator


def make_forecasts(predictor, test_data, n_sampl):
    """
    Function that automates the process of prediction and model evaluation.

    :param predictor: <gluonts.model.predictor> Predictor used for forecasting
    :param test_data: <gluonts.dataset.common.ListDataset> Dataset used for forecasting
    :param n_sampl: <int> Number of sample paths we want for evaluation
    :return: <generator>  Forecast sample paths, Timeseries used for forecasting
    """
    forecasts = []
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_data,
        predictor=predictor,
        num_samples=n_sampl
    )
    forecasts.append(list(forecast_it))
    timeseries = list(ts_it)
    return forecasts, timeseries


def generate_evaluation(timeseries, forecasts, test_ds):
    """
    Evaluate the quality of our forecasts numerically, compute aggregate performance metrics, as well as metrics per
    time series
    :param timeseries: <generator> Timeseries generator used for evaluation
    :param forecasts: <generator> Forecast sample paths to be evaluated
    :param test_ds: <gluonts.dataset.common.ListDataset> Dataset used for evaluation
    :return: Lis
    """
    for forecast in forecasts:
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        agg_metrics, item_metrics = evaluator(iter(timeseries), iter(forecast), num_series=len(test_ds))
    return agg_metrics, item_metrics

def create_test_ds(df):
    grouped = df.groupby('PRODUCT')
    starts = []
    # print(grouped.groups.keys())
    # for key in grouped.groups.keys():
    #     print(key)
    dataframes = []
    for key in grouped.groups.keys():
        df = grouped.get_group(key)
        df = df.drop(columns=['PRODUCT', 'KEY_AS_STRING']).set_index('TIME')
        df_month = df.resample('M').sum()
        df_month=df_month[df_month.index > '2007-05-31']
        df_month=df_month[df_month.index < '2019-09-30']

        df_month['DOC_COUNT'] = df_month['DOC_COUNT'] + 1
        df_month['pct'] = df_month['DOC_COUNT'].pct_change()
        df_month.dropna(inplace=True)

        dataframes.append(df_month)

    return dataframes



