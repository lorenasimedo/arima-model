from datetime import date
from statsmodels.tsa.arima_model import ARIMA


def arima(serie_list):
    train = serie_list
    size = len(train)
    model = ARIMA(train, order=(0, 0, 0))
    model_fit = model.fit(disp=False)
    output = model_fit.forecast()
    prediction = output[0]
    return prediction


def main():
    extract_access_frequence_list = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    last_access = date.today()
    prediction = arima(extract_access_frequence_list)
    print("Prediction: {}".format(prediction))


if __name__ == "__main__":
    # execute only if run as a script
    main()
