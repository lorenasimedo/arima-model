from datetime import date
from statsmodels.tsa.arima_model import ARIMA


def arima(serie_list):
    train = serie_list
    size = len(train)
    model = ARIMA(train, order=(2, 1, 1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    prediction = output[0]
    return prediction


def main():
    extract_access_frequence_list = [1, 2, 1, 2, 1, 2, 1, 2]
    last_access = date.today()
    predicition = arima(extract_access_frequence_list)
    print("Prediction: {}".format(prediction))


if __name__ == "__main__":
    # execute only if run as a script
    main()
