from visualization.visualize import *
import os
import mongodb
import json


def test_history():
    # csv_path = '/home/thulx/Master/BI/jena_climate_2009_2016.csv'
    # csv_path = [{'DateTime': 1232298600.0, 'p(mbar)': '978.50', 'T(degC)': '2.73', 'Tpot(K)': '277.61',
    #          'Tdew(degC)': '-1.36', 'rh(%)': '74.30', 'VPmax(mbar)': '7.44', 'VPact(mbar)': '5.53',
    #          'VPdef(mbar)': '1.91', 'sh(g/kg)': '3.52', 'H2OC(mmol/mol)': '5.65', 'rho(g/m**3)': '1232.89',
    #          'wv(m/s)': '1.93', 'max.wv(m/s)': '4.50', 'wd(deg)': '233.30'}]
    csv_path = mongodb.queryAll({})
    with open("../full.json", "w") as file:
        json.dump(csv_path,file)
    # xlabel = 'Date time'
    # ylabel = ''
    # title = ''
    # save_path = 'images/full'
    # if not os.path.isdir(save_path):
    #     os.mkdir(save_path)
    # history(csv_path, save_path, xlabel, ylabel, title)
    #
    # # csv_path = mongodb.queryAll({"DateTime": {"$gt": 1451606400}})
    # with open("data/full1year.json", "r") as file:
    #     csv_path = json.load(file)
    #
    # xlabel = 'Date time'
    # ylabel = ''
    # title = ''
    # save_path = 'images/full1year'
    # if not os.path.isdir(save_path):
    #     os.mkdir(save_path)
    # history(csv_path, save_path, xlabel, ylabel, title)


def test_drawpredict():
    # truth = mongodb.queryAll({
    #     "$or": [
    #         {
    #             "$and":
    #                 [
    #                     {"DateTime": {"$lt": 1477353610}},
    #                     {"DateTime": {"$gt": 1457870400}},
    #                     {"DateTime": {"$mod": [24 * 3600, 0]}}
    #                 ]
    #         },
    #         {
    #             "$and":
    #                 [
    #                     {"DateTime": {"$gt": 1477353600}},
    #                     {"DateTime": {"$mod": [24 * 3600, 58200]}}
    #                 ]
    #         }
    #     ]
    # })
    # print(len(truth))

    pred = mongodb.queryPredict(
        {
            "$or": [
                {
                    "$and":
                        [
                            {"DateTime": {"$lt": 1477353610}},
                            {"DateTime": {"$mod": [24 * 3600, 0]}}
                        ]
                },
                {
                    "$and":
                        [
                            {"DateTime": {"$gt": 1477353600}},
                            {"DateTime": {"$mod": [24 * 3600, 58200]}}
                        ]
                }
            ]
        }
    )
    print(len(pred))
    with open("data/predictFullTruth.json", "r") as file:
        truth = json.load(file)

    with open("data/predictFullPred.json", "r") as file:
        # json.dump(pred,file)
        pred = json.load(file)

    xlabel = 'Date time'
    ylabel = ''
    title = ''
    save_path = 'images/predict'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    draw_predict(truth, pred, save_path, xlabel, ylabel, title)

    # truth = mongodb.queryAll(
    #     {
    #         "$and":
    #             [
    #                 {"DateTime": {"$gt": 1477958400}},
    #                 {"DateTime": {"$mod": [24 * 3600, 58200]}}
    #             ]
    #     }
    # )
    # print(len(truth))
    #
    # pred = mongodb.queryPredict(
    #     {
    #         "$and":
    #             [
    #                 {"DateTime": {"$gt": 1477958400}},
    #                 {"DateTime": {"$mod": [24 * 3600, 58200]}}
    #             ]
    #     }
    # )
    # print(len(pred))
    with open("data/predictFull1MonthTruth.json", "r") as file:
        truth = json.load(file)

    with open("data/predictFull1MonthPred.json", "r") as file:
        pred = json.load(file)
    xlabel = 'Date time'
    ylabel = ''
    title = ''
    save_path = 'images/predict1month'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    draw_predict(truth, pred, save_path, xlabel, ylabel, title)


def test_err():
    # truth = mongodb.queryAll({
    #     "$or": [
    #         {
    #             "$and":
    #                 [
    #                     {"DateTime": {"$lt": 1477353610}},
    #                     {"DateTime": {"$gt": 1457870400}},
    #                     {"DateTime": {"$mod": [24 * 3600, 0]}}
    #                 ]
    #         },
    #         {
    #             "$and":
    #                 [
    #                     {"DateTime": {"$gt": 1477353600}},
    #                     {"DateTime": {"$mod": [24 * 3600, 58200]}}
    #                 ]
    #         }
    #     ]
    # })
    # print(len(truth))
    #
    # pred = mongodb.queryPredict(
    #     {
    #         "$or": [
    #             {
    #                 "$and":
    #                     [
    #                         {"DateTime": {"$lt": 1477353610}},
    #                         {"DateTime": {"$mod": [24 * 3600, 0]}}
    #                     ]
    #             },
    #             {
    #                 "$and":
    #                     [
    #                         {"DateTime": {"$gt": 1477353600}},
    #                         {"DateTime": {"$mod": [24 * 3600, 58200]}}
    #                     ]
    #             }
    #         ]
    #     }
    # )
    with open("data/errorTruth.json", "r") as file:
        truth = json.load(file)

    with open("data/errorPred.json", "r") as file:
        pred = json.load(file)
    print(len(pred))
    # truth = [{'DateTime': 1232298600.0, 'p(mbar)': '978.50', 'T(degC)': '2.73', 'Tpot(K)': '277.61',
    #          'Tdew(degC)': '-1.36', 'rh(%)': '74.30', 'VPmax(mbar)': '7.44', 'VPact(mbar)': '5.53',
    #          'VPdef(mbar)': '1.91', 'sh(g/kg)': '3.52', 'H2OC(mmol/mol)': '5.65', 'rho(g/m**3)': '1232.89',
    #          'wv(m/s)': '1.93', 'max.wv(m/s)': '4.50', 'wd(deg)': '233.30'}]
    # pred = [{'DateTime': 1232298600.0, 'p(mbar)': '970.50', 'T(degC)': '1.73', 'Tpot(K)': '277.61',
    #              'Tdew(degC)': '-1.36', 'rh(%)': '80.30', 'VPmax(mbar)': '7.44', 'VPact(mbar)': '5.53',
    #              'VPdef(mbar)': '1.91', 'sh(g/kg)': '3.52', 'H2OC(mmol/mol)': '5.65', 'rho(g/m**3)': '1232.89',
    #              'wv(m/s)': '2.93', 'max.wv(m/s)': '4.50', 'wd(deg)': '300.30'}]
    xlabel = ''
    ylabel = 'Percen Error'
    title = ''
    save_path = 'images/err'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    percen_err(truth, pred, save_path, xlabel, ylabel, title)


def test_meanNstd_yy():
    # csv_path = '/home/thulx/Master/BI/jena_climate_2009_2016.csv'
    # csv_path = [{'DateTime': 1232298600.0, 'p(mbar)': '978.50', 'T(degC)': '2.73', 'Tpot(K)': '277.61',
    #              'Tdew(degC)': '-1.36', 'rh(%)': '74.30', 'VPmax(mbar)': '7.44', 'VPact(mbar)': '5.53',
    #              'VPdef(mbar)': '1.91', 'sh(g/kg)': '3.52', 'H2OC(mmol/mol)': '5.65', 'rho(g/m**3)': '1232.89',
    #              'wv(m/s)': '1.93', 'max.wv(m/s)': '4.50', 'wd(deg)': '233.30'}]
    # csv_path = mongodb.queryAll({})
    with open("../full.json", "r") as file:
        csv_path = json.load(file)
    save_path = 'images/meanstd'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    plot_cols = []
    xlabel = 'Date time'
    ylabel = ''
    title = ''
    meanNstd_yy(csv_path, save_path, plot_cols, xlabel, ylabel, title)


def test_wind():
    with open("../full.json", "r") as file:
        csv_path = json.load(file)
    # csv_path = mongodb.queryAll({})
    # csv_path = [{'DateTime': 1232298600.0, 'p(mbar)': '978.50', 'T(degC)': '2.73', 'Tpot(K)': '277.61',
    #          'Tdew(degC)': '-1.36', 'rh(%)': '74.30', 'VPmax(mbar)': '7.44', 'VPact(mbar)': '5.53',
    #          'VPdef(mbar)': '1.91', 'sh(g/kg)': '3.52', 'H2OC(mmol/mol)': '5.65', 'rho(g/m**3)': '1232.89',
    #          'wv(m/s)': '1.93', 'max.wv(m/s)': '4.50', 'wd(deg)': '233.30'}]
    save_path = 'images/win.pdf'
    title = ''
    wind(csv_path, save_path, title)


def test_mean_yy():
    # csv_path = '/home/thulx/Master/BI/jena_climate_2009_2016.csv'
    csv_path = mongodb.queryAll({})
    save_path = 'mean_yy2.png'
    plot_cols = ['T (degC)', 'p (mbar)']
    xlabel = 'Date time'
    ylabel = ''
    title = ''
    mean_yy(csv_path, save_path, plot_cols, xlabel, ylabel, title)


def test_predict():
    csv_path = '/home/thulx/Master/BI/jena_climate_2009_2016.csv'
    df = pd.read_csv(csv_path)
    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')[:200].to_numpy()
    y = df['T (degC)'][:200].to_numpy()
    yhat = df['T (degC)'][200:400].to_numpy()
    save_path = 'predict.png'
    xlabel = 'Date time'
    ylabel = 'T (degC)'
    title = 'Prediction'
    predict(date_time, y, yhat, save_path, xlabel, ylabel, title)


def test_predict_dif():
    csv_path = '/home/thulx/Master/BI/jena_climate_2009_2016.csv'
    df = pd.read_csv(csv_path)
    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')[:200].to_numpy()
    y = df['T (degC)'][:200].to_numpy()
    yhat = df['T (degC)'][200:400].to_numpy()
    save_path = 'predict-dif.png'
    xlabel = 'Date time'
    ylabel = 'Difference'
    title = 'Evaluate prediction'
    predict_dif(date_time, y, yhat, save_path, xlabel, ylabel, title)


def test_correlation():
    csv_path = '/home/thulx/Master/BI/jena_climate_2009_2016.csv'
    save_path = 'corr.png'
    xlabel = ''
    ylabel = ''
    title = 'Correlation Matrix'
    correlation(csv_path, save_path, xlabel, ylabel, title)


def test_load():
    data = [{'DateTime': 1232298600.0, 'p(mbar)': '978.50', 'T(degC)': '2.73', 'Tpot(K)': '277.61',
             'Tdew(degC)': '-1.36', 'rh(%)': '74.30', 'VPmax(mbar)': '7.44', 'VPact(mbar)': '5.53',
             'VPdef(mbar)': '1.91', 'sh(g/kg)': '3.52', 'H2OC(mmol/mol)': '5.65', 'rho(g/m**3)': '1232.89',
             'wv(m/s)': '1.93', 'max.wv(m/s)': '4.50', 'wd(deg)': '233.30'}]
    data = load(data)
    print(data)


if __name__ == '__main__':
    print('history')
    test_history()
    print('predict')
    test_drawpredict()
    print('error')
    test_err()
    print('mean')
    test_meanNstd_yy()
    print("wind")
    test_wind()
