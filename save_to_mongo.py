import csv
import time
import mongodb

with open('jena_climate_2009_2016.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    i = 0
    data = []
    for row in spamreader:
        if row[0] != "Date Time":
            i = i + 1
            print(i)
            data.append({
                "DateTime": time.mktime(time.strptime(row[0], "%d.%m.%Y %H:%M:%S")),
                "p(mbar)": row[1],
                "T(degC)": row[2],
                "Tpot(K)": row[3],
                "Tdew(degC)": row[4],
                "rh(%)": row[5],
                "VPmax(mbar)": row[6],
                "VPact(mbar)": row[7],
                "VPdef(mbar)": row[8],
                "sh(g/kg)": row[9],
                "H2OC(mmol/mol)": row[10],
                "rho(g/m**3)": row[11],
                "wv(m/s)": row[12],
                "max.wv(m/s)": row[13],
                "wd(deg)": row[14]
            })
            if i % 5000 == 0:
                mongodb.insert_weather_data(data)
                data = []
