import csv
import time
import mongodb

# with open('predict.csv') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',')
#     i = 0
#     data = []
#     for row in spamreader:
#         if row[1] != "Date Time":
#             i = i + 1
#             print(i)
#             data.append({
#                 "DateTime": int(time.mktime(time.strptime(row[1], "%d.%m.%Y %H:%M:%S"))),
#                 "p(mbar)": row[2],
#                 "T(degC)": row[3],
#                 "Tpot(K)": row[4],
#                 "Tdew(degC)": row[5],
#                 "rh(%)": row[6],
#                 "VPmax(mbar)": row[7],
#                 "VPact(mbar)": row[8],
#                 "VPdef(mbar)": row[9],
#                 "sh(g/kg)": row[10],
#                 "H2OC(mmol/mol)": row[11],
#                 "rho(g/m**3)": row[12],
#                 "wv(m/s)": row[13],
#                 "max.wv(m/s)": row[14],
#                 "wd(deg)": row[15]
#             })
#             if i % 5000 == 0:
#                 mongodb.insert_predicted_weather_data(data)
#                 data = []
#     mongodb.insert_predicted_weather_data(data)

# with open('preprocessed_data.csv') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',')
#     i = 0
#     data = []
#     for row in spamreader:
#         if row[1] != "Date Time":
#             i = i + 1
#             print(i)
#             data.append({
#                 "DateTime": int(time.mktime(time.strptime(row[1], "%d.%m.%Y %H:%M:%S"))),
#                 "p(mbar)": row[2],
#                 "T(degC)": row[3],
#                 "Tpot(K)": row[4],
#                 "Tdew(degC)": row[5],
#                 "rh(%)": row[6],
#                 "VPmax(mbar)": row[7],
#                 "VPact(mbar)": row[8],
#                 "VPdef(mbar)": row[9],
#                 "sh(g/kg)": row[10],
#                 "H2OC(mmol/mol)": row[11],
#                 "rho(g/m**3)": row[12],
#                 "wv(m/s)": row[13],
#                 "max.wv(m/s)": row[14],
#                 "wd(deg)": row[15]
#             })
#             if i % 5000 == 0:
#                 mongodb.insert_pre_process_weather_data(data)
#                 data = []
#     mongodb.insert_pre_process_weather_data(data)

# with open('jena_climate_2009_2016.csv') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',')
#     i = 0
#     data = []
#     for row in spamreader:
#         if row[0] != "Date Time":
#             i = i + 1
#             print(i)
#             data.append({
#                 "DateTime": int(time.mktime(time.strptime(row[0], "%d.%m.%Y %H:%M:%S"))),
#                 "p(mbar)": row[1],
#                 "T(degC)": row[2],
#                 "Tpot(K)": row[3],
#                 "Tdew(degC)": row[4],
#                 "rh(%)": row[5],
#                 "VPmax(mbar)": row[6],
#                 "VPact(mbar)": row[7],
#                 "VPdef(mbar)": row[8],
#                 "sh(g/kg)": row[9],
#                 "H2OC(mmol/mol)": row[10],
#                 "rho(g/m**3)": row[11],
#                 "wv(m/s)": row[12],
#                 "max.wv(m/s)": row[13],
#                 "wd(deg)": row[14]
#             })
#             if i % 5000 == 0:
#                 mongodb.insert_weather_data(data)
#                 data = []
#     mongodb.insert_weather_data(data)