import pymongo

mongoClient = pymongo.MongoClient("mongodb://localhost:27017/")
weather_data = mongoClient["weather-data"]

weather_data_all_collection = weather_data["WeatherDataAll"]
predicted_weather_data_all_collection = weather_data["PredictedWeatherDataAll"]


def insert_weather_data(datas):
    weather_data_all_collection.insert_many(datas)


def insert_predicted_weather_data(datas):
    predicted_weather_data_all_collection.insert_many(datas)


def query(_query):
    result = []
    responses = weather_data_all_collection.find(_query)
    for doc in responses:
        result.append(doc)
    return result
#
#
# # example
# # https://www.w3schools.com/python/python_mongodb_query.asp
# print(query({"VPact(mbar)": "5.53"}))
