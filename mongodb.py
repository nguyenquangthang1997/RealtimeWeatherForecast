import pymongo

mongoClient = pymongo.MongoClient("mongodb://localhost:27017/")
weather_data = mongoClient["weather-data"]

weather_data_all_collection = weather_data["WeatherDataAll"]


def insert_weather_data(datas):
    weather_data_all_collection.insert_many(datas)
