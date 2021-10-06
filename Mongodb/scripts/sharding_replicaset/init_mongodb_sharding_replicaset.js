var addShard0 = sh.addShard("rs0/shard00:27017,shard01:27017,shard02:27017");
var addShard1 = sh.addShard("rs1/shard03:27017,shard04:27017,shard05:27017");
// sh.enableSharding("weather-data")
// sh.shardCollection("weather-data.PredictedWeatherDataAll", {"_id": 1})
// sh.shardCollection("weather-data.WeatherDataAll", {"_id": 1})
printjson(addShard0);
printjson(addShard1);
