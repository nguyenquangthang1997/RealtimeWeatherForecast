var addShard0 = sh.addShard("rs0/shard00:27017,shard01:27017,shard02:27017");
var addShard1 = sh.addShard("rs1/shard03:27017,shard04:27017,shard05:27017");
var addShard2 = sh.addShard("rs2/shard06:27017,shard07:27017,shard08:27017");

printjson(addShard0);
printjson(addShard1);
printjson(addShard2);