var cfg = {
    "_id": "rs1",
    "members": [
        {
            "_id": 0,
            "host": "shard03:27017"
        },
        {
            "_id": 1,
            "host": "shard04:27017"
        },
        {
            "_id": 2,
            "host": "shard05:27017"
        }
    ]
};

var status = rs.initiate(cfg);

printjson(status);