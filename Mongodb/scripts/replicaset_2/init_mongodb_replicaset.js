var cfg = {
    "_id": "rs2",
    "members": [
        {
            "_id": 0,
            "host": "shard06:27017"
        },
        {
            "_id": 1,
            "host": "shard07:27017"
        },
        {
            "_id": 2,
            "host": "shard08:27017"
        }
    ]
};

var status = rs.initiate(cfg);

printjson(status);