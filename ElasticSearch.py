from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

es = Elasticsearch([{"host": "localhost", "port": "9200"}])


def create_weather_data_index():
    if not es.indices.exists(index="bml_user"):
        body = {
            "number_of_replicas": 2,
            "mappings": {
                "properties": {
                    "DateTime": {"type": "date"},
                    "p(mbar)": {"type": "float"},
                    "T(degC)": {"type": "float"},
                    "Tpot(K)": {"type": "float"},
                    "Tdew(degC)": {"type": "float"},
                    "rh(%)": {"type": "float"},
                    "VPmax(mbar)": {"type": "float"},
                    "VPact(mbar)": {"type": "float"},
                    "VPdef(mbar)": {"type": "float"},
                    "sh(g/kg)": {"type": "float"},
                    "H2OC(mmol/mol)": {"type": "float"},
                    "rho(g/m**3)": {"type": "float"},
                    "wv(m/s)": {"type": "float"},
                    "max.wv(m/s)": {"type": "float"},
                    "wd(deg)": {"type": "float"},
                }
            }
        }
        try:
            res = es.indices.create(index="weather_data", body=body)
            return res
        except Exception as e:
            print("already exist")


def create_weather_data(weather_data):
    res = es.index(index='weather_data', doc_type='_doc', body=weather_data, request_timeout=30)
    return {"result": res['result']}


def create_bulk_weather_data(weather_datas):
    bulk(es, weather_datas, index="weather_data")
