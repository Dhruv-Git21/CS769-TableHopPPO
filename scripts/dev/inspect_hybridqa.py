import json, sys
R = "data/raw/hybridqa/"
tr = json.load(open(R + "HybridQA/released_data/train.traced.json"))
ex = tr[0]
print("question keys:", sorted(ex))
print("answer-node n:", len(ex["answer-node"]), "node len:", len(ex["answer-node"][0]), "types:", [type(x).__name__ for x in ex["answer-node"][0]])
tb = json.load(open(R + "WikiTables-WithLinks/tables_tok/%s.json" % ex["table_id"]))
rq = json.load(open(R + "WikiTables-WithLinks/request_tok/%s.json" % ex["table_id"]))
print("table keys:", sorted(tb), "n_rows:", len(tb["data"]), "n_cols:", len(tb["header"]))
print("header item:", type(tb["header"][0]).__name__, "len", len(tb["header"][0]))
print("cell item:", type(tb["data"][0][0]).__name__, "len", len(tb["data"][0][0]))
print("request type:", type(rq).__name__, "n:", len(rq), "sample key sample:", list(rq)[:1] and type(rq[list(rq)[0]]).__name__)
n_links = sum(len(c[1]) for r in tb["data"] for c in r)
joined = sum(1 for r in tb["data"] for c in r for l in c[1] if l in rq)
print("links in table:", n_links, "resolved in request:", joined)
print("sample node:", ex["answer-node"][0])
