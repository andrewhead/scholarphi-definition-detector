import requests
import json
from pprint import pprint

data = { "sents": [{"sent_id": 0, "text": "We define the width and height as (w, h)."}], "terms": [{"term_id": 0, "sent_id": 0, "start": 35, "end": 36, "text": "w"}, {"term_id": 1, "sent_id": 0, "start": 38, "end": 39, "text": "h"}] }

r = requests.post('http://localhost:8080/get_prediction', data = json.dumps(data)  )
output = r.json()

pprint(output)

