# ScholarPhi Definition Detector

This repository contains a PyTorch implementation of the HEDDEx definition 
detector for detecting symbol definitions and abbreviation expansions in 
scientific papers.

## Launching the service

This repository provides two services for detecting definitions: one service for 
detecting expansions of abbreviations, and one service for detecting definitions 
of symbols. The two services share most of the same underlying code, though 
there are differences in the model weights.

### Abbreviation service

Build the abbreviation expansion service as follows:

```bash
docker build . -t abbreviation-detector -f Dockerfile.abbreviations
```

This build will take a long time, because one of the steps in the build process 
is to download a rather large file containing model weights.

Then, launch the service as follows:

```bash
docker run -it --rm -p 8000:8080 abbreviation-detector
```

To serve the service on a custom port, change the value of `8000` in the `-p` 
argument to a port of your choice. To use the shell in the container, use the 
`--entrypoint /bin/bash` argument, placing the argument before the name of the 
image (`abbreviation-detector`).

Test out the service, by running the following Python code (assuming you have 
already installed `requests` with `pip install requests`).

```python
import requests
import json

params = {
  # The 'text' attribute should consist of a single sentence.
  "text": (
    "We use a CNN (convolutional neural network) and an RNN " +
    "(recurrent neural network)."
  )
}

resp = requests.post(
  # The port here should match the port the service was launched with
  # in the 'docker run' command.
  'http://localhost:8000/get_prediction_from_raw_text',
  params=params
)

print(json.dumps(resp.json(), indent=2))
```

The result of submitting this query should be as follows. Note that irrelevant fields have been elided.

```json
{
  "message": "Successfully predicted symbol definitions of input text",
  "output": [
    {
      "tokens": [
        "We",
        "use",
        "a",
        "CNN",
        "(",
        "convolutional",
        "neural",
        "network",
        ")",
        "and",
        "an",
        "RNN",
        "(",
        "recurrent",
        "neural",
        "network",
        ")",
        "."
      ],
      ...
      "slot_prediction": {
        ...
        "AI2020": [
          "O",
          "O",
          "O",
          "TERM",
          "O",
          "DEF",
          "DEF",
          "DEF",
          "O",
          "O",
          "O",
          "TERM",
          "O",
          "DEF",
          "DEF",
          "DEF",
          "O",
          "O"
        ],
        ...
      }
    }
  ]
}
```

In the output, the `tokens` attribute contains a list of tokens that the abbreviation detector split the sentence into. Each of these tokens is assigned a tag in the `["slot_prediction"]["AI2020"]` property, which suggests whether that token belongs to an abbreviation (`TERM`), an expansion (`DEF`) or neither of the above `O`.

If a sentence contains multiple abbreviations, the abbreviation detector currently has no way of detecting which abbreviations correspond to which expansions. We have had some success in our own use cases of the tool in greedily pairing abbreviations and expansions from left-to-right.

### Symbol definition service

Build the symbol definition detection service as follows:

```bash
docker build . -t symbol-definition-detector -f Dockerfile.definitions
```

This build will take a long time, because one of the steps in the build process 
is to download a rather large file containing model weights.

Then, launch the service as follows:

```bash
docker run -it --rm -p 8000:8080 abbreviation-detector
```

See the notes above in the "Abbreviation service" section for suggestions of 
arguments you may wish to pass to this command.

Test out the service, by running the following Python code (assuming you have 
already installed `requests` with `pip install requests`).

```python
import requests
import json

query = {
  # Note that the service currently only supports definition detection
  # for one sentence at a time.
  "sents": [{
    "sent_id": 0,
    "text": "We define the width as $w$."
  }],
  # For the definition detector to work, it must be provided with a list of
  # the symbols for which definitions will be searched for in the sentence.
  # Each symbol should have an entry in the list of 'terms'. Each symbol
  # should be assigned a distinct term_id.
  "terms": [{
    "term_id": 0,
    "sent_id": 0,
    "start": 23,
    "end": 26,
    "text": "$w$"
  }]
}

resp = requests.post(
  # The port here should match the port the service was launched with
  # in the 'docker run' command.
  'http://localhost:8000/get_prediction',
  data=json.dumps(query)
)

print(json.dumps(resp.json(), indent=2))
```

The result of submitting this query should be:

```json
{
  "message": "Successfully predicted symbol definitions of input text",
  "output": [
    {
      "term_id": 0,
      "def_spans": [
        {
          "sent_id": 0,
          "start": 14,
          "end": 20,
          "text": "width ",
          "model_text": "width"
        }
      ]
    }
  ]
}
```

The definition of the term can be found in the `text` attribute of the 
`def_spans` attribute of the output.

## Advanced usage

Those who wish to retrain / evaluate the definition detectors may wish to consult the guidance in the [Advanced Usage](docs/advanced_usage) document. Note that this document was written some time ago and (unfortunately) may no longer represent the current state of the code, due to changes in the personnel maintaining this repository. We include it here with the hopes that it can provide some value to those who are diving deep into the particulars of this repository.
