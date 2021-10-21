# HEDDEx

This repository consists of the Pytorch implementation (with HuggingFace) of HEDDEx for predicting terms-definition, abbreviation-expansion and symbol-nickname in sentences.

## Installation

Try running ```pip install -r requirements.txt ``` to install python dependencies.
Try running the following scripts to download datasets and pre-trained model weights. Please see [Datasets](##Datasets) and [Pre-Trained-Models](
##Pre-Trained-Models) sections below.
```
./run_download_model.sh
./run_download_data.sh
```

---

## Training

*Note :*  You can train AI2020, W00, DocDef2 separately or in any combination (joint model). DocDefQueryInplace can only be trained separately.

- Edit the `scripts/run_train_v3_joint.sh` file to reflect the correct location of the `DATA_DIR`
- The datasets you want to train the joint model on, must be `'+'` separated value of `DATASETS` variable
``` bash run_train_v3_joint.sh```

You can also train HEDDEx on your curstom dataset. A new input processor needs to be added to the `data_loader.py` file, to be able to process the new dataset, and then add the dataset t the `DATASETS` variable. 

## Evaluation
- Edit the `scripts/run_test_v3_joint.sh` file to reflect the correct location of the `DATA_DIR` (which should contain the dataset and the models)
- The datasets you want to evaluate with the joint model on, must be `'+'` separated value of `DATASETS` variable
``` bash run_test_v3_joint.sh```


## Inference
Once you either train the models or download pre-trained ones, there are three ways of running inference on any new sentence.

1. #### Jupyter Notebook
    - For interactive inference, use the jupyter notebook `inference/joint_inference_notebook.ipynb`. You can find further instructions and details in the notebook
2. #### Command-line Inference
    - For inference on a batch of sentences, first create an input file with each sentence in a new line. Symbols must be enclosed in double square brackets (Eg : The input to the matrix [[M]] is a vector [[v]] of specified lengths.)
    - To infer from the query model, Symbol in question must be enclosed in `---` (Eg : The input to the matrix --- [[M]] --- is a vector [[v]] of specified lengths.)
    - Run `python run_inference.py --model <model path : eg : ./models/AI2020_model> --task <Task names : eg : AI2020> --input <input file : eg : input.txt>` script and run it.  The model file path needs to be the path to the directory containing the `roberta-large...` folder
    - Task names much be `+` separated. Possible task names :
        - Slot prediction : W00, DocDef2, AI2020
        - Query based prediction : DocDefQueryInplace
3. #### Interactive Inference on the Browser
    - For interactive inference on the browser, run `python inference/inference_server.py`
    - Open `http://0.0.0.0:5000/` on any browser.


## Standalone API service
You can use our definition detectors as standalone API services.

### Run service from a script

Running fastAPI service:

```
    cd ./tools/
    ./run_uviconr_server.sh
```

### Run with Docker file

Setting up using Docker:

1. Create the docker image:
    ```
    docker build -t heddex .
    ```

2. Start the docker container and it will start a detection service:
    ```bash
    docker run -p 8080:8080 -ti heddex   
    ```
    
3. Send a post request with input text with sy and get the layout data (possibly after a while). Here is an exemplar script:
    ```python
    import requests
    import json
    from pprint import pprint

    data = { "sents": [{"sent_id": 0, "text": "We define the width and height as (w, h)."}], \
    "terms": [{"term_id": 0, "sent_id": 0, "start": 35, "end": 36, "text": "w"}, \
    {"term_id": 1, "sent_id": 0, "start": 38, "end": 39, "text": "h"}] }

    r = requests.post('http://localhost:8080/get_prediction', data = json.dumps(data)  )
    output = r.json()

    pprint(output)

    ```
 
 
 Then, you will get output like below:
 
```json
{'message': 'Successfully predicted symbol definitions of input text',
 'output': [{'def_spans': [{'end': 31,
                            'model_text': 'width and height',
                            'sent_id': 0,
                            'start': 14,
                            'text': 'width and height '}],
             'term_id': 0},
            {'def_spans': [{'end': 31,
                            'model_text': 'height',
                            'sent_id': 0,
                            'start': 24,
                            'text': 'height '}],
             'term_id': 1}]}
```

---

## Datasets

| Task | Types | Original |  Our pre-processed version |
| --- | --- | --- | --- |
| AI2020 | acronym expansions | [link](https://github.com/amirveyseh/AAAI-21-SDU-shared-task-1-AI) | [link](https://scholarphi4nlp.s3-us-west-2.amazonaws.com/data/AI2020.zip) |
| W00 | term definitions | [link](https://aclanthology.org/D13-1073/) | [link](https://scholarphi4nlp.s3-us-west-2.amazonaws.com/data/W00.zip) |
| DocDef2 | Symbol nicknames | | [link](https://scholarphi4nlp.s3-us-west-2.amazonaws.com/data/DocDef2.zip) |
| DocDefQueryInplace | DocDef2 in query format. If a sentence has more than one symbol, it is duplicated and each symbol is queried separately. | | [link](https://scholarphi4nlp.s3-us-west-2.amazonaws.com/data/DocDefQueryInplace.zip) |

## Pre-Trained Models 

| Task | Types | Link to downalod |
| --- | --- | --- |
| AI2020 | acronym expansions | [v4](https://scholarphi4nlp.s3-us-west-2.amazonaws.com/model/v4/abbrexp.zip) |
| W00 | term definitions | [v4](https://scholarphi4nlp.s3-us-west-2.amazonaws.com/model/v4/termdef.zip) |
| DocDef2 | Symbol nicknames | [v4](https://scholarphi4nlp.s3-us-west-2.amazonaws.com/model/v4/symnick.zip) |
| DocDefQueryInplace | DocDef Query format | [v4](https://scholarphi4nlp.s3-us-west-2.amazonaws.com/model/v4/symnick_query.zip) [v5](https://scholarphi4nlp.s3-us-west-2.amazonaws.com/model/v5/symnick_query_mia.zip) |
| W00+AI2020+DocDef2 | Joint Model | [v4](https://scholarphi4nlp.s3-us-west-2.amazonaws.com/model/v4/joint_symnick_abbrexp_termdef.zip) [v5](https://scholarphi4nlp.s3-us-west-2.amazonaws.com/model/v5/joint_symnick_abbrexp_termdef.zip) |
| W00+AI2020+DocDef2QueryInplace | Joint Model | [v5](https://scholarphi4nlp.s3-us-west-2.amazonaws.com/model/v5/joint_symnickquery_abbrexp_termdef.zip) |

Note: `v5` verion includes various features including additional symbol definition dataset, error fixs, etc.

