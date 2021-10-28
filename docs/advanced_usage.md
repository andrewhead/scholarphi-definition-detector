# Advanced usage

Those who are building upon the definition detector may wish to train the definition detector, and to have additional support in running and debugging the detector. The following notes offer suggestions for how you might do so. Note that these notes are from a previous version of this repository, so use them at your own risk; the authors do not guarantee that everything in these instructions is up-to-date.

## Local installation

To set up the definition detector locally (rather than using the Docker container), we recommend following along with the commands present in the Dockerfile corresponding to which model you want to develop (see the instructions on Docker installation).

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

### 1. Jupyter Notebook

- For interactive inference, use the jupyter notebook `inference/joint_inference_notebook.ipynb`. You can find further instructions and details in the notebook
  
### 2. Command-line Interface

 - For inference on a batch of sentences, first create an input file with each sentence in a new line. Symbols must be enclosed in double square brackets (Eg : The input to the matrix [[M]] is a vector [[v]] of specified lengths.)
 - To infer from the query model, Symbol in question must be enclosed in `---` (Eg : The input to the matrix --- [[M]] --- is a vector [[v]] of specified lengths.)
 - Run `python run_inference.py --model <model path : eg : ./models/AI2020_model> --task <Task names : eg : AI2020> --input <input file : eg : input.txt>` script and run it.  The model file path needs to be the path to the directory containing the `roberta-large...` folder
 - Task names much be `+` separated. Possible task names :
     - Slot prediction : W00, DocDef2, AI2020
     - Query based prediction : DocDefQueryInplace
  
### 3. Interactive Inference on the Browser

- For interactive inference on the browser, run `python inference/inference_server.py`
- Open `http://0.0.0.0:5000/` on any browser.


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

