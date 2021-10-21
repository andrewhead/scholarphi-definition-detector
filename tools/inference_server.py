import sys
import os
import json
import argparse
import tempfile
from pprint import pprint
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, create_model
import requests
import aiofiles
import aiohttp
from configparser import ConfigParser


from definition_detection_module import DefinitionModule
sys.path.append("../")
from utils import match_tokenized_to_untokenized

app = FastAPI()

term_placeholder = "SYMBOL"
separation_token = "|||"



class InputForm(BaseModel):
    sents: List[create_model('Sent', sent_id=(int, ...), text=(str, ...))] = []
    terms: List[create_model('Term', term_id=(int, ...), sent_id=(int, ...), start=(int, ...), end=(int, ...), text=(str, ...))] = []

class OutputForm(BaseModel):
    message: str
    DefPred = create_model(
        "DefPred",
        sent_id = (str, ...),
        start = (int, ...),
        end = (int, ...),
        text = (str, ...)
    )
    output: List[create_model('Pred', term_id=(int, ...), def_spans=(List[DefPred], ...))] = []


def load_model():
    # MODEL_NAME = "DocDefQueryInplaceFixedMIA"
    # definition_model = DefinitionModule("/home/dongyeok//data/ScholarPhi/v4/model/v5/{}/MAXLEN=100/".format(MODEL_NAME), MODEL_NAME)

    config = ConfigParser()
    PROJ_DIR = os.path.join(os.path.dirname(__file__),"..")
    config.read(os.path.join(PROJ_DIR, "config.ini"))
    MODEL_NAME = config["heddex"]["model_name"]
    MODEL_DIR = config["heddex"]["model_dir"]
    definition_model = DefinitionModule(os.path.join(PROJ_DIR, MODEL_DIR), MODEL_NAME)

    return definition_model, MODEL_NAME




def merge_consecutive_definition_spans(spans):
    merged = True
    while(merged):
        merged = False
        for dfid in range(len(spans) - 1):
            if spans[dfid]["end"] == spans[dfid+1]["start"]:
                assert spans[dfid]["sent_id"] ==spans[dfid+1]["sent_id"]
                new_span = {"sent_id": spans[dfid]["sent_id"],
                            "start": spans[dfid]["start"],
                            "end": spans[dfid+1]["end"],
                            "text": spans[dfid]["text"] + spans[dfid+1]["text"],
                            "model_text": spans[dfid]["model_text"] + " " + spans[dfid+1]["model_text"]
                            }
                # print("Removing ",dfid, spans[dfid])
                # print("Removing ",dfid+1, spans[dfid+1])
                # print("Adding ",new_span)
                # spans = [new_span] + spans[0:dfid,dfid+2:]
                spans.pop(dfid+1)
                spans.pop(dfid)
                spans = [new_span] + spans
                merged = True
                break
    print(spans)
    return spans



definition_model, MODEL_NAME = load_model()

@app.post("/get_prediction/")
async def get_prediction(data: InputForm):
    """
    input = {
      'sents': [{'sent_id': 0, 'text': 'Let $x$ be a cool thing.'}, ... {}],
      'terms': [{'term_id': 0, 'sent_id': 0, 'start': 4, 'end': 7, 'text': '$x$'}, ... {}]
    }
    output = [
       {'term_id': 0, 'def_spans': [{'sent_id': 0, 'start': 11, 'end': 23, 'text': 'a cool thing'}, ...]},
       ...
    ]
    """


    error_message = ""
    outputs = []
    # change data format to HEDDEx-compatible format
    for sentence in data.sents:
        # retrieved matched terms
        matched_terms = []
        for term in data.terms:
            if sentence.sent_id == term.sent_id:
                matched_terms.append(term)

        # sort by start position
        matched_terms.sort(key=lambda x: x.start, reverse=False)

        print(sentence)
        print(matched_terms)
        print("-----------------")

        # targeting term SYMBOLs;
        output_definitions = []
        for target_term in matched_terms:
            # character mapping between original sentence and training format
            char_mapping_dict = {}
            offset_history = [0]
            text = str(sentence.text)
            # print(sentence.text[target_term.start:target_term.end], target_term.text)
            if sentence.text[target_term.start:target_term.end] != target_term.text:
                return {"message": "Fail to predict definitions. Error message: {}".format("Term text is not matched with text extracted from the start and end position."),"output": []}

            offset = 0
            for tid, aterm in enumerate(matched_terms):
                # replace terms with SYMBOLs
                # adding ||| before and after the target term
                if aterm.term_id == target_term.term_id:
                    symbol_with_placeholder = f"{separation_token} " +  term_placeholder + f" {separation_token}"
                else:
                    symbol_with_placeholder = term_placeholder
                text = text[:aterm.start+offset] + symbol_with_placeholder + text[aterm.end+offset:]

                char_mapping_dict[tid] = (aterm.start, aterm.end, offset, len(symbol_with_placeholder))

                offset += len(symbol_with_placeholder)-len(aterm.text)


            #TODO Docker, Model change in S3, etc.
            print("=>",text)

            definitions = False
            try:
                definitions = definition_model.predict([text])
            except Exception as e:
                print(e)
                return {"message": "Fail to predict definitions. Error message: {}".format(e),"output": []}




            definition = definitions[0]
            intent_pred = definition["intent_prediction"][MODEL_NAME]
            slot_preds = definition["slot_prediction"][MODEL_NAME]
            tokens = definition["tokens"]

            print(tokens)
            print(slot_preds)

            # mapping between tokens from model prediction and raw text again
            assert len(tokens) == len(slot_preds)
            is_target_symbol = False
            symbol_index = 0
            offset = 0
            raw_text_offset = 0
            definition_spans = []

            # # match_tokenized_to_untokenizeda
            # print("-----------")
            # print(sentence.text)
            # print(" ".join(tokens))
            # print("-----------")

            local_offset = 0
            for slot, token in zip(slot_preds, tokens):
                if slot == "DEF":

                    token_length = len(token)
                    if "SYMBOL" in token:
                        local_length = len(token) - len("SYMBOL")
                        token_length = char_mapping_dict[symbol_index][1] - char_mapping_dict[symbol_index][0]  + local_length

                    def_span = {"sent_id": sentence.sent_id,
                                "start": raw_text_offset,
                                "end": raw_text_offset + token_length + 1,
                                "text": sentence.text[raw_text_offset:raw_text_offset + token_length + 1],
                                "model_text": token}
                    # print("\t\tAdded:",def_span)
                    definition_spans.append(def_span)

                if "|||" in token:
                    local_length = len(token) - len("|||")
                    raw_text_offset +=  local_length
                    # print(symbol_index, local_offset, raw_text_offset, token)
                elif "SYMBOL" in token:
                    local_length = len(token) - len("SYMBOL")
                    print(symbol_index, local_offset, raw_text_offset, token, sentence.text[raw_text_offset:raw_text_offset+len(matched_terms[symbol_index].text)+1])
                    # print("\t",char_mapping_dict[symbol_index])
                    raw_text_offset +=  char_mapping_dict[symbol_index][1] - char_mapping_dict[symbol_index][0]  + local_length + 1
                    #len(matched_terms[symbol_index].text) + 1 + len(token) - len("SYMBOL")

                    symbol_index += 1
                else:
                    # print(symbol_index, local_offset, raw_text_offset, token, sentence.text[raw_text_offset:raw_text_offset+len(token)+1])
                    raw_text_offset += len(token) +1 # 1 for space

                local_offset += len(token) + 1


            # TODO merge consecutive spans?
            merged_definition_spans = merge_consecutive_definition_spans(definition_spans)


            output = {"term_id": target_term.term_id,
                      "def_spans": merged_definition_spans}

            print("=========")
            for dfid, df in enumerate(output["def_spans"]):
                print(dfid, output["term_id"], df)
                print("\t{}\t{}".format(df["text"], df["model_text"]))
                # assert sentence.text[df["start"]:df["end"]] == df["text"]
            print("=========")
            print("")

            outputs.append(output)

        if outputs:
            return {"message": "Successfully predicted symbol definitions of input text",
                    "output": outputs}
        else:
            return {"message": "Fail to predict definitions. Error message: {}".format(error_message),"output": []}


@app.post("/get_prediction_from_raw_text/")
async def get_prediction(text: str):
    # definition_model, MODEL_NAME = load_model()

    definitions = False
    error_message = ""
    try:
        text = text.split('\n')
        definitions = definition_model.predict(text)
    except Exception as e:
        print(e)
        error_message = e

    if definitions:
        return {"message": "Successfully predicted symbol definitions of input text",
                "output": definitions}
    else:
        return {"message": "Fail to predict definitions. Error message: {}".format(error_message),
                "output": []}


@app.get("/")
async def root():
    return {"message": "Hello World"}



