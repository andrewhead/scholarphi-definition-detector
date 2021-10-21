import os
import sys
import os
import sys
import json
from pprint import pprint
from collections import defaultdict

from captum.attr._utils.visualization import *

def show_annotation_to_html(tokens,symbols, labels, label):
    symbol_index = 0
    token_with_preds = []
    for token, label in zip(tokens, labels):
        if token == "SYMBOL":
            token = '<span class="tooltip" data-text="{}">{}</span>'.format(symbols[symbol_index],token)
            symbol_index += 1
        token_ = ""
        if label.endswith('TERM'):
            token_ = '<font color="red"><u>{}</u></font>'.format(token)
        elif label.endswith('DEF'):
            token_ = '<font color="blue"><i>{}</i></font>'.format(token)
        else:
            token_ = token
        token_with_preds.append(token_)

    token_with_labels = []
    for token, label in zip(tokens, labels):
        token_ = ""
        if label.endswith('TERM'):
            token_ = '<font color="red"><u>{}</u></font>'.format(token)
        elif label.endswith('DEF'):
            token_ = '<font color="blue"><i>{}</i></font>'.format(token)
        else:
            token_ = token
        token_with_labels.append(token_)

    row = "<td>{}</td>".format(' '.join(token_with_labels))

    return row


def visualize_instance(data, is_label_provided=True, filename=False) -> None:
    dom = []
    rows = []
    preds_ids,label_ids = [], []
    for d in data:
        color_intent = ''

        if d['intent_pred'] == d['intent_label']:
            color_intent = "bgcolor=white"
        else:
            color_intent = "bgcolor=grey"
        color_eval = ''


        symbol_index = 0
        token_with_preds = []
        for token, label in zip(d['token'], d['slot_pred']):
            if token == "SYMBOL":
                token = '<span title="{}" class="tool_tip" >{}</span>'.format(d["raw"]["symbols"][symbol_index],token)
                symbol_index += 1

            token_ = ""
            if label.endswith('TERM'):
                token_ = '<font color="red"><u>{}</u></font>'.format(token)
            elif label.endswith('DEF'):
                token_ = '<font color="blue"><i>{}</i></font>'.format(token)
            else:
                token_ = token
            token_with_preds.append(token_)


        token_with_labels = []
        for token, label in zip(d['token'], d['slot_label']):
            token_ = ""
            if label.endswith('TERM'):
                token_ = '<font color="red"><u>{}</u></font>'.format(token)
            elif label.endswith('DEF'):
                token_ = '<font color="blue"><i>{}</i></font>'.format(token)
            else:
                token_ = token
            token_with_labels.append(token_)



        # evalu column
        eval_column = ""
        if is_label_provided:
            # eval_column = '<td bgcolor="{}" opacity:0.9; >Intent:{}<br>TERM:{} DEF:{}<br>Confidence(Slot):{:.1f}<br>Confidence(Sentence):{:.1f}</td>'.format(
                # color_eval,
                # d["results"]['intent_acc'],
                # d["results"]['slot_merged_TERM_f1'],
                # d["results"]['slot_merged_DEFINITION_f1'],
                # np.round(float(np.average(d['slot_conf'])) * 100.0,1),
                # np.round(float(d['intent_conf']) * 100.0,1)
            # )
            eval_column ='<td class="noborder" bgcolor="{}" opacity:0.9; >Confidence(Slot):{:.1f}<br>Confidence(Sentence):{:.1f}</td>'.format(
                color_eval,
                np.round(float(np.average(d['slot_conf'])) * 100.0,1),
                np.round(float(d['intent_conf']) * 100.0,1)
            )
        else:
            eval_column ='<td class="noborder" bgcolor="{}" opacity:0.9; >Confidence(Slot):{:.1f}<br>Confidence(Sentence):{:.1f}</td>'.format(
                color_eval,
                np.round(float(np.average(d['slot_conf'])) * 100.0,1),
                np.round(float(d['intent_conf']) * 100.0,1)
            )


        # make a row for visualization
        each_row = [
            '<td class="noborder"  {}>{} ({})</td>'.format(
                color_intent, d['intent_pred'], d['intent_label']
            ),
            format_word_importances(
                token_with_preds, (np.array(d['slot_conf'])-0.5) * 2.0),
            '<td class="noborder" >{}</td>'.format(d['rule']),
            eval_column

        ]

        if is_label_provided:
            each_row.insert(1,
                        "<td>{}</td>".format(' '.join(token_with_labels)))

        rows.append(each_row)

    if filename:
        rows = ["<tr>{}</tr>".format(" ".join(r)) for r in rows]

        head = [
            "<table width: 100%><tr>"
            "<th>Intent Predicted (True)</th>"
            "<th>Slot Predicted</th>"
            "<th>Confidence Score</th>"
            "<th>Rules</th>"
            "</tr>"
        ]

        if is_label_provided:
            head.insert(1,
                "<th>Slot True</th>"
                        )

        rows = head + rows

        dom.append("".join(rows))
        dom.append("</table>")

        html = HTML("".join(dom))
        with open(filename, 'w') as f:
            f.write(html.data)

    return rows


def format_special_tokens(token):
    if token.startswith("<") and token.endswith(">"):
        return "#" + token.strip("<>")
    return token

def _get_color(attr):
    # clip values to prevent CSS errors (Values should be from [-1,1])
    attr = max(-1, min(1, attr))
    if attr > 0:
        # hue = 120
        hue = 90
        sat = 75
        # lig = 100 - int(50 * attr)
        lig = 100 - int(47 * attr)
    else:
        hue = 0
        sat = 75
        # lig = 100 - int(-40 * attr)
        lig = 100 - int(-43 * attr)
    return "hsl({}, {}%, {}%)".format(hue, sat, lig)

def check_color(word):
    if "red" in word or "blue" in word:
        return True
    else:
        return False

def format_word_importances(words, importances):
    if importances is None or len(importances) == 0:
        return "<td></td>"
    assert len(words) <= len(importances)
    tags = ["<td>"]
    for word, importance in zip(words, importances[: len(words)]):
        # word = format_special_tokens(word)
        color = _get_color(importance)

        if check_color(word):
            unwrapped_tag = '<mark style="background-color: {color}; opacity:0.9; \
                        line-height:1.00"><font color="black"> {word}\
                        </font></mark>'.format(color=color, word=word)
        else:
            unwrapped_tag = '<mark style="opacity:0.9; \
                        line-height:1.00"><font color="black"> {word}\
                        </font></mark>'.format(word=word)

        # unwrapped_tag = '<mark style="background-color: {color}; opacity:0.9; \
                    # line-height:1.00"><font color="black"> {word}\
        #             </font></mark>'.format(color=color, word=word)
        tags.append(unwrapped_tag)
    tags.append("</td>")
    return "".join(tags)


