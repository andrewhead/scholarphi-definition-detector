import torch
import torch.nn as nn
# from transformers import BERT_PRETRAINED_MODEL_ARCHIVE_LIST , BertPreTrainedModel, BertModel, BertConfig
from transformers import PreTrainedModel
from torchcrf import CRF
from .module import IntentClassifier, SlotClassifier, Pooler

class Model(): #PreTrainedModel):
    # config_class = BertConfig
    # pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_LIST
    # base_model_prefix = "bert"

    def __init__(self, config, args, intent_label_lst, slot_label_lst, pos_label_lst, model):
        # super(JointModel, self).__init__(config)
        self.args = args

        self.intent_label_lst = intent_label_lst
        self.slot_label_lst = slot_label_lst

        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)

        self.pos_label_lst = slot_label_lst
        self.num_pos_labels = len(pos_label_lst)

        # model_type = config.model_type
        # self.model = None #BertModel(config=config)  # Load pretrained bert
        self.model = model

        hidden_size = config.hidden_size

        #TODO pos_emb = 50 -< variables
        if args.use_pos:
            pos_dim = 50
            hidden_size += pos_dim
            self.pos_emb = nn.Embedding(self.num_pos_labels, pos_dim) if pos_dim > 0 else None

        self.custom_pooler = Pooler(hidden_size=hidden_size)
        self.intent_classifier = IntentClassifier(hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(hidden_size, self.num_slot_labels, args.dropout_rate)


        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids, pos_label_ids):
        outputs = self.model(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]


        if self.args.use_pos:
            # torch.cat([word_emb,pos_emb], dim=2)
            pos_output = self.pos_emb(pos_label_ids)
            sequence_output = torch.cat([sequence_output, pos_output], dim=2)
            pooled_output = self.custom_pooler(sequence_output)

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
