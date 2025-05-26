import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, MT5EncoderModel
from transformers import PretrainedConfig, PreTrainedModel
import json
from tqdm import tqdm
import os
import torch.nn as nn
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
input_trnalsation1 = '/data/home/ysu132/HTDM/qwen14b/humour_consistency.json'
input_translation2 = '/data/home/ysu132/Github/MAPS/MAPS-mt/model/alpaca/maps_kw.json'

with open(input_trnalsation1, 'r', encoding='utf-8') as csvtopic:
    filtered_translation1 = json.load(csvtopic)
with open(input_translation2, 'r', encoding='utf-8') as csvangle:
    filtered_translation2 = json.load(csvangle)


class MTRankerConfig(PretrainedConfig):
    
	def __init__(self, backbone='google/mt5-large', **kwargs):
            self.backbone = backbone
            super().__init__(**kwargs)
            
	

class MTRanker(PreTrainedModel):
    config_class = MTRankerConfig

    def __init__(self, config):
        super().__init__(config)
        self.encoder = MT5EncoderModel.from_pretrained(config.backbone)
        self.num_classes = 2
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, self.num_classes)
    
    def forward(self, input_ids, attention_mask):
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        seq_lengths = torch.sum(attention_mask, keepdim=True, dim=1)
        pooled_hidden_state = torch.sum(encoder_output * attention_mask.unsqueeze(-1).expand(-1, -1, self.encoder.config.hidden_size), dim=1)
        pooled_hidden_state /= seq_lengths
        prediction_logit = self.classifier(pooled_hidden_state)
        return prediction_logit

tokenizer = AutoTokenizer.from_pretrained('ibraheemmoosa/mt-ranker-xxl')
model = MTRanker.from_pretrained('ibraheemmoosa/mt-ranker-xxl')
model = nn.DataParallel(model)
model = model.cuda()

total = torch.tensor([[0.00,0.00]]).cuda()
l1, l2 = 0.0, 0.0
for i in tqdm(range(len(filtered_translation2))):
    text = "Source: "+filtered_translation1[i]["joke"]+" Translation 0: "+filtered_translation1[i]["translation"]+". Translation 1: "+filtered_translation2[i]["translation"]+"."

    tokenized = tokenizer(text, return_tensors="pt")

    input_ids = tokenized.input_ids.cuda()
    attn_mask = tokenized.attention_mask.cuda()
    generated_ids = model(input_ids, attn_mask)
    new = generated_ids.tolist()
    print(new)
    l1 += new[0][0]
    l2 += new[0][1]
    # total += generated_ids
    # # print(generated_ids)

print(l1,l2)
