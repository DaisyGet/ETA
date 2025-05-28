import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils.autoregressive_sampling import autoregressive_sampling
from utils.contrastive_sampling import contrastive_sampling
import json
import re


def getQuadStructureABSA(out):
    quads = []
    getQs = "\( quad \(.*?\) \) \)"
    prog = re.compile(getQs)
    string = out
    qs = prog.findall(string)

    for q in qs:
        temp = []
        getQ = "\([^\(\)]*?\)"
        prog = re.compile(getQ)
        string = q
        quad = prog.findall(q)

        for p in quad:
            temp.extend([i.strip() for i in p.strip("(").strip(")").split(",")])

        quads.append(tuple(temp))

    return quads


test_data_file = './data/res/test.json'
result_file = './result/test_contrastive_0.5.json'

best_output_dir = './outputs/best_output_dir'
first_output_dir = './outputs/first_output_dir'

model_expert = best_output_dir
model_student = first_output_dir

num_tokens = 128
a = 0.5
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = T5Tokenizer.from_pretrained('t5-base')
tokenizer.pad_token = tokenizer.eos_token
eos = tokenizer.eos_token
# tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

load_in_8bit = False  

model = T5ForConditionalGeneration.from_pretrained(model_expert)
model = model.to(torch_device)
model_s = T5ForConditionalGeneration.from_pretrained(model_student)
model_s = model_s.to(torch_device)

model.config.use_cache = True
model_s.config.use_cache = True

test_datas = []
gold_labels = []
with open(test_data_file, 'r', encoding='utf-8') as f:
    datas = json.load(f)

for data in datas:
    sentence = data['sentence']
    lable = data['structured']
    test_datas.append(sentence)
    gold_labels.append(lable)

results = []
total = 0
right = 0
right_c = 0
for sentence, label in zip(test_datas, gold_labels):

    input_ids = tokenizer.encode(sentence, return_tensors='pt').to(torch_device)
    attention_mask = tokenizer.encode_plus(sentence, return_tensors='pt')["attention_mask"].to(torch_device)

    output_autoregressive_1 = autoregressive_sampling(input_ids, attention_mask, model, num_tokens, eos_token_id=1)
    generated_autoregressive_1 = tokenizer.decode(output_autoregressive_1[0], skip_special_tokens=True)

    output_contrastive_1 = contrastive_sampling(input_ids, attention_mask, model, model_s, num_tokens, a=a, eos_token_id=1)
    generated_text_contrastive_1 = tokenizer.decode(output_contrastive_1[0], skip_special_tokens=True)

    out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=256)[0]

    pred = generated_autoregressive_1
    pred_c = generated_text_contrastive_1
    pred_g = tokenizer.decode(out, skip_special_tokens=True)

    result = dict()
    result['sentence'] = sentence
    result['pred'] = pred
    result['pred_c'] = pred_c
    result['label'] = label
    result['pred_g'] = pred_g
    results.append(result)

with open(result_file, 'w') as f1:
    json.dump(results, f1)

quad_true_num = 0
quad_pred_num = 0
quad_gold_num = 0

quad_true_num_c = 0
quad_pred_num_c = 0
quad_gold_num_c = 0

for data in datas:
    label = data['label']
    pred = data['pred']
    pred_c = data['pred_c']

    label = getQuadStructureABSA(label)
    pred = getQuadStructureABSA(pred)
    pred_c = getQuadStructureABSA(pred_c)

    p = pred
    p_c = pred_c
    g = label

    quad_gold_num += len(set(g))
    quad_pred_num += len(set(p))
    quad_true_num += len(set(g) & set(p))

    quad_gold_num_c += len(set(g))
    quad_pred_num_c += len(set(p_c))
    quad_true_num_c += len(set(g) & set(p_c))


quad_p = quad_true_num / quad_pred_num if quad_true_num != 0 else 0
quad_r = quad_true_num / quad_gold_num if quad_true_num != 0 else 0
quad_f1 = (2 * quad_p * quad_r) / (quad_p + quad_r) if quad_true_num != 0 else 0

quad_p_c = quad_true_num_c / quad_pred_num_c if quad_true_num_c != 0 else 0
quad_r_c = quad_true_num_c / quad_gold_num_c if quad_true_num_c != 0 else 0
quad_f1_c = (2 * quad_p_c * quad_r_c) / (quad_p_c + quad_r_c) if quad_true_num_c != 0 else 0

print("autoregressive_sampling:")
print("Gold Num:{0}  Pred Num:{1}  True Num:{2} :".format(quad_gold_num, quad_pred_num, quad_true_num))
print("Precision:{0}".format(quad_p))
print("Recall:{0}".format(quad_r))
print("F1:{0}".format(quad_f1))

print("contrastive-decoding:")
print("Gold Num:{0}  Pred Num:{1}  True Num:{2} :".format(quad_gold_num_c, quad_pred_num_c, quad_true_num_c))
print("Precision:{0}".format(quad_p_c))
print("Recall:{0}".format(quad_r_c))
print("F1:{0}".format(quad_f1_c))