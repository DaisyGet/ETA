import transformers
from utils.data import ACOSProcessor
from utils.dataset import FeaturizedDataset
from utils.dataloader import FeaturizedDataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.optim as optim
import argparse
import torch
from tqdm import tqdm
import numpy as np
import re


def load_and_cache_examples(args, tokenizer, split='train'):
    processor = ACOSProcessor()
    data_dir = args.data_dir+'/'+args.area
    if split == 'train':
        examples = processor.get_train_examples(data_dir)
    elif split == 'dev':
        examples = processor.get_dev_examples(data_dir)
    else:
        examples = processor.get_test_examples(data_dir)

    dataset = FeaturizedDataset(examples, args, tokenizer)

    return dataset


def getQuadStructureABSA(out):
    quads = []
    getQs = "\( quad \(.*?\) \) \)"
    prog = re.compile(getQs)
    string = out
    qs = prog.findall(string)
    print('qs', qs)

    for q in qs:
        print('q', q)
        temp = []
        getQ = "\([^\(\)]*?\)"
        prog = re.compile(getQ)
        string = q
        quad = prog.findall(q)
        print('quad', quad)

        for p in quad:
            temp.extend([i.strip() for i in p.strip("(").strip(")").split(",")])

        quads.append(tuple(temp))

    return quads


def calculate_f1(pred, gold):
    gold_list = [getQuadStructureABSA(i) for i in gold]
    pred_list = [getQuadStructureABSA(i) for i in pred]

    quad_true_num = 0
    quad_pred_num = 0
    quad_gold_num = 0

    for p, g in zip(pred_list, gold_list):
        print('p:', p)
        print('q:', g)
        quad_gold_num += len(set(g))
        quad_pred_num += len(set(p))
        quad_true_num += len(set(g) & set(p))

    quad_p = quad_true_num / quad_pred_num if quad_true_num != 0 else 0
    quad_r = quad_true_num / quad_gold_num if quad_true_num != 0 else 0
    quad_f1 = (2 * quad_p * quad_r) / (quad_p + quad_r) if quad_true_num != 0 else 0

    print("Quad:")
    print("Gold Num:{0}  Pred Num:{1}  True Num:{2} :".format(quad_gold_num, quad_pred_num, quad_true_num))
    print("Precision:{0}".format(quad_p))
    print("Recall:{0}".format(quad_r))
    print("F1:{0}".format(quad_f1))

    return quad_f1, gold_list, pred_list


def evaluate(model, val_loader, tokenizer, device, last_ground_truth=None, last_input_sentence=None):

    model.eval()

    total_output_sentence = []
    total_ground_truth = []
    total_input = []

    val_iteration = tqdm(val_loader, desc="Iteration")
    for i, batch in enumerate(val_iteration):

        input_ids = batch['input_ids'].to(device)
        src_mask = batch['attention_mask'].to(device)
        target = batch['labels'].to(device)

        outputArgmax = model.generate(input_ids=input_ids, attention_mask=src_mask,
                                      max_length=256)

        s = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputArgmax]
        total_output_sentence.extend(s)

        if last_ground_truth == None:
            ground_truth = [tokenizer.decode(ids, skip_special_tokens=True) for ids in target]
            total_ground_truth.extend(ground_truth)

            input_sentence = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            total_input.extend(input_sentence)

        else:
            total_ground_truth = last_ground_truth
            total_input = last_input_sentence

    quad_f1, gold_list, pred_list = calculate_f1(total_output_sentence,
                                                 total_ground_truth)
    return quad_f1, gold_list, pred_list, total_ground_truth, total_input, total_output_sentence


def saveOutput(pred, gold, config):
    pred_out = pred
    gold_out = gold

    output_path = config.output_dir+'/'+config.model_name+'_result'
    f = open(output_path, "w")

    for p, g in zip(pred_out, gold_out):
        f.write("pred:\n")
        for sp in p:
            f.write(str(sp) + "\n")
        f.write("\n")

        f.write("gold:\n")
        for sg in g:
            f.write(str(sg) + "\n")
        f.write("\n\n\n")

    f.close()


def train(config):

    init_seed = config.seed
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)

    device = config.device
    first_output_dir = config.first_output_dir
    best_output_dir = config.best_output_dir

    t5_path = config.t5_path
    tokenizer = T5Tokenizer.from_pretrained(t5_path)

    train_dataset = load_and_cache_examples(config, tokenizer, split='train')
    dev_dataset = load_and_cache_examples(config, tokenizer, split='dev')
    test_dataset = load_and_cache_examples(config, tokenizer, split='test')

    train_batch_size = config.train_batch_size
    dev_batch_size = config.dev_batch_size
    test_batch_size = config.test_batch_size

    num_workers = config.num_workers

    train_loader = FeaturizedDataLoader(dataset=train_dataset, opt=config, batch_size=train_batch_size,
                                        shuffle=True,
                                        num_workers=num_workers)
    dev_loader = FeaturizedDataLoader(dataset=dev_dataset, opt=config, batch_size=dev_batch_size,
                                      shuffle=False,
                                      num_workers=num_workers)
    test_loader = FeaturizedDataLoader(dataset=test_dataset, opt=config, batch_size=test_batch_size,
                                       shuffle=False,
                                       num_workers=num_workers)

    model = T5ForConditionalGeneration.from_pretrained(t5_path)
    model = model.to(device)

    trainable_parameters = [
        param for param in model.parameters() if param.requires_grad
    ]
    lr = config.learning_rate
    optimizer = optim.AdamW(trainable_parameters, lr=lr, betas=(0.9, 0.98), eps=1e-08)

    bestF1 = 0
    epoch = config.num_train_epochs

    print("training.....")
    for epoch in range(epoch):
        model.train()

        print('epoch: ' + str(epoch))
        epochLoss = []
        train_iteration = tqdm(train_loader, desc="Iteration")
        for i, batch in enumerate(train_iteration):
            model.zero_grad()
            batch['labels'][batch['labels'][:, :] == tokenizer.pad_token_id] = -100

            input_ids = batch['input_ids'].to(device)
            src_mask = batch['attention_mask'].to(device)
            target = batch['labels'].to(device)
            decoder_attention_mask = batch['decoder_attention_mask'].to(device)

            output = model(input_ids=input_ids, attention_mask=src_mask, labels=target,
                           decoder_attention_mask=decoder_attention_mask)

            loss = output[0]
            print(loss)
            loss.backward()

            epochLoss.append(loss.item())
            optimizer.step()

        print("loss: {0}".format(sum(epochLoss) / len(epochLoss)))
        quadF1, gold, pred, total_ground_truth, total_input, total_output_sentence = evaluate(model, dev_loader, tokenizer, device)

        if epoch == 0:
            bestF1 = quadF1
            config = model.config
            model.save_pretrained(first_output_dir)
            config.save_pretrained(first_output_dir)

        elif bestF1 < quadF1:
            bestF1 = quadF1
            print("\ncurrently best, test model on testset:")
            quadF1_test, gold_test, pred_test, total_ground_truth_test, total_input_test, total_output_sentence_test = \
                evaluate(model, test_loader, tokenizer, device)
            saveOutput(pred_test, gold_test, config)
            config = model.config
            model.save_pretrained(best_output_dir)
            config.save_pretrained(best_output_dir)


        print("current best Quad F1: {0}".format(bestF1))


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--area", default="res")
    parser.add_argument("--t5_path", default="t5-base",type=str)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cpu",type=str,help="device id")
    parser.add_argument("--model_name",default="t5-base",type=str)
    parser.add_argument("--max_length",default=512, type=int)
    parser.add_argument("--max_target_length", default=156, type=int)
    parser.add_argument("--num_train_epochs",default=30, type=int)
    parser.add_argument("--train_batch_size",default=8, type=int)
    parser.add_argument("--dev_batch_size",default=16, type=int)
    parser.add_argument("--test_batch_size",default=16, type=int)
    parser.add_argument("--learning_rate",default=5e-5, type=float)
    parser.add_argument("--num_workers", default=0,type=int)
    parser.add_argument("--save_model", default=True, type=bool)
    parser.add_argument("--first_output_dir", default='./output/first_output_dir', type=str)
    parser.add_argument("--best_output_dir", default='./output/best_output_dir', type=str)

    args = parser.parse_args()

    return args


def main():

    args = args_parser()
    train(args)


if __name__ == "__main__":
    main()

