import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm._tqdm import tqdm
import os
import random
import gc
from scipy.special import softmax
import spacy
from spacy.attrs import ORTH
# from spacy.tokenizer import Tokenizer

from transformers import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer

# set random seeds
torch.backends.cudnn.deterministic = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

nlp = spacy.load('en')
nlp.tokenizer.add_special_case("<pre>", [{ORTH: "<pre>"}])
nlp.tokenizer.add_special_case("</pre>", [{ORTH: "</pre>"}])
nlp.tokenizer.add_special_case("<event>", [{ORTH: "</event>"}])
nlp.tokenizer.add_special_case("</event>", [{ORTH: "</event>"}])
# nlp.tokenizer = Tokenizer(nlp.vocab)

# Model locations
CLF_MODEL = "models/PrecondCLFModel.pt"
ES_CTX_0 = "models/EventSampler_Ctx_0.pt"
ES_CTX_2 = "models/EventSampler_Ctx_2.pt"
ES_CTX_5 = "models/EventSampler_Ctx_5.pt"


# Model for precondition classifer (reranking purpose)
class Model(nn.Module):
    def __init__(self, tokenizer, encoder, embedding_dim, hidden_dim, n_class):
        super(Model, self).__init__()
        self.use_cuda = True if torch.cuda.is_available() else False
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.tokenizer = tokenizer
        self.encoder = encoder

        self.output = nn.Linear(self.embedding_dim*2, n_class)
        self.softmax = nn.LogSoftmax(dim=-1)

    def get_var(self, tensor):
        if self.use_cuda:
            return Variable(tensor.cuda())
        else:
            return Variable(tensor)

    def encode(self, indexed_tokens):
        max_len = max([len(ids) for ids in indexed_tokens]) + 2
        tokens_tensor = []
        token_type_ids = []
        attention_mask = []
        for instance in indexed_tokens:
            encoded_input = self.tokenizer.prepare_for_model(
                    instance, max_length=max_len, pad_to_max_length=True)
            tokens_tensor.append(encoded_input['input_ids'])
            token_type_ids.append(encoded_input['token_type_ids'])
            attention_mask.append(encoded_input['attention_mask'])

        tokens_tensor = torch.tensor(tokens_tensor)
        token_type_ids = torch.tensor(token_type_ids)
        attention_mask = torch.tensor(attention_mask)

        if self.use_cuda:
            tokens_tensor = tokens_tensor.cuda()
            token_type_ids = token_type_ids.cuda()
            attention_mask = attention_mask.cuda()

        return self.encoder(
                input_ids=tokens_tensor,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask)[0]

    def forward(self, sentences, relation):

        sent_output = self.encode(sentences)

        batch_size, seq_len, dim = sent_output.size()

        # rel_repr = []
        rel_repr = None
        for para, rels in zip(sent_output, relation):
            e1, e2 = rels
            e1_idx = torch.arange(e1[0], e1[1])
            e2_idx = torch.arange(e2[0], e2[1])
            e1_repr = torch.sum(
                    para.index_select(0, self.get_var(e1_idx)),
                    dim=0)
            e2_repr = torch.sum(
                    para.index_select(0, self.get_var(e2_idx)),
                    dim=0)

            e_repr = torch.cat((e1_repr, e2_repr)).unsqueeze(0)
            if rel_repr is None:
                rel_repr = e_repr
            else:
                rel_repr = torch.cat((rel_repr, e_repr), dim=0)

        logits = self.output(rel_repr)

        return self.softmax(logits)


def load_data(files, max_len=50, context=-1, eos='<eos>'):
    # If context is set as -1
    # data is loaded for Precondition Generator
    # Or (context >= 0), data is loaded for Event Sampler
    # Default value: 0
    dataset = {'train': {}, 'dev': {}}
    for set_info, f in files.items():
        with open(f) as fin:
            input_data = []
            target = []
            generation_seeds = []
            line_tqdm = tqdm(fin)
            for line in line_tqdm:
                row = line.strip().split("\t")
                if len(row[0].split()) > max_len:
                    continue
                if "<event>" not in row[0]:
                    continue

                precond = row[1].split("<pre> ")[1].split(" </pre>")[0]

                if context != -1:
                    fcontext = row[0].split(" <event> ")
                    if len(fcontext) != 2:
                        continue
                    bcontext = fcontext[1].split(" </event> ")
                    if len(bcontext) != 2:
                        continue
                    event = bcontext[0]
                    bcontext = bcontext[1].split()
                    fcont = []
                    fcontext = fcontext[0].split()[::-1]
                    for i in range(context):
                        if i > len(fcontext)-1 or fcontext[i] == '[BLANK]':
                            break
                        else:
                            fcont.append(fcontext[i])
                    bcont = []
                    for i in range(context):
                        if i > len(bcontext)-1 or bcontext[i] == '[BLANK]':
                            break
                        else:
                            bcont.append(bcontext[i])

                    if context != 0:
                        event = fcont[::-1] + ['<event>'] \
                            + [event] + ['</event>'] + bcont
                    else:
                        event = [event]
                    input_data.append(event + ['<sep>'] + [precond] + [eos])
                    generation_seeds.append(event + ['<sep>'])
                    target.append([precond] + [eos])

                else:
                    input_data.append(
                            row[0].split()
                            + ['<E>', precond, '<sep>']
                            + row[1].split() + [eos]
                    )
                    generation_seeds.append(
                            row[0].split()
                            + ['<E>', precond, '<sep>']
                    )
                    target.append(row[1].split() + [eos])

            dataset[set_info]['input'] = input_data
            dataset[set_info]['target'] = target
            dataset[set_info]['seed'] = generation_seeds

    return dataset


def prepare(dataset, tokenizer):
    data_input = {}
    gen_seed = {}
    target = {}
    for set_info, data in dataset.items():
        data_input[set_info] = []
        gen_seed[set_info] = []
        target[set_info] = []
        for input_text in data['input']:
            data_input[set_info].append(tokenizer.encode(" ".join(input_text)))
        for input_text in data['seed']:
            gen_seed[set_info].append(tokenizer.encode(" ".join(input_text)))
        for input_text in data['target']:
            target[set_info].append(tokenizer.encode(" ".join(input_text)))

    return data_input, gen_seed, target


def clf_prepare(data, tokenizer):
    paragraphs = []
    relations = []
    for rid, row in enumerate(data):

        sent = row['sent'].split()

        tokens = tokenizer.tokenize(" ".join(sent))
        if len(tokens) > 512:
            continue
        i, j, start_idx = 0, 0, 0
        new_idxs = []
        text_buf = []
        while i < len(sent):
            if sent[i] == " "*len(sent[i]):
                i += 1
                new_idxs.append(0)
            else:
                break
        while i < len(sent) and j < len(tokens):
            text_buf.append(tokens[j])
            if tokenizer.convert_tokens_to_string(text_buf) \
               == tokenizer.convert_tokens_to_string(
                       tokenizer.tokenize(sent[i])
            ):
                i += 1
                new_idxs.append(start_idx)
                start_idx = j+1
                text_buf = []
            j += 1
        new_idxs.append(len(tokens))

        paragraphs.append(tokenizer.convert_tokens_to_ids(tokens))

        relations.append(
                [[new_idxs[ii]+1 for ii in row['source']['idx']],
                    [new_idxs[ii]+1 for ii in row['target']['idx']]]
        )

    return paragraphs, relations


def get_input_for_model(tokenizer, indexed_tokens, event_lens):
    lengths = [len(ids) for ids in indexed_tokens]
    max_len = min(max(lengths), 1024)
    tokens_tensor = []
    token_type_ids = []
    attention_mask = []
    targets = []
    for instance, l in zip(indexed_tokens, event_lens):
        # This returns:
        #       padded input
        #       token_type_ids
        #       attention mask
        encoded_input = tokenizer.prepare_for_model(
            instance, max_length=max_len, pad_to_max_length=True)

        tokens_tensor.append(encoded_input['input_ids'])
        token_type_ids.append(encoded_input['token_type_ids'])
        attention_mask.append(encoded_input['attention_mask'])

        # Masked out token labels before the <sep> (inclusive)
        # and all <PAD> tokens
        # This makes loss calculated only on the precondition part
        # (after <sep> before <PAD>)
        targets.append(
                [-100]*l
                + instance[l:]
                + [-100]*(max_len - len(instance))
        )

    tokens_tensor = torch.tensor(tokens_tensor)
    token_type_ids = torch.tensor(token_type_ids)
    attention_mask = torch.tensor(attention_mask)
    targets = torch.tensor(targets)

    if torch.cuda.is_available():
        tokens_tensor = tokens_tensor.cuda()
        token_type_ids = token_type_ids.cuda()
        attention_mask = attention_mask.cuda()
        targets = targets.cuda()

    return (tokens_tensor, attention_mask, token_type_ids, targets)


def finetuning(args):

    torch.cuda.set_device(args.device)
    print("Load Data")
    print(args.train_data, args.dev_data)
    files = {'train': args.train_data, 'dev': args.dev_data}

    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, pad_token='<PAD>')
    # Add new tokens
    # <sep>: separator, a cue for model to generate
    # [BLANK]: a masked-out precondition part
    # <pre> ... </pre>: precondition markers
    # <event> ... </event>: target event markers
    # <E>: precondition candidate marker
    tokenizer.add_tokens(['<sep>', '[BLANK]',
                          '<pre>', '</pre>', '<event>', '</event>', '<E>'])

    dataset = load_data(
            files,
            max_len=args.max_len,
            context=args.context,
            eos=tokenizer.eos_token
    )
    if args.load_model is not None:
        model = torch.load(args.load_model, map_location=f'cuda:{args.device}')

    else:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        # Resize model according to the updated vocab
        model.resize_token_embeddings(len(tokenizer))

    if torch.cuda.is_available():
        model.cuda()

    # tokenize and get generatin seeds from data
    data_input, gen_seed, target = prepare(dataset, tokenizer)

    save_model_path = os.path.join(args.save_model_path, args.experiment)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    # add parameters to optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {
                'params':
                [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            },
            {
                'params':
                [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
    optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=args.learning_rate,
                    eps=1e-8
                )

    n_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("#parameters: {}".format(n_params))

    N = len(data_input['train'])
    print(N//args.batch_size)
    best_dev_loss = 9999

    for epoch in range(1, args.epochs+1):
        print("Epoch {}:".format(epoch))
        batch_idxs = np.random.permutation(N//args.batch_size+1)
        line_tqdm = tqdm(batch_idxs, dynamic_ncols=True)
        total_loss = []

        for i, batch_idx in enumerate(line_tqdm):
            model.train()
            enc_input = data_input['train'][batch_idx*args.batch_size:min((batch_idx+1)*args.batch_size, N)]
            tmp = gen_seed['train'][batch_idx*args.batch_size:min((batch_idx+1)*args.batch_size, N)]
            event_lens = [len(s) for s in tmp]

            # get adjusted input for training
            input_feed = get_input_for_model(tokenizer, enc_input, event_lens)

            model.zero_grad()

            # train the model
            loss = model(
                    input_ids=input_feed[0],
                    attention_mask=input_feed[1],
                    token_type_ids=input_feed[2],
                    labels=input_feed[3]
                    )[0]

            loss.backward()
            total_loss.append(loss.data.cpu().numpy().tolist())
            optimizer.step()
            gc.collect()
            torch.cuda.empty_cache()

            # print("Loss: {}".format(sum(total_loss)/len(total_loss)))
            if i != 0 and (i % 3000 == 0 or i == len(batch_idxs)-1):
                model.eval()
                with torch.no_grad():
                    # example generation
                    for d, t in zip(gen_seed['dev'][:5], target['dev'][:5]):
                        test = torch.tensor(d).unsqueeze(0).cuda()
                        sent = model.generate(
                            input_ids=test,
                            max_length=100,
                            top_p=0.95,
                            repetition_penalty=1.2)
                        print("Seed: ", tokenizer.decode(d))
                        text = tokenizer.decode(sent[0][len(d):])
                        text = text.split(tokenizer.eos_token)[0]
                        print("Generated: ", text)
                        print("Referece: ", tokenizer.decode(t))

                    # check loss on dev set
                    for set_info in ['dev']:
                        NN = len(data_input[set_info])
                        total_loss = []
                        for idx in range(NN//args.batch_size):
                            enc_input = data_input[set_info][idx*args.batch_size:min((idx+1)*args.batch_size, NN)]
                            tmp = gen_seed[set_info][idx*args.batch_size:min((idx+1)*args.batch_size, NN)]
                            event_lens = [len(s) for s in tmp]

                            input_feed = get_input_for_model(
                                            tokenizer,
                                            enc_input,
                                            event_lens
                                        )

                            loss = model(
                                input_ids=input_feed[0],
                                attention_mask=input_feed[1],
                                token_type_ids=input_feed[2],
                                labels=input_feed[3]
                                )[0]

                            total_loss.append(loss.data.cpu().numpy().tolist())

                        loss = sum(total_loss) / len(total_loss)
                        print("Test on {} set:".format(set_info))
                        print("\tLoss: {}".format(loss))
                        if set_info == 'dev':
                            if best_dev_loss > loss:
                                best_dev_loss = loss
                                torch.save(
                                        model,
                                        os.path.join(
                                            save_model_path,
                                            "DevBest.pt"
                                        )
                                )

    return


def topk_generate(model, context, k=10, max_len=50):
    logits = model(
                input_ids=context
            )[0]

    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    scores, idxs = torch.topk(probs, k)
    scores = scores.cpu().numpy().tolist()

    context = context.repeat(k, 1)
    context = torch.cat((context, idxs.view(k, 1)), dim=-1)

    for i in range(max_len-1):
        logits = model(
                    input_ids=context
                )[0]

        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        next_tokens = torch.argmax(probs, dim=1, keepdim=True)

        context = torch.cat([context, next_tokens], dim=1)

    return scores, context


def get_event(data, context=0):

    fcontext = data.split(" <event> ")
    bcontext = fcontext[1].split(" </event> ")
    event = bcontext[0]
    bcontext = bcontext[1].split()
    fcont = []
    fcontext = fcontext[0].split()[::-1]
    for i in range(context):
        if i > len(fcontext)-1 or fcontext[i] == '[BLANK]':
            break
        else:
            fcont.append(fcontext[i])
    bcont = []
    for i in range(context):
        if i > len(bcontext)-1 or bcontext[i] == '[BLANK]':
            break
        else:
            bcont.append(bcontext[i])

    event = fcont[::-1] + ['<event>'] + [event] + ['</event>'] + bcont
    if len(event) == 3:
        event = [event[1]]

    event += ['<sep>']

    return " ".join(event)


def clf_test(model, test_para, test_relation, thr=0.5):
    model.eval()

    score = model(test_para, test_relation)
    score, idxs = torch.max(F.softmax(score, dim=-1), dim=-1)
    pred = idxs.cpu().tolist()
    score = score.cpu().tolist()

    for i in range(len(pred)):
        if pred[i] == 1:
            if score[i] < thr:
                pred[i] = 0
                score[i] = 1-score[i]

    # Apply softmax to scores to make them
    # in the same range of scores from Event Sampler
    out = (pred, softmax(score))

    return out


def precond_rerank(generated_precond, alpha=1.):
    alpha = 0.99
    rerank_score = [
            alpha*data['precond_score']
            + (1-alpha)*data['event_score']
            for data in generated_precond]
    sorted_idxs = np.argsort(rerank_score)[::-1]

    sorted_precond = []
    for idx in sorted_idxs:
        generated_precond[idx]['rerank_score'] = rerank_score[idx]
        sorted_precond.append(generated_precond[idx])

    return sorted_precond


def similarity_filter(clf_tokenizer, clf_model, sorted_precond, k=10):

    paragraphs = []
    for data in sorted_precond:
        paragraphs.append(clf_tokenizer.encode(data['precondition_text']))

    sent_encoding = clf_model.encode(paragraphs)
    cls_tokens = sent_encoding[:, 0]

    magnitude = torch.sqrt(
                    torch.sum(
                        torch.mul(
                            cls_tokens,
                            cls_tokens
                        ),
                        dim=-1,
                        keepdim=True
                    )
                )
    cls_tokens /= magnitude
    cos_sim = torch.matmul(cls_tokens, cls_tokens.transpose(1, 0))
    cos_sim = cos_sim.cpu().numpy()

    cos_mean = np.mean(cos_sim)
    cos_std = np.std(cos_sim)
    thr = cos_mean + cos_std

    result = []
    flag = [1]*len(cos_sim)
    for i in range(len(cos_sim)):
        if flag[i] == 0:
            continue
        else:
            result.append(sorted_precond[i])
            if len(result) == k:
                break
            # Filter out similar preconditions
            for j in range(i+1, len(cos_sim)):
                if flag[j] and cos_sim[i, j] >= thr:
                    flag[j] = 0

    return result


def generation(args):

    torch.cuda.set_device(args.device)

    pretrain_model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(
                    pretrain_model_name,
                    pad_token='<PAD>'
                )
    tokenizer.add_tokens(['<sep>', '[BLANK]',
                          '<pre>', '</pre>', '<event>', '</event>', '<E>'])

    model = torch.load(args.load_model, map_location=f'cuda:{args.device}')
    model.eval()
    if args.context == 0:
        event_sampler = torch.load(
                            ES_CTX_0,
                            map_location=f'cuda:{args.device}'
                        )
    elif args.context == 3:
        event_sampler = torch.load(
                            ES_CTX_2,
                            map_location=f'cuda:{args.device}'
                        )
    elif args.context == 5:
        event_sampler = torch.load(
                            ES_CTX_5,
                            map_location=f'cuda:{args.device}'
                        )

    clf_model_name = 'bert-base-cased'
    clf_tokenizer = BertTokenizer.from_pretrained(
                        clf_model_name,
                        pad_token='<PAD>'
                    )
    clf_model = torch.load(CLF_MODEL)

    if torch.cuda.is_available():
        model.cuda()
        event_sampler.cuda()
        clf_model.cuda()

    with torch.no_grad():
        if args.val:
            with open("data/val_multi_auto.txt", "r") as fin, \
                    open(f"val_{model_name}_c={args.context}.txt", "w") \
                    as fout:
                header = ["Target Event", "Generated Precondition"]
                fout.write("\t".join(header) + "\n")
                for lid, line in enumerate(fin):
                    generated_precond = []
                    row = line.strip().split("\t")

                    print("Target Event: ", row[0])

                    event = get_event(row[0], context=args.context)
                    print(event)
                    event_ids = tokenizer.encode(event)
                    test = torch.tensor(event_ids).unsqueeze(0)
                    if torch.cuda.is_available():
                        test = test.cuda()
                    scores, pre_events = topk_generate(
                                            event_sampler,
                                            test,
                                            k=100
                                        )

                    line_tqdm = tqdm(
                                    enumerate(zip(pre_events, scores[0])),
                                    dynamic_ncols=True
                                )
                    for i, (e, score) in line_tqdm:
                        e_text = tokenizer.decode(e[len(event_ids):])
                        e_text = e_text.split(tokenizer.eos_token)[0]
                        token_ids = tokenizer.encode(
                                        row[0]
                                        + f" <E> {e_text} <sep>"
                                    )
                        gen_input = torch.tensor(token_ids).unsqueeze(0)
                        if torch.cuda.is_available():
                            gen_input = gen_input.cuda()

                        sent = model.generate(
                            input_ids=gen_input,
                            max_length=150,
                            top_p=0.95,
                            repetition_penalty=1.2)
                        print("Seed: ", tokenizer.decode(token_ids))
                        text = tokenizer.decode(sent[0][len(token_ids):])
                        text = text.split(tokenizer.eos_token)[0]
                        print("Generated: ", text)

                        sent = row[0].replace("[BLANK]", text)
                        doc = nlp(sent)
                        sent = [t.text for t in doc if t.text != " "]

                        doc = nlp(text)
                        precond = [
                                    t.text for t in doc if t.text != " "
                                    and t.text != "<pre>"
                                    and t.text != "</pre>"
                                ]

                        sent_list = []
                        pre_idx, post_idx = [], []
                        for tid, t in enumerate(sent):
                            if t in ["<pre>", "<event>", "</pre>", "</event>"]:
                                length = len(sent_list)
                                if t == "<pre>" or t == "</pre>":
                                    pre_idx.append(length)
                                else:
                                    post_idx.append(length)
                            else:
                                sent_list.append(t)

                        data = {}
                        pre = " ".join(sent_list[pre_idx[0]:pre_idx[1]])
                        post = " ".join(sent_list[post_idx[0]:post_idx[1]])
                        event = event.split()[0]
                        data['sent_id'] = f"{event}{lid:03d}_{i:03d}"
                        data['source'] = {'event': pre, 'idx': pre_idx}
                        data['target'] = {'event': post, 'idx': post_idx}
                        data['label'] = 0
                        data['event_score'] = score
                        data['sent'] = " ".join(sent_list)
                        data['precondition_text'] = " ".join(precond)

                        generated_precond.append(data)

                    paragraphs, relations = clf_prepare(
                                                generated_precond,
                                                clf_tokenizer
                                            )

                    pred, scores = clf_test(clf_model, paragraphs, relations)
                    for data, p, s in zip(generated_precond, pred, scores):
                        data['prediction'] = p
                        data['precond_score'] = s

                    sorted_precond = precond_rerank(
                                        generated_precond,
                                        alpha=0.99
                                    )
                    filtered_precond = similarity_filter(
                                            clf_tokenizer,
                                            clf_model,
                                            sorted_precond,
                                            k=10
                                        )

                    for data in filtered_precond:
                        print(data)
                        fout.write(json.dumps(data) + "\n")

        else:
            with open("data/test_gen_peko_blank_target.txt", "r") as fin, \
                    open(f"DiP_c={args.context}_eventsampling.txt", "w") as eout, \
                    open(f"DiP_c={args.context}_reranking.txt", "w") as fout, \
                    open(f"DiP_c={args.context}_reranking_filtering.txt", "w") as ffout:
                # header = ["Target Event", "Reference", "Generated Precondition"]
                # fout.write("\t".join(header) + "\n")
                inputs = []
                for line in fin:
                    row = line.strip().split("\t")
                    inputs.append(row)

                # Generate preconditions from 500 examples
                idxs = np.random.permutation(len(inputs))[:5]
                for lid, idx in enumerate(idxs):
                    generated_precond = []
                    row = inputs[idx]

                    print(f"{lid}\t Target Event: {row[0]}")

                    event = get_event(row[0], context=args.context)
                    print(f"\t Event trigger with context: {event}")
                    event_ids = tokenizer.encode(event)
                    test = torch.tensor(event_ids).unsqueeze(0)
                    if torch.cuda.is_available():
                        test = test.cuda()

                    # Generate TopK (K = 100) precondition events
                    # using Event Sampler
                    scores, pre_events = topk_generate(
                                            event_sampler,
                                            test,
                                            k=100
                                        )

                    line_tqdm = tqdm(
                                    enumerate(zip(pre_events, scores[0])),
                                    dynamic_ncols=True
                                )
                    for i, (e, score) in line_tqdm:
                        e_text = tokenizer.decode(e[len(event_ids):])
                        e_text = e_text.split(tokenizer.eos_token)[0]
                        token_ids = tokenizer.encode(
                                        row[0]
                                        + f" <E> {e_text} <sep>"
                                    )
                        gen_input = torch.tensor(token_ids).unsqueeze(0)
                        if torch.cuda.is_available():
                            gen_input = gen_input.cuda()

                        # Precondition Generation
                        sent = model.generate(
                            input_ids=gen_input,
                            max_length=150,
                            top_p=0.95,
                            repetition_penalty=1.2)
                        text = tokenizer.decode(sent[0][len(token_ids):])
                        text = text.split(tokenizer.eos_token)[0]

                        sent = row[0].replace("[BLANK]", text)
                        doc = nlp(sent)
                        sent = [t.text for t in doc if t.text != " "]
                        sent_list = []

                        doc = nlp(text)
                        precond = [
                                    t.text for t in doc if t.text != " "
                                    and t.text != "<pre>"
                                    and t.text != "</pre>"
                                ]

                        pre_idx, post_idx = [], []
                        for tid, t in enumerate(sent):
                            if t in ["<pre>", "<event>", "</pre>", "</event>"]:
                                length = len(sent_list)
                                if t == "<pre>" or t == "</pre>":
                                    pre_idx.append(length)
                                else:
                                    post_idx.append(length)
                            else:
                                sent_list.append(t)

                        data = {}
                        if len(pre_idx) < 2 or len(post_idx) < 2:
                            continue
                        pre = " ".join(sent_list[pre_idx[0]:pre_idx[1]])
                        post = " ".join(sent_list[post_idx[0]:post_idx[1]])
                        event = event.split()[0]
                        data['sent_id'] = f"{event}{lid:03d}_{i:03d}"
                        data['source'] = {'event': pre, 'idx': pre_idx}
                        data['target'] = {'event': post, 'idx': post_idx}
                        data['label'] = 0
                        data['event_score'] = score
                        data['sent'] = " ".join(sent_list)
                        data['precondition_text'] = " ".join(precond)

                        generated_precond.append(data)

                    # Top 10 preconditions after
                    # event sampling + candidate generation
                    for data in generated_precond[:10]:
                        eout.write(json.dumps(data) + "\n")

                    paragraphs, relations = clf_prepare(
                                                generated_precond,
                                                clf_tokenizer
                                            )

                    pred, scores = clf_test(clf_model, paragraphs, relations)
                    for data, p, s in zip(generated_precond, pred, scores):
                        data['prediction'] = p
                        data['precond_score'] = s

                    # Precondition Reranking
                    sorted_precond = precond_rerank(
                                        generated_precond,
                                        alpha=0.99
                                    )

                    # Top 10 preconditions after reranking
                    for data in sorted_precond[:10]:
                        fout.write(json.dumps(data) + "\n")

                    # Similarity Filter
                    filtered_precond = similarity_filter(
                                            clf_tokenizer,
                                            clf_model,
                                            sorted_precond,
                                            k=10
                                        )

                    # Top 10 preconditions after
                    # filtering based on similarity score
                    for data in filtered_precond:
                        ffout.write(json.dumps(data) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data', type=str, default="data/train_gen_peko_blank_target.txt")
    parser.add_argument('--dev_data', type=str, default="data/dev_gen_peko_blank_target.txt")
    parser.add_argument('--test_data', type=str, default="../")

    parser.add_argument('-ep', '--epochs', type=int, default=100)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)

    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('-bin', '--save_model_path', type=str, default='data/PrecondGen/')
    parser.add_argument('-ex', '--experiment', type=str, default='test')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-c', '--context', type=int, default=0)
    parser.add_argument('--val', action='store_true')
    parser.add_argument('-d', '--device', type=int, default=0)
    parser.add_argument('--max_len', type=int, default=100)

    args = parser.parse_args()

    if args.test:
        generation(args)
    else:
        finetuning(args)
