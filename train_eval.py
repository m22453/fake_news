import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from tensorboardX import SummaryWriter
from tools.util import get_time_dif
# from apex import apm

from transformers.optimization import AdamW, get_linear_schedule_with_warmup


def bert_train(config, args, model, train_iter, valid_iter):
    start_time = time.time()
    model.train()
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')  # 设备
    # if args.fp16:
    #     model.half()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_iter) * args.epochs
    )
    # if args.fp16:

    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    total_batch = 0  # 当前进行的batch
    last_improve = 0
    flag = False  # 是否早停
    dev_best_loss = float("inf")
    best_acc_val = 0.0  # 最佳验证集准确率

    writer = SummaryWriter(log_dir=config['log_dir'] + '/' +
                           time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(args.epochs):
        print("epoch : {} / {}".format(epoch+1, args.epochs))

        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            # with amp.scaled_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.step()
            scheduler.step()
            if total_batch % 100 == 0:
                label_data = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(label_data, predic)
                dev_acc, dev_loss = evaluate(config, model, valid_iter)
                if best_acc_val < dev_acc:
                    dev_best_loss = dev_loss
                    best_acc_val = dev_acc
                    torch.save(model.state_dict(), config['save_path'])
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc,
                                 dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1

            if total_batch - last_improve > args.require_improvement:
                print("no optimization for a long time, early stop....")
                flag = True
                break
        if flag:
            break
    writer.close()
    # test(config, model, test_iter)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(
            labels_all, predict_all, target_names=config['class_list'], digits=2)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config['save_path']))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(
        config, model, test_iter, test=True)
    msg = '\\nn Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision\t Recall \t F1-Score...")

    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def train_bert(config, model, train_iter, valid_iter, test_iter):
    start_time = time.time()
    model.train()
