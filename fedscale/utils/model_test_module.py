# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# libs from fedscale
import fedscale.cloud.config_parser as parser
from fedscale.dataloaders.nlp import mask_tokens

if parser.args.task == "detection":
    import numpy as np
    import torch.nn as nn
    import torch.optim as optim
    from torch.autograd import Variable

    from fedscale.dataloaders.rcnn.lib.datasets.pascal_voc import readClass
    from fedscale.dataloaders.rcnn.lib.model.roi_layers import nms
    from fedscale.dataloaders.rcnn.lib.model.rpn.bbox_transform import (
        bbox_transform_inv, clip_boxes)
    from fedscale.dataloaders.rcnn.lib.model.utils.config import cfg
    from fedscale.dataloaders.rcnn.lib.roi_data_layer.roidb import \
        combined_roidb
elif parser.args.task == 'voice':
    from fedscale.dataloaders.decoder import GreedyDecoder


def cal_accuracy(targets, outputs):
    temp_acc = 0
    temp_all_or_false = 0

    temp_len = 0

    for idx, sample in enumerate(targets):
        flag = True
        for item in outputs[idx]:
            if item in sample:
                temp_acc += 1
            else:
                flag = False

        if flag:
            temp_all_or_false += 1

        temp_len += len(sample)

    temp_all_or_false = (temp_all_or_false/float(len(targets)) * temp_len)

    return temp_acc, temp_all_or_false, temp_len


def test_pytorch_model(rank, model, test_data, device='cpu', criterion=nn.NLLLoss(), tokenizer=None):

    test_loss = 0
    correct = 0
    top_5 = 0

    correct2 = 0
    test_len = 0
    perplexity_loss = 0.

    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0

    model = model.to(device=device)  # load by pickle
    model.eval()
    targets_list = []
    preds = []

    decoder = None

    if parser.args.task == 'voice':
        decoder = GreedyDecoder(
            model.labels, blank_index=model.labels.index('_'))

    with torch.no_grad():

        if parser.args.task == 'detection':
            imdb, _, _, _ = combined_roidb(
                "voc_2007_test", ['DATA_DIR', parser.args.data_dir], server=True)
            data_iter = iter(test_data)
            num_images = len(test_data.dataset)
            num_classes = len(readClass(parser.args.data_dir + "/class.txt"))

            all_boxes = [[[] for _ in range(num_images)]
                         for _ in range(num_classes)]
            max_per_image = 100
            thresh = 0.0
            empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
            for i in range(num_images):
                data = next(data_iter)
                im_data = Variable(torch.FloatTensor(1).to(device))
                im_info = Variable(torch.FloatTensor(1).to(device))
                num_boxes = Variable(torch.LongTensor(1).to(device))
                gt_boxes = Variable(torch.FloatTensor(1).to(device))

                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])

                rois, cls_prob, bbox_pred, \
                    rpn_loss_cls, rpn_loss_box, \
                    RCNN_loss_cls, RCNN_loss_bbox, \
                    rois_label = model(im_data, im_info, gt_boxes, num_boxes)
                scores = cls_prob.data
                boxes = rois.data[:, :, 1:5]

                if cfg.TEST.BBOX_REG:
                    # Apply bounding-box regression deltas
                    box_deltas = bbox_pred.data
                    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).to(device=device) \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(device=device)
                        box_deltas = box_deltas.view(1, -1, 4 * num_classes)

                    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                else:
                    # Simply repeat the boxes, once for each class
                    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

                pred_boxes /= data[1][0][2].item()

                scores = scores.squeeze()
                pred_boxes = pred_boxes.squeeze()

                for j in range(1, num_classes):
                    inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                    # if there is det
                    if inds.numel() > 0:
                        cls_scores = scores[:, j][inds]
                        _, order = torch.sort(cls_scores, 0, True)
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                        cls_dets = torch.cat(
                            (cls_boxes, cls_scores.unsqueeze(1)), 1)
                        # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                        cls_dets = cls_dets[order]
                        keep = nms(cls_boxes[order, :],
                                   cls_scores[order], cfg.TEST.NMS)
                        cls_dets = cls_dets[keep.view(-1).long()]
                        all_boxes[j][i] = cls_dets.cpu().numpy()
                    else:
                        all_boxes[j][i] = empty_array

                # Limit to max_per_image detections *over all classes*
                if max_per_image > 0:
                    image_scores = np.hstack([all_boxes[j][i][:, -1]
                                              for j in range(1, num_classes)])
                    if len(image_scores) > max_per_image:
                        image_thresh = np.sort(image_scores)[-max_per_image]
                        for j in range(1, num_classes):
                            keep = np.where(
                                all_boxes[j][i][:, -1] >= image_thresh)[0]
                            all_boxes[j][i] = all_boxes[j][i][keep, :]

                imdb._reset_index(test_data.dataset.index)
                output_dir = parser.args.test_output_dir + \
                    "/learner/" + str(parser.args.this_rank)
                _, mean_ap = imdb.evaluate_detections(
                    all_boxes, output_dir, parser.args.this_rank)
                return 0, mean_ap, mean_ap, {'top_1': mean_ap, 'top_5': mean_ap, 'test_loss': 0, 'test_len': num_images}

        for data, target in test_data:
            try:
                if parser.args.task == 'nlp':

                    # if parser.args.mlm else (data, data)
                    data, target = mask_tokens(
                        data, tokenizer, parser.args, device=device)
                    data, target = Variable(data).to(
                        device=device), Variable(target).to(device=device)

                    # if parser.args.mlm else model(data, labels=target)
                    outputs = model(data, labels=target)

                    loss = outputs[0]
                    test_loss += loss.data.item()
                    perplexity_loss += loss.data.item()

                    acc = accuracy(
                        outputs[1].reshape(-1, outputs[1].shape[2]), target.reshape(-1), topk=(1, 5))

                    correct += acc[0].item()
                    top_5 += acc[1].item()

                elif parser.args.task == 'tag':
                    data, target = Variable(data).to(
                        device=device), Variable(target).to(device=device)
                    output = model(data)
                    loss = criterion(output, target)

                    # we have to scan the sample one by one
                    for idx, sample in enumerate(output):
                        target_index = torch.nonzero(
                            target[idx]).flatten().cpu().numpy().tolist()
                        maxk = len(target_index)
                        preds += [sample.topk(maxk)[1].cpu().numpy().tolist()]
                        targets_list += [target_index]

                    test_loss += loss.data.item()

                elif parser.args.task == 'speech':
                    data, target = Variable(data).to(
                        device=device), Variable(target).to(device=device)
                    data = torch.unsqueeze(data, 1)

                    output = model(data)
                    loss = criterion(output, target)

                    test_loss += loss.data.item()  # Variable.data
                    acc = accuracy(output, target, topk=(1, 5))

                    correct += acc[0].item()
                    top_5 += acc[1].item()

                elif parser.args.task == 'text_clf' and parser.args.model == 'albert-base-v2':
                    (inputs, masks) = data
                    (inputs, masks, target) = (Variable(inputs).to(device=device),
                                               Variable(masks).to(device=device), Variable(target).to(device=device))
                    outputs = model(inputs, token_type_ids=None,
                                    attention_mask=masks, labels=target)

                    loss = outputs.loss
                    output = outputs.logits

                    test_loss += loss.item()  # Variable.data
                    acc = accuracy(output, target, topk=(1, 2))

                    correct += acc[0].item()
                    top_5 += acc[1].item()

                elif parser.args.task == 'voice':
                    (inputs, target, input_percentages, target_sizes) = data

                    input_sizes = input_percentages.mul_(
                        int(inputs.size(3))).int()
                    inputs = Variable(inputs).to(device=device)

                    # unflatten targets
                    split_targets = []
                    offset = 0
                    for size in target_sizes:
                        split_targets.append(target[offset:offset + size])
                        offset += size

                    out, output_sizes = model(inputs, input_sizes)

                    decoded_output, _ = decoder.decode(out, output_sizes)
                    target_strings = decoder.convert_to_strings(split_targets)

                    for x in range(len(target_strings)):
                        transcript, reference = decoded_output[x][0], target_strings[x][0]
                        wer_inst = decoder.wer(transcript, reference)
                        cer_inst = decoder.cer(transcript, reference)
                        total_wer += wer_inst
                        total_cer += cer_inst
                        num_tokens += len(reference.split())
                        num_chars += len(reference.replace(' ', ''))

                    outputs = out.transpose(0, 1)
                    outputs = outputs.float()
                    loss = criterion(
                        outputs, target, output_sizes, target_sizes)
                    test_loss += loss.data.item()
                else:
                    data, target = Variable(data).to(
                        device=device), Variable(target).to(device=device)

                    output = model(data)

                    loss = criterion(output, target)
                    test_loss += loss.data.item()  # Variable.data
                    acc = accuracy(output, target, topk=(1, 5))

                    correct += acc[0].item()
                    top_5 += acc[1].item()

            except Exception as ex:
                logging.info(f"Testing of failed as {ex}")
                break
            test_len += len(target)

    if parser.args.task == 'voice':
        correct,  top_5, test_len = float(
            total_wer), float(total_cer), float(num_tokens)

    test_len = max(test_len, 1)
    # loss function averages over batch size
    test_loss /= len(test_data)
    perplexity_loss /= len(test_data)

    sum_loss = test_loss * test_len

    # in NLP, we care about the perplexity of the model
    acc = round(correct / test_len, 4)
    acc_5 = round(top_5 / test_len, 4)
    test_loss = round(test_loss, 4)

    if parser.args.task == 'tag':
        # precision, recall, f1, sup = precision_recall_fscore_support(targets_list, preds, average='samples')
        top_5, correct, test_len = cal_accuracy(targets_list, preds)

    logging.info('Rank {}: Test set: Average loss: {}, Top-1 Accuracy: {}/{} ({}), Top-5 Accuracy: {}'
                 .format(rank, test_loss, correct, len(test_data.dataset), acc, acc_5))

    testRes = {'top_1': correct, 'top_5': top_5,
               'test_loss': sum_loss, 'test_len': test_len}

    return test_loss, acc, acc_5, testRes


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)

        return res
