import torch
import numpy as np

def batch_pix_accuracy(predict, target):

    predict = predict.int() + 1
    target = target.int() + 1

    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target)*(target > 0)).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

def batch_intersection_union(predict, target, num_class):
    '''
    if the label shape == (N, C, H, W), do squeeze for label, make it's shape to (N, H, W), which shape equal to output's shape size
    :param output:  N * H * W
    :param target:  N * H * W
    :param num_class: 21
    :return:
    '''

    if len(predict.shape) == 4:
        predict = torch.argmax(predict, dim=1)

    if len(target.shape) == 4:
        target = torch.squeeze(target, dim=1)

    predict = predict + 1
    target = target + 1

    predict = predict * (target > 0).long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()

def eval_metrics(output, target, num_classes):
    # correct, labeled = batch_pix_accuracy(output.data, target)
    inter, union = batch_intersection_union(output.data, target, num_classes)
    return [np.round(inter, 5), np.round(union, 5)]  # round(list, 5) mean keep 5 digits after point for everyone in list
