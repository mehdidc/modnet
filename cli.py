import pandas as pd
import os
import shutil
import time
import numpy as np
from collections import defaultdict
from joblib import dump
from clize import run
from glob import glob
from skimage.io import imsave
import cv2

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import auc

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms


import models

from data import ImageFilenamesDataset
from data import ImageFolderDataset


cudnn.benchmark = True

def train(config='config.cfg', *, validate_only=False):
    args = _read_config(config)
    # Data loading code
    train_dataset, eval_train_dataset, eval_valid_dataset = _load_dataset(args)
    nb_classes = len(train_dataset.classes)
    # define loss function (criterion) and optimizer
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss().to(args.device)
    elif args.loss == 'cross_entropy':
        ce = nn.CrossEntropyLoss().to(args.device)
        def criterion(input, target):
            _, target = target.max(1)
            return ce(input, target)
    else:
        raise ValueError(args.loss)
    # optionally resume from a checkpoint
    if args.resume and os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        best_auc = checkpoint['best_auc']
        model = checkpoint['model']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        # create model
        if hasattr(args, 'model_init'):
            print('Initializing weights using {}'.format(args.model_init))
            checkpoint = torch.load(args.model_init)
            model = checkpoint['model']
            model.config = args
            start_epoch = 0
            best_auc = 0
            checkpoint = None
        else:
            model = getattr(models, args.model)(nb_colors=3, nb_classes=len(train_dataset.classes))
            model.config = args
            start_epoch = 0
            best_auc = 0
            checkpoint = None
    model = model.to(args.device)
    parameters = model.parameters()
    optimizer = torch.optim.SGD(
        parameters,
        args.lr_init,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    if args.resume and checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    model.classes = train_dataset.classes
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    eval_train_loader = torch.utils.data.DataLoader(
        eval_train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)
    eval_valid_loader = torch.utils.data.DataLoader(
        eval_valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)
    valid_transform = eval_valid_dataset.transform
    stats_filename = os.path.join(args.folder, 'stats.csv')
    if os.path.exists(stats_filename):
        stats = pd.read_csv(stats_filename).to_dict(orient='list')
    else:
        stats = defaultdict(list)
    train_stats_filename = os.path.join(args.folder, 'train_stats.csv')
    if os.path.exists(train_stats_filename):
        train_stats = pd.read_csv(train_stats_filename).to_dict(orient='list')
    else:
        train_stats = defaultdict(list)

    print('Train size : {}'.format(len(train_dataset)))
    print('Valid size : {}'.format(len(eval_valid_dataset)))
    print('Classes : {}'.format(model.classes))
    folders = [
        args.folder,
        os.path.join(args.folder, 'pr_curves'),
        os.path.join(args.folder, 'pr_curves', 'train'),
        os.path.join(args.folder, 'pr_curves', 'valid')
    ]
    for f in folders:
        if not os.path.exists(f):
            os.makedirs(f)
    if args.test_time_augmentation:
        validate_func = validate_with_tta
    else:
        validate_func = validate
    loaders = (
        ('train', eval_train_loader),
        ('valid', eval_valid_loader)
    )
    if validate_only:
        stats = {}
        for split, loader in loaders:
            res = validate_func(loader, model, criterion, args)
            y_true = res['y_true']
            y_pred_probas = res['y_pred_probas']
            st = _get_stats(
                y_true, y_pred_probas,
                class_names=model.classes,
                neg_classes=args.neg_classes,
                threshold=args.threshold,
                object_threshold=args.object_threshold)
            for k, v in st.items():
                k = k + '_' + split
                if type(v) == float:
                    stats[k] = v
        for k, v in stats.items():
            print(k, v)
        return

    for epoch in range(start_epoch, args.epochs):
        optimizer = adjust_learning_rate(optimizer, epoch, args.schedule)
        train_loader = adjust_loader(train_loader, epoch, args.schedule)
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            train_stats,
            train_stats_filename,
            args,
        )
        # evaluate
        if epoch % args.eval_interval != 0:
            continue

        #nb_epoch_stats = len(stats[list(stats.keys())[0]])
        for split, loader in loaders:
            res = validate_func(loader, model, criterion, args)
            y_true = res['y_true']
            y_pred_probas = res['y_pred_probas']
            st = _get_stats(
                y_true, y_pred_probas,
                class_names=model.classes,
                neg_classes=args.neg_classes,
                threshold=args.threshold,
                object_threshold=args.object_threshold)
            for k, v in st.items():
                k = k + '_' + split
                if type(v) == float:
                    stats[k].append(v)
            report = classification_report(y_true.argmax(axis=1), y_pred_probas.argmax(axis=1), target_names=model.classes)
            print(report)
            for class_name in list(model.classes) + ['object']:
                precisions = st['precisions_' + class_name]
                recalls = st['recalls_' + class_name]
                thresholds = st['thresholds_' + class_name]
                d = {
                    'precisions': precisions,
                    'recalls': recalls,
                    'thresholds': thresholds
                }
                filename = os.path.join(
                    args.folder,
                    'pr_curves',
                    split,
                    'pr_curve_{}_{:04d}.pkl'.format(class_name, epoch)
                )
                dump(d, filename)
        stats['epoch'].append(epoch)
        df = pd.DataFrame(stats)
        df.to_csv(stats_filename, index=False)
        print(df.iloc[-1])
        auc = stats['auc_object_valid'][-1]
        data = {
            'epoch': epoch + 1,
            'model': args.model,
            'config': args,
            'model': model,
            'best_auc': max(auc, best_auc),
            'optimizer': optimizer.state_dict(),
            'valid_transform': valid_transform,
        }
        ckpt = os.path.join(args.folder, 'checkpoint.pth.tar')
        torch.save(data, ckpt)
        if auc > best_auc:
            model_best = os.path.join(args.folder, 'model_best.pth.tar')
            print('Model improved: auc valid went form {:.2f} to {:.2f}'.format(
                best_auc, auc))
            shutil.copyfile(ckpt, model_best)
            best_auc = auc


def _load_dataset(args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if hasattr(args, 'train_transform'):
        train_transform = args.train_transform
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.image_resize_init),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    if hasattr(args, 'valid_transform'):
        valid_transform = args.valid_transform
    else:
        if args.test_time_augmentation:
            valid_transform = train_transform
        else:
            valid_transform = transforms.Compose([
                transforms.Resize(args.eval_image_resize),
                transforms.CenterCrop(args.eval_image_center_crop),
                transforms.ToTensor(),
                normalize,
            ])
    
    if args.data_type == 'folder':
        train_dataset = ImageFolderDataset(
            os.path.join(args.data, args.train_folder),
            transform=train_transform,
        )
        eval_train_dataset = ImageFolderDataset(
            os.path.join(args.data, args.valid_folder),
            transform=valid_transform,
        )
        eval_valid_dataset = ImageFolderDataset(
            os.path.join(args.data, args.valid_folder),
            transform=valid_transform,
        )
    return train_dataset, eval_train_dataset, eval_valid_dataset


def _get_stats(y_true, y_pred_probas,
               class_names,
               neg_classes,
               threshold=0.5,
               object_threshold=0.5):
    y_pred = y_pred_probas > threshold
    stats = {}
    min_recalls = (0.5, 0.8, 0.9, 0.95)
    # Precision recall curves
    for cl, name in enumerate(class_names):
        name = class_names[cl]
        precisions, recalls, thresholds = precision_recall_curve(
            y_true[:, cl],
            y_pred_probas[:, cl],
            pos_label=1,
        )
        yt = y_true[:, cl]
        yp = y_pred[:, cl]
        precision = precision_score(yt, yp)
        recall = recall_score(yt, yp)
        try:
            auc = roc_auc_score(yt, yp)
        except ValueError as ex:
            print('Exception when computing auc : {}. Setting to zero'.format(ex))
            auc = np.nan
        stats['auc_{}'.format(name)] = float(auc)
        stats['precision_{}'.format(name)] = float(precision)
        for min_recall in min_recalls:
            prs = [p for p, r in zip(precisions, recalls) if r >= min_recall]
            stats['max_precision_{}(recall>={:.2f})'.format(name, min_recall)] = float(max(prs)) if len(prs) else 0
        stats['recall_{}'.format(name)] = float(recall)
        stats['precisions_{}'.format(name)] = precisions
        stats['recalls_{}'.format(name)] = recalls
        stats['thresholds_{}'.format(name)] = thresholds
    acc = (y_pred.flatten() == y_true.flatten()).mean()
    stats['acc'] = float(acc)
    if len(neg_classes):
        pos_class_indices = [
            cl for cl, name in enumerate(class_names) if name not in neg_classes]
        neg_class_indices = [
            cl for cl, name in enumerate(class_names) if name in neg_classes]
        prob_object = (
            y_pred_probas[:, pos_class_indices].sum(1) if len(pos_class_indices) else 0 -
            y_pred_probas[:, neg_class_indices].sum(1) if len(neg_class_indices) else 0)
        stats['prob_object'] = prob_object
        ytrue_object = ((y_true[:, pos_class_indices].sum(1)) > 0).astype('int32')
        precisions, recalls, thresholds = precision_recall_curve(
            ytrue_object,
            prob_object,
            pos_label=1,
        )
        for min_recall in min_recalls:
            prs = [p for p, r in zip(precisions, recalls) if r >= min_recall]
            stats['max_precision_{}(recall>={:.2f})'.format('object', min_recall)] = float(max(prs)) if len(prs) else 0
        try:
            auc = roc_auc_score(ytrue_object, prob_object)
        except ValueError:
            auc = np.nan
        stats['auc_{}'.format('object')] = float(auc)
        stats['precisions_{}'.format('object')] = precisions
        stats['recalls_{}'.format('object')] = recalls
        stats['thresholds_{}'.format('object')] = thresholds

        precision = precision_score(ytrue_object, prob_object > object_threshold)
        recall = recall_score(ytrue_object, prob_object > object_threshold)
        stats['precision_{}'.format('object')] = float(precision)
        stats['recall_{}'.format('object')] = float(recall)
    return stats


def train_one_epoch(train_loader, model, criterion, optimizer,
                    epoch, train_stats, train_stats_filename, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.to(args.device, non_blocking=True)
        input = input.to(args.device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        if args.loss == 'bce':
            pred = nn.Sigmoid()(output) > args.threshold
        elif args.loss == 'cross_entropy':
            probas = nn.Softmax(dim=1)(output)
            _, pred = torch.max(probas, 1)
            _, target = torch.max(target, 1)
        else:
            raise ValueError(args.loss)
        acc = accuracy(pred, target)
        train_stats['acc'].append(acc.item())
        train_stats['loss'].append(loss.item())
        losses.update(loss.item(), input.size(0))
        accs.update(acc.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {accs.val:.3f} ({accs.avg:.3f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, accs=accs))
            pd.DataFrame(train_stats).to_csv(train_stats_filename, index=False)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluate mode
    model.eval()
    y_true = []
    y_pred_probas = []
    filenames = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target, filenames_) in enumerate(val_loader):
            filenames.extend(filenames_)
            y_true.append(target.numpy())

            target = target.to(args.device, non_blocking=True)
            input = input.to(args.device)

            # compute output
            output = model(input)
            loss = criterion(output, target)
            losses.update(loss.item(), target.size(0))

            if args.loss == 'bce':
                probas = nn.Sigmoid()(output)
                pred = probas > args.threshold
            elif args.loss == 'cross_entropy':
                probas = nn.Softmax(dim=1)(output)
                _, pred = torch.max(probas, 1)
                _, target = torch.max(target, 1)
            else:
                raise ValueError(args.loss)

            acc = accuracy(pred,  target)
            accs.update(acc.item(), target.size(0))

            probas = probas.cpu().numpy()
            y_pred_probas.append(probas)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {accs.val:.3f} ({accs.avg:.3f})\t'.format(
                          i, len(val_loader),
                          batch_time=batch_time,
                          loss=losses,
                          accs=accs))
        print(' * Acc {accs.avg:.3f}'.format(accs=accs))
    y_true = np.concatenate(y_true, axis=0)
    y_pred_probas = np.concatenate(y_pred_probas, axis=0)
    return {'y_true': y_true, 'y_pred_probas': y_pred_probas, 'filenames': filenames}


def validate_with_tta(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluate mode
    model.eval()
    nb_classes = len(model.classes)
    nb = len(val_loader.dataset)
    y_true = np.empty((nb, nb_classes))
    y_pred_probas = np.empty(
        (nb, args.test_time_augmentation_factor, nb_classes))
    with torch.no_grad():
        end = time.time()
        for j in range(args.test_time_augmentation_factor):
            print('Pass {} on dataset'.format(j + 1))
            idx = 0
            for i, (input, target, _) in enumerate(val_loader):
                bs = len(input)
                y_true[idx:idx+bs] = target.numpy()

                target = target.to(args.device, non_blocking=True)
                input = input.to(args.device)

                # compute output
                output = model(input)
                loss = criterion(output, target)
                losses.update(loss.item(), target.size(0))

                if args.loss == 'bce':
                    probas = nn.Sigmoid()(output)
                    pred = probas > args.threshold
                elif args.loss == 'cross_entropy':
                    probas = nn.Softmax(dim=1)(output)
                    _, pred = torch.max(probas, 1)
                    _, target = torch.max(target, 1)
                else:
                    raise ValueError(args.loss)

                acc = accuracy(pred, target)
                accs.update(acc.item(), target.size(0))

                probas = probas.cpu().numpy()
                y_pred_probas[idx:idx+bs, j] = probas

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if i % args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc {accs.val:.3f} ({accs.avg:.3f})\t'.format(
                              i, len(val_loader),
                              batch_time=batch_time,
                              loss=losses,
                              accs=accs))
                idx += len(input)
            print(' * Acc {accs.avg:.3f}'.format(accs=accs))
    y_pred_probas = y_pred_probas.mean(axis=1)
    return {'y_true': y_true, 'y_pred_probas': y_pred_probas}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, nb_iter, schedule):
    sc = get_schedule_params(nb_iter, schedule)
    new_lr = sc['lr']
    old_lr = optimizer.param_groups[0]['lr']
    if old_lr != new_lr:
        print('Chaning LR from {:.5f} to {:.5f}'.format(old_lr, new_lr))
    for g in optimizer.param_groups:
        g['lr'] = new_lr
    return optimizer


def adjust_loader(loader, nb_iter, schedule):
    sc = get_schedule_params(nb_iter, schedule)
    if 'image_size' not in sc:
        return loader
    new_image_size = sc['image_size']
    tfs = loader.dataset.transform.transforms
    for i, tf in enumerate(tfs):
        class_ = tf.__class__
        if hasattr(tf, 'size'):
            if tf.size != new_image_size:
                print('Chaning size from {} to {}'.format(
                    tf.size, new_image_size))
                tfs[i] = class_(new_image_size)
    return loader


def get_schedule_params(nb_iter, schedule):
    for sc in schedule:
        (start_iter, end_iter) = sc['iter']
        if start_iter <= nb_iter < end_iter:
            break
    return sc


def accuracy(output, target):
    with torch.no_grad():
        pred = (output).float()
        target = target.float()
        return (pred == target).float().mean()


def save_dataset(config):
    args = _read_config(config)
    _load_dataset(args)
    train_dataset, eval_train_dataset, eval_valid_dataset = _load_dataset(args)
    for x, target, filename in eval_valid_dataset:
        source = os.path.abspath(filename)
        for i, t in enumerate(target):
            if t == 1:
                name = train_dataset.classes[i]
                dest_folder = os.path.join(
                    'test_images', os.path.basename(args.data), name)
                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder)
                dest = os.path.join(dest_folder, os.path.basename(filename))
                if not os.path.exists(dest):
                    os.symlink(source, dest)


def plot_roc_curves(config, *, split='valid'):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from joblib import load
    args = _read_config(config)
    folder = os.path.join(args.folder, 'pr_curves', 'figs', split)
    if not os.path.exists(folder):
        os.makedirs(folder)
    filenames = glob(os.path.join(args.folder, 'pr_curves', split, '*.pkl'))
    filenames = sorted(filenames)
    for filename in filenames:
        curve = load(filename)
        precision = curve['precisions']
        recall = curve['recalls']
        threshold = curve['thresholds']
        P = [(p, i ) for i, (p, r) in enumerate(zip(precision, recall)) if r >= 0.9]
        if len(P) > 0:
            p, i = max(P, key=lambda v: v[0])
        else:
            p, i = 0, 0
        print('{} precision:{:.2f} recall:{:.2f} threshold:{:.2f}'.format(
            filename, precision[i], recall[i], threshold[i]))
        auc_val = _smooth_then_compute_auc(precision, recall)
        print('{} auc : {:.2f}'.format(filename, auc_val))
        fig = plt.figure()
        plt.plot(precision, recall)
        plt.scatter(precision, recall, color='green')
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        title = '{}, auc:{:.2f}'.format(os.path.basename(filename), auc_val)
        plt.title(title)
        name = os.path.basename(filename)
        name, ext = name.split('.', 2)
        name = name + '.png'
        dest = os.path.join(args.folder, 'pr_curves', 'figs', split, name)
        plt.savefig(dest)
        plt.close(fig)

def _smooth_then_compute_auc(precisions, recalls):
    precisions = np.maximum.accumulate(precisions)
    return auc(1 - precisions, recalls)


def test(checkpoint, pattern, *,
         threshold=0.5,
         batch_size=None,
         force_tta=None,
         smooth=False,
         predict='sum',
         out='test_results'):
    ckpt = torch.load(checkpoint)
    print(ckpt['epoch'])
    model = ckpt['model']
    args = ckpt['config']
    if batch_size is None:
        batch_size = args.batch_size
    else:
        batch_size = int(batch_size)
    valid_transform = ckpt['valid_transform']
    model.eval()
    pos_class_indices = [
        cl for cl, name in enumerate(model.classes)
        if name not in args.neg_classes]
    neg_class_indices = [
        cl for cl, name in enumerate(model.classes)
        if name in args.neg_classes]
    dataset = ImageFilenamesDataset(
        sorted(glob(pattern)),
        valid_transform
    )
    if force_tta is None:
        nb_tta = args.test_time_augmentation_factor
    else:
        nb_tta = int(force_tta)
    probas_avg = np.zeros((len(dataset), len(model.classes)))
    for tta in range(nb_tta):
        i = 0
        loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            num_workers=0,
            batch_size=batch_size,
            collate_fn=lambda x: x,
            pin_memory=True
        )
        for samples in loader:
            X = [x for _, x, _ in samples]
            X = torch.stack(X).to(args.device)
            ims = [im for im, _,  _ in samples]
            filenames = [filename for _, _, filename in samples]
            bs = len(X)
            with torch.no_grad():
                output = model(X)
                if args.loss ==' bce':
                    probas = nn.Sigmoid()(output)
                elif args.loss == 'cross_entropy':
                    probas = nn.Softmax(dim=1)(output)
                else:
                    raise ValueError(args.loss)
            probas = probas.cpu().numpy()
            probas_avg[i:i+bs] = (
                (probas_avg[i:i+bs] * tta + probas) / (tta + 1))
            probas_cur = probas_avg[i:i+bs]
            i += bs
            print('[{}]/[{}]'.format(i, len(dataset)))
            if tta < nb_tta - 1:
                continue
            XX = X.clone()
            XX.requires_grad = True
            #masks = get_detection_mask(model, XX)
            X = X.cpu().numpy()
            nb = 0
            pr_frame = np.zeros((len(model.classes),))
            for (im, filename, pr) in zip(ims, filenames, probas_cur):
                im = np.array(im)
                if smooth:
                    pr_frame = pr_frame * 0.9 + pr * 0.1
                else:
                    pr_frame = pr
                nb += 1
                if predict == 'sum':
                    prob_object = (
                        (pr_frame[pos_class_indices].sum() if len(pos_class_indices) else 0) - 
                        (pr_frame[neg_class_indices].sum() if len(neg_class_indices) else 0)
                    )
                    pred = prob_object > threshold
                elif predict == 'argmax':
                    prob_object = pr_frame.max()
                    pred = pr_frame.argmax() in pos_class_indices
                    if prob_object < threshold:
                        pred = False
                else:
                    raise ValueError(predict)
                captions = [
                    '{}({:.2f})'.format(cli, pi)
                    for cli, pi in zip(model.classes, pr_frame)
                ]
                class_name = (
                    'object'
                    if (pred)
                    else 'background'
                )
                font = cv2.FONT_HERSHEY_PLAIN
                font_scale = 1
                text_color = [255, 255, 255]
                dx = 200
                img = np.zeros((im.shape[0], im.shape[1] + dx, 3)).astype('int32')
                img[:, 0:im.shape[1]] = im
                dy = 0.3
                for caption in captions + [class_name]:
                    img = cv2.putText(
                        img, caption, (int(im.shape[1]), int(dy*im.shape[0])),
                        font, font_scale,
                        text_color,
                        2, cv2.LINE_AA)
                    dy += 0.05
                if captions:
                    xmin = im.shape[1]
                    xmax = im.shape[1]+dx
                    xmax_object = int(xmin + (xmax - xmin) * np.clip(prob_object/threshold, 0, 1))
                    ymin = 0.1 * im.shape[0]
                    ymax = 0.2 * im.shape[0]
                    xmin = int(xmin)
                    xmax = int(xmax)
                    xmax_object = int(xmax_object)
                    ymin = int(ymin)
                    ymax = int(ymax)
                    img = cv2.rectangle(
                        img, (xmin, ymin), (xmax, ymax),
                        (255, 255, 255), thickness=cv2.FILLED)
                    img = cv2.rectangle(
                        img, (xmin, ymin), (xmax_object, ymax),
                        (255, 0, 0), thickness=cv2.FILLED)
                dest = os.path.join(out, os.path.basename(filename))
                imsave(dest, img)


def get_detection_mask(model, X, mask_threshold=0.1):
    # Get gradients of inputs with respect to class with max probability
    grads = {}
    def store_val(x):
        grads['dx'] = x
    X.register_hook(store_val)
    y_pred = model(X)
    vals, indices = y_pred.max(1)
    L = y_pred[0, indices[0].data[0]]
    L.backward()
    # Compute mask
    xgrad = grads['dx']
    mask = xgrad.data.abs().max(1)[0].cpu().numpy() >= mask_threshold
    return mask


def leaderboard(pattern='trained_models/*'):
    from tabulate import tabulate
    pd.set_option('max_colwidth', 40)
    rows = []
    for path in glob(pattern):
        model = os.path.basename(path)
        stats_filename = os.path.join('trained_models', model, 'stats.csv')
        if not os.path.exists(stats_filename):
            continue
        df = pd.read_csv(stats_filename)
        ckpts = df
        for i in range(len(ckpts)):
            ckpt = ckpts.iloc[i].to_dict()
            ckpt['model'] = model
            rows.append(ckpt)
    df = pd.DataFrame(rows)
    p = df['precision_object_valid']
    r = df['recall_object_valid']
    df['f1_object_valid'] = 2 * ((p * r) / (p + r))
    df = df.sort_values(by=['auc_object_valid'], ascending=False)
    pd.set_option('precision', 3)
    cols = [
        'model',
        'epoch',
        'precision_object_valid',
        'recall_object_valid',
        'f1_object_valid',
        'auc_object_valid',
        'max_precision_object(recall>=0.90)_valid',
    ]
    df = df[cols]
    df = df.rename(columns={
        'max_precision_object(recall>=0.90)_valid': 'prec(rec>=.9)'
    })
    print(tabulate(df, headers='keys'))


def _read_config(config):
    cfg = {}
    exec(open(config).read(), {}, cfg)
    obj = Object()
    obj.__dict__ = cfg
    return obj


def clean_images(pattern):
    from skimage.io import imread
    for filename in glob.glob(pattern):
        try:
            imread(filename)
        except Exception:
            os.remove(filename)


class Object(object):
    pass


if __name__ == '__main__':
    run([train, test, plot_roc_curves, leaderboard, clean_images, save_dataset])
