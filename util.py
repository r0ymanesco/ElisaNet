import os
import numpy as np

import torch
import torch.nn.functional as F


def np_to_torch(img):
    img = np.swapaxes(img, 0, 1) #w, h, c
    img = np.swapaxes(img, 0, 2) #c, h, w
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img).float()


def save_nets(nets, solver, epoch, args):
    if not os.path.exists(args.model_dir):
        print('Creating model directory: {}'.format(args.model_dir))
        os.makedirs(args.model_dir)

    for net_idx, net in enumerate(nets):
        torch.save(
            net.state_dict(), '{}/ELISA-NET_EPOCH{}.pth'
            .format(args.model_dir, epoch)
        )

    torch.save(
        solver.state_dict(), '{}/ELISA-NET_SOLVER_EPOCH{}.pth'
        .format(args.model_dir, epoch)
    )


def load_weights(nets, solver, epoch, args):
    for net_idx, net in enumerate(nets):
        net.load_state_dict(
            torch.load('{}/ELISA-NET_EPOCH{}.pth'
                       .format(args.model_dir, epoch))
        )

    solver.load_state_dict(
        torch.load('{}/ELISA-NET_SOLVER_EPOCH{}.pth'
                   .format(args.model_dir, epoch))
    )


def lr_resume(optimizer, lr_resume):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_resume
    return optimizer


def forward_pass(loader, elisa_net, solver, scheduler, mode, epoch, args):
    L = []
    incorrect = []

    if mode == 'TRAIN':
        torch.set_grad_enabled(True)
    else:
        torch.set_grad_enabled(False)

    for batch_idx, (imgs, label) in enumerate(loader):
        if mode == 'TRAIN':
            solver.zero_grad()

        imgs = imgs.cuda()
        label = label.cuda()
        pred_prob = elisa_net(imgs)

        loss = F.binary_cross_entropy_with_logits(pred_prob, label)
        L.append(loss.item())

        if mode == 'TRAIN':
            loss.backward()
            solver.step()

            print('[EPOCH {}/{}][TRAIN {}/{}] Loss: {:.5f}'
                  .format(epoch, args.epochs, batch_idx+1, len(loader),
                          loss.item()))
            print('Learning rate: {:.8f}'.format(scheduler.get_last_lr()[0]))
        else:
            pred_prob = torch.sigmoid(pred_prob)
            predictions = torch.round(pred_prob)
            accuracy = label.size(0) - torch.sum(torch.abs(predictions - label))
            incorrect.append(torch.sum(torch.abs(predictions - label)).item())

            if mode == 'VALIDATE':
                print('[EPOCH {}/{}][VALIDATE {}/{}] Loss: {:.5f}; ACCURACY: {:.3f}'
                      .format(epoch, args.epochs, batch_idx+1, len(loader),
                              loss.item(),
                              accuracy.item()))
            elif mode == 'EVALUATE':
                print('[EVALUATE {}/{}] Loss: {:.5f}; ACCURACY: {}/{}'
                      .format(batch_idx+1, len(loader), loss.item(),
                              accuracy.item(), label.size(0)))
    return L, incorrect


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            # self.step = lambda a: False

    def step(self, metrics):
        if self.patience == 0:
            return False, self.best, self.num_bad_epochs

        if self.best is None:
            self.best = metrics
            return False, self.best, 0

        if torch.isnan(metrics):
            return True, self.best, self.num_bad_epochs

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True, self.best, self.num_bad_epochs

        return False, self.best, self.num_bad_epochs

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)





