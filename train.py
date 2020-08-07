import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.optim.lr_scheduler as LS

import network
from dataset import ElisaDataset
from train_options import parser
from torch.utils.tensorboard import SummaryWriter
from util import save_nets, load_weights, forward_pass, lr_resume
from util import EarlyStopping


writer = SummaryWriter()
args = parser.parse_args()
print(args)

print('Creating train loader')
train_set = ElisaDataset('elisadata/standard', 'TRAIN')
train_loader = data.DataLoader(
    dataset=train_set,
    batch_size=args.train_batch_size,
    shuffle=True,
    num_workers=0
)
print('Creating valid loader')
valid_set = ElisaDataset('elisadata/standard', 'VALIDATE')
valid_loader = data.DataLoader(
    dataset=valid_set,
    batch_size=args.eval_batch_size,
    shuffle=False,
    num_workers=0
)
print('Creating eval loader')
eval_set = ElisaDataset('elisadata/standard', 'EVALUATE')
eval_loader = data.DataLoader(
    dataset=eval_set,
    batch_size=args.eval_batch_size,
    shuffle=False,
    num_workers=0
)

elisa_net = network.ElisaNet(args.c_feat).cuda()

params = [{'params': elisa_net.parameters()}]
solver = optim.Adam(params, lr=args.lr)

lmda = lambda x: 0.5  # TODO: can change this based on bad_epochs
scheduler = LS.MultiplicativeLR(solver, lr_lambda=lmda)

es = EarlyStopping(mode=args.es_mode, min_delta=args.loss_delta,
                   patience=args.patience)

epoch = 0

if args.resume_epoch != 0:
    load_weights([elisa_net], solver, args.resume_epoch, args)
    epoch = args.resume_epoch
    solver = lr_resume(solver, args.lr_resume)
    print('Loaded weights from epoch {}'.format(args.resume_epoch))

while epoch < args.epochs and not args.eval:
    epoch += 1

    train_loss, _ = forward_pass(train_loader, elisa_net, solver,
                                 scheduler, 'TRAIN', epoch, args)

    writer.add_scalars('Train Loss', {'ELISANET':
                                      np.mean(train_loss)}, epoch)

    if epoch % 50 == 0:
        valid_loss, valid_incorrect = forward_pass(valid_loader, elisa_net,
                                                   None, None, 'VALIDATE',
                                                   epoch, args)

        valid_loss = np.mean(valid_loss)
        valid_accuracy = 1. - (sum(valid_incorrect) / len(valid_set))
        print('[EPOCH {}/{}][VALIDATE AVG] Loss: {:.4f}; ACCURACY: {:.4f}'
              .format(epoch, args.epochs, valid_loss, valid_accuracy))

        writer.add_scalars('Valid Loss', {'ELISANET':
                                          valid_loss}, epoch)
        writer.add_scalars('Valid ACCURACY', {'ELISANET':
                                              valid_accuracy}, epoch)

        flag, best, bad_epochs = es.step(torch.Tensor([valid_loss]))
        if flag:
            print('Early stopping criterion met')
            break
        else:
            if bad_epochs == 0:
                save_nets([elisa_net], solver, epoch, args)
                best_epoch = epoch
                print('Saving best net weights')
            elif bad_epochs % (args.patience//2) == 0:
                scheduler.step()

        print('[EPOCH {}] Current Valid Loss: {:.6f}; Best: {:.6f}; Bad epochs: {}; Best epoch: {}'
              .format(epoch, valid_loss, best.item(), bad_epochs, best_epoch))

if not args.eval:
    load_weights([elisa_net], solver, best_epoch, args)
    print('Training done')
    print('Loaded best net weights from epoch {}'.format(best_epoch))

print('Evaluating...')
eval_loss, eval_incorrect = forward_pass(eval_loader, elisa_net,
                                         None, None, 'EVALUATE', None, args)

eval_accuracy = 1. - (sum(eval_incorrect) / len(eval_set))
eval_loss = np.mean(eval_loss)
print('[EVALUATE AVG] Loss: {}; ACCURACY: {:.4f}; Best Epoch: {}'
      .format(eval_loss, eval_accuracy, best_epoch))

writer.close()
