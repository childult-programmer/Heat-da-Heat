import time
from tqdm import tqdm
import numpy as np
import argparse
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import KaistPdDataset
from utils import *
import wandb
import sys
sys.path.append('/home/silee/workspace/kroc/classification_dn')
from dn_model import DNClassifier


# Data parameters
data_path = '../kaistPD_json'      # path with json files
keep_difficult = True   # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dn classification model
dn_checkpoint = '/home/silee/workspace/kroc/classification_dn/checkpoint.pth.tar'
dn_model = DNClassifier(n_classes=2)
dn_model.load_state_dict(torch.load(dn_checkpoint))
dn_model.to(device)

# Fix Random seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic=True
cudnn.benchmark = True

############################################### Parser ###############################################

parser = argparse.ArgumentParser(description='PyTorch Kaist Pedestrian Transformer')
parser.add_argument('--model',         default=None, type=str, help='Model name')
parser.add_argument('--checkpoint',    default=None, type=str, help='checkpoint')
parser.add_argument('--SF_checkpoint', default=None, type=str, help='SF checkpoint')
parser.add_argument('--mode',          default='SF+', type=str, help='Select Mode SF/SF+/AF/AF+')
parser.add_argument('--batch_size',    default=16, type=int, help='Batch size')
parser.add_argument('--workers',       default=4, type=int, help='CPU')
parser.add_argument('--epochs',        default=100, type=int, help='Total epochs')
parser.add_argument('--lr',            default=5e-4, type=float, help='Learning rate')
parser.add_argument('--weight_decay',  default=5e-4, type=float, help='Weight decay')
parser.add_argument('--decay_at',  default=None, type=int, help='Learning rate decay at')
parser.add_argument('--print_freq',    default=50, type=int, help='Print training status every __ batches')
parser.add_argument('--n_classes',     default=2, type=int, help='Number of Class')
parser.add_argument('--wandb_enable',  action='store_true', help='Wandb Enable/Disable')

args = parser.parse_args()

############################################### Wandb ###############################################
if args.wandb_enable:
    wandb.init(project='kroc', name=args.model)
    wandb.run.log_code('./out/', include_fn=lambda path: path.endswith(".py"))

def main():
    """
    Training.
    """
    # Read label map for dn classification
    with open('/home/silee/workspace/kroc/classification_dn/label_map.json', 'r') as j:
        dn_label_map = json.load(j)

    # Initialize model or load checkpoint
    if args.checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=args.n_classes, mode=args.mode, SF_checkpoint=args.SF_checkpoint)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)

        optimizer = torch.optim.Adam(params=[{'params': biases, 'lr': 2 * args.lr}, {'params': not_biases}], betas=(0.9, 0.999),
                                     lr=args.lr, weight_decay=args.weight_decay)

        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[ int(args.epochs*0.5), int(args.epochs*0.75) ], gamma=0.1)

    else:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoad checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[ int(args.epochs*0.5), int(args.epochs*0.75) ], gamma=0.1)

    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device) 

    # Loss, Move to default device
    # criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)    
    model = model.to(device)

    # Custom dataloaders
    train_dataset = KaistPdDataset(data_path,
                                   split='train',
                                   keep_difficult=keep_difficult)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=args.workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Epochs
    for epoch in tqdm(range(start_epoch, args.epochs)):
        if args.wandb_enable:
            wandb.log({"Epoch": epoch})

        # Decay learning rate at particular epochs
        if epoch == args.decay_at:
            adjust_learning_rate(optimizer, args.decay_at)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              label_map=dn_label_map)

        optim_scheduler.step()

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, temporal=False)

        if epoch % 5 == 0:
            save_checkpoint(epoch, model, optimizer, temporal=True)


def dn_prediction(images, dn_model, dn_label_map):
    inverse_label_map = {v:k for k, v in dn_label_map.items()}
    hypothesis = dn_model(images)
    prob = F.softmax(hypothesis, dim=1) 
    day_prob, night_prob = torch.tensor_split(prob, 2, dim=1)
    day_prob = day_prob.detach().cpu().numpy()
    
    return day_prob


def train(train_loader, model, criterion, optimizer, epoch, label_map):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (visible_images, lwir_images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        visible_images = visible_images.to(device)  # (batch_size (N), 3, 300, 300)
        lwir_images = lwir_images.to(device)        # (batch_size (N), 3, 300, 300)
        
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        d_prob = dn_prediction(visible_images, dn_model, label_map)

        # Forward prop.
        predicted_locs, predicted_scores = model(visible_images, lwir_images, d_prob)  # (N, 8732, 4), (N, 8732, n_classes)
        
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)
        
        if args.wandb_enable:
            wandb.log({"loss": loss})
        
        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        # if grad_clip is not None:
        #     clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), visible_images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))


    del predicted_locs, predicted_scores, visible_images, lwir_images, boxes, labels  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()
