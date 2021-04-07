import torch
import os
from .train_utils import AverageMeter, accuracy
from .torch_utils import use_gpu_if_possible

def train_epoch(model, dataloader, loss_fn, optimizer, loss_meter, performance_meter, performance, device, lr_scheduler): # note: I've added a generic performance to replace accuracy
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        # 1. reset the gradients previously accumulated by the optimizer
        #    this will avoid re-using gradients from previous loops
        optimizer.zero_grad() 
        # 2. get the predictions from the current state of the model
        #    this is the forward pass
        y_hat = model(X)
        # 3. calculate the loss on the current mini-batch
        loss = loss_fn(y_hat, y)
        # 4. execute the backward pass given the current loss
        loss.backward()
        # 5. update the value of the params
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        # 6. calculate the accuracy for this mini-batch
        acc = performance(y_hat, y)
        # 7. update the loss and accuracy AverageMeter
        loss_meter.update(val=loss.item(), n=X.shape[0])
        performance_meter.update(val=acc, n=X.shape[0])

def train_model(model, dataloader, loss_fn, optimizer, num_epochs, checkpoint_loc=None, checkpoint_name="checkpoint.pt", performance=accuracy, lr_scheduler=None, device=None, lr_scheduler_step_on_epoch=True):

    # create the folder for the checkpoints (if it's not None)
    if checkpoint_loc is not None:
        os.makedirs(checkpoint_loc, exist_ok=True)

    if device is None:
        device = use_gpu_if_possible()
    
    model = model.to(device)
    model.train()

    # epoch loop
    for epoch in range(num_epochs):

        loss_meter = AverageMeter()
        performance_meter = AverageMeter()

        # added print for LR
        print(f"Epoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.5f}")
        
        lr_scheduler_batch = lr_scheduler if not lr_scheduler_step_on_epoch else None

        train_epoch(model, dataloader, loss_fn, optimizer, loss_meter, performance_meter, performance, device, lr_scheduler_batch)

        print(f"Epoch {epoch+1} completed. Loss - total: {loss_meter.sum} - average: {loss_meter.avg}; Performance: {performance_meter.avg}")

        # produce checkpoint dictionary -- but only if the name and folder of the checkpoint are not None
        if checkpoint_name is not None and checkpoint_loc is not None:
            checkpoint_dict = {
                "parameters": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint_dict, os.path.join(checkpoint_loc, checkpoint_name))
        
        if lr_scheduler is not None and lr_scheduler_step_on_epoch:
            lr_scheduler.step()

    return loss_meter.sum, performance_meter.avg

def test_model(model, dataloader, performance=accuracy, loss_fn=None, device=None):
    # create an AverageMeter for the loss if passed
    if loss_fn is not None:
        loss_meter = AverageMeter()
    
    if device is None:
        device = use_gpu_if_possible()

    model = model.to(device)

    performance_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            
            y_hat = model(X)
            loss = loss_fn(y_hat, y) if loss_fn is not None else None
            acc = performance(y_hat, y)
            if loss_fn is not None:
                loss_meter.update(loss.item(), X.shape[0])
            performance_meter.update(acc, X.shape[0])
    # get final performances
    fin_loss = loss_meter.sum if loss_fn is not None else None
    fin_perf = performance_meter.avg
    print(f"TESTING - loss {fin_loss if fin_loss is not None else '--'} - performance {fin_perf}")
    return fin_loss, fin_perf

