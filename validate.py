
def validate(val_loader, model, criterion, plot_acc=False, weights=None, useArcface=False, ):
    
    taskNum = 40 # change this for other tasks instead of the 40-attribute facial attributes estimation
    bar = Bar('Processing', max=len(val_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for _ in range(taskNum)]

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        metric = Conf_Metric()

        for i, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if useArcface:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True) # target: <class 'torch.Tensor'> torch.int64
            else:
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            # measure accuracy and record loss
            loss = []

            # for each attribute
            # output is a list, length=40, the 1st element shape: torch.Size([batch_size, 2])
            # print('length, output[0].shape, output[0]:', len(output), output[0].shape, output[0])
            batch_size = target.shape[0]
            pred = torch.zeros((batch_size, taskNum))
            for j in range(len(output)):
                if weights is not None:
                    target_j = one_hot(target[:, j], target.shape[0])  # torch.Size([20]) -> torch.Size([20, 2])
                    target_j = target_j.float()

                    loss_j = criterion(output[j], target_j) if criterion else 0
                    loss_j = (loss_j * weights[j]).mean()  # IMPORTANT!!!  .mean(): torch.Size([20]) -> a loss value
                    loss.append(loss_j)
                    useML = True

                else:
                    target_j = target[:, j]
                    loss_j = criterion(output[j], target_j) if criterion else 0
                    # print('val_lossj=', loss_j)
                    loss.append(loss_j)
                    useML = False


                # tensor([[0.9, 0.1]]) -> values=tensor([0.9]), indices=tensor([0])), 1: 按行； 0：按列
                _, pred[:, j] = torch.max(output[j], 1)

                updateLoss = loss[j].item() if criterion else 0 # .item() get the value of 1 element tensor
                # print('val_loss=', updateLoss)
                losses[j].update(updateLoss, input.size(0))

            losses_avg = [losses[k].avg for k in range(len(losses))]
            # top1_avg = [top1[k].avg for k in range(len(top1))]
            # bacc_curr_avg = [bacc[k].val for k in range(len(bacc))]

            loss_avg = sum(losses_avg) / len(losses_avg)
            # prec1_avg = sum(top1_avg) / len(top1_avg)
            # bacc_curr_avg = sum(bacc_curr_avg) / len(bacc_curr_avg)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            pred = pred.type_as(target) # make the two have same data type
            sampleAcc, avgAttrAccs, avgAttrBaccs = metric.cal_batch(pred, target)

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                          'Loss: {loss:.4f} | top1: {top1: .4f} | BAcc: {bacc: .4f}'.format(
                        batch=i + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=loss_avg,
                        top1=avgAttrAccs,
                        bacc = avgAttrBaccs
                        )
            bar.next()
    bar.finish()

    return (loss_avg, avgAttrAccs, avgAttrBaccs)