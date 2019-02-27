import torch


from dice_loss import dice_coeff


def eval_net(net, dataset, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    for i, (images, masks) in enumerate(dataset):

        img = images.type(torch.FloatTensor)
        true_mask = masks.type(torch.FloatTensor)

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)[0]
        mask_pred = (mask_pred > 0.5).float()

        tot += dice_coeff(mask_pred, true_mask).item()
    return tot / i
