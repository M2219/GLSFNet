import torch.nn.functional as F

def model_loss_train(depth_est, depth_gt, mask_d, disp_ests, disp_gts, img_masks, cv_scale):
    if cv_scale == 4:
        weights = [1.00/3, 1.00/6]
        all_losses = []
        for disp_est, disp_gt, weight, mask_img in zip(disp_ests, disp_gts[0:2], weights, img_masks[0:2]):
            all_losses.append(weight * F.smooth_l1_loss(disp_est[mask_img], disp_gt[mask_img], reduction="mean"))

        d_l = 1 * F.mse_loss(depth_est[mask_d], depth_gt[mask_d], reduction="mean") + 0.8 * F.smooth_l1_loss(depth_est[mask_d], depth_gt[mask_d], reduction="mean")

    return sum(all_losses) + d_l

def model_loss_test(depth_est, depth_gt, mask_d):

    loss = F.mse_loss(depth_est[mask_d], depth_gt[mask_d], reduction="mean")

    return loss
