import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *



def cas_rednet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * depth_loss
        else:
            total_loss += 1.0 * depth_loss

    return total_loss, depth_loss

class FeatureNet(nn.Module):
    def __init__(self, base_channels, num_stage=3, stride=4, arch_mode="unet"):
        super(FeatureNet, self).__init__()
        assert arch_mode in ["unet", "fpn"], print("mode must be in 'unet' or 'fpn', but get:{}".format(arch_mode))
        print("*************feature extraction arch mode:{}****************".format(arch_mode))
        self.arch_mode = arch_mode
        self.stride = stride
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]

        if self.arch_mode == 'unet':
            if num_stage == 3:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
                self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
                self.out_channels.append(2 * base_channels)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out_channels.append(2 * base_channels)
        elif self.arch_mode == "fpn":
            final_chs = base_channels * 4
            if num_stage == 3:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
                self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
                self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels * 2)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        outputs["stage1"] = out
        if self.arch_mode == "unet":
            if self.num_stage == 3:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

                intra_feat = self.deconv2(conv0, intra_feat)
                out = self.out3(intra_feat)
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

        elif self.arch_mode == "fpn":
            if self.num_stage == 3:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
                out = self.out3(intra_feat)
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

        return outputs


# train & test Regularization module
class RED_Regularization(nn.Module):
    def __init__(self, in_channels, base_channels = 8):
        super(RED_Regularization, self).__init__()
        self.base_channels = base_channels
        self.conv_gru1 = ConvGRUCell2(in_channels, base_channels, 3)
        self.conv_gru2 = ConvGRUCell2(base_channels * 2 , base_channels * 2, 3)
        self.conv_gru3 = ConvGRUCell2(base_channels * 4, base_channels * 4, 3)
        self.conv_gru4 = ConvGRUCell2(base_channels * 8 , base_channels * 8, 3)
        self.conv1 = ConvReLU(in_channels, base_channels * 2, 3, 2, 1)
        self.conv2 = ConvReLU(base_channels * 2, base_channels * 4, 3, 2, 1)
        self.conv3 = ConvReLU(base_channels * 4, base_channels * 8, 3, 2, 1)
        self.upconv3 = ConvTransReLU(base_channels * 8, base_channels * 4, 3, 2, 1, 1)
        self.upconv2 = ConvTransReLU(base_channels * 4, base_channels * 2, 3, 2, 1, 1)
        self.upconv1 = ConvTransReLU(base_channels * 2, base_channels, 3, 2, 1, 1)
        self.upconv2d = nn.ConvTranspose2d(base_channels, 1, kernel_size=3, stride=1, padding=1, output_padding=0)

    def forward(self, volume_variance):
        depth_costs = []
        b_num, f_num, d_num, img_h, img_w = volume_variance.shape
        state1 = torch.zeros((b_num, 8, img_h, img_w)).cuda()
        state2 = torch.zeros((b_num, 16, int(img_h / 2), int(img_w / 2))).cuda()
        state3 = torch.zeros((b_num, 32, int(img_h / 4), int(img_w / 4))).cuda()
        state4 = torch.zeros((b_num, 64, int(img_h / 8), int(img_w / 8))).cuda()

        cost_list = volume_variance.chunk(d_num, dim=2)
        cost_list = [cost.squeeze(2) for cost in cost_list]

        for cost in cost_list:
            # Recurrent Regularization
            conv_cost1 = self.conv1(-cost)
            conv_cost2 = self.conv2(conv_cost1)
            conv_cost3 = self.conv3(conv_cost2)
            reg_cost4, state4 = self.conv_gru4(conv_cost3, state4)
            up_cost3 = self.upconv3(reg_cost4)
            reg_cost3, state3 = self.conv_gru3(conv_cost2, state3)
            up_cost33 = torch.add(up_cost3, reg_cost3)
            up_cost2 = self.upconv2(up_cost33)
            reg_cost2, state2 = self.conv_gru2(conv_cost1, state2)
            up_cost22 = torch.add(up_cost2, reg_cost2)
            up_cost1 = self.upconv1(up_cost22)
            reg_cost1, state1 = self.conv_gru1(-cost, state1)
            up_cost11 = torch.add(up_cost1, reg_cost1)
            reg_cost = self.upconv2d(up_cost11)
            depth_costs.append(reg_cost)
        prob_volume = torch.stack(depth_costs, dim=1)
        prob_volume = prob_volume.squeeze(2)

        return prob_volume


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()

    def compute_depth(self, prob_volume, depth_values=None):
        '''
        prob_volume: 1 x D x H x W
        '''
        B, M, H, W = prob_volume.shape[0], prob_volume.shape[1],prob_volume.shape[2],prob_volume.shape[3]
        # prob_indices = HW shaped vector
        probs, indices = prob_volume.max(1)
        # depth_range = depth_start + torch.arange(depth_num).float() * depth_interval
        # depth_range = depth_values.to(prob_volume.device)
        depths = torch.index_select(depth_values, 1, indices.flatten())
        depth_image = depths.view(B, H, W)
        prob_image = probs.view(B, H, W)

        return depth_image, prob_image


    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)
        num_views = len(features)

        # step 1. feature extraction
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            warped_volume = homo_warping_float(src_fea, src_proj, ref_proj, depth_values)
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        # step 3. cost volume regularization
        prob_volume = cost_regularization(volume_variance)
        prob_volume = F.softmax(prob_volume, dim=1)

        # regression
        depth = depth_regression(prob_volume, depth_values=depth_values)
        photometric_confidence, indices = prob_volume.max(1)


        return {"depth": depth, "photometric_confidence": photometric_confidence}


# train & test  CascadeREDNet
class CascadeREDNet(nn.Module):
    def __init__(self, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1], share_cr=False, cr_base_chs=[8, 8, 8]):
        super(CascadeREDNet, self).__init__()
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        print("**********netphs:{}, depth_intervals_ratio:{}, chs:{}************".format(ndepths,
              depth_interals_ratio, self.cr_base_chs))
        assert len(ndepths) == len(depth_interals_ratio)
        self.stage_infos = {
            "stage1":{
                "scale": 4.0,
            },
            "stage2": {
                "scale": 2.0,
            },
            "stage3": {
                "scale": 1.0,
            }
        }

        self.feature = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stage, arch_mode='unet')
        if self.share_cr:
            self.cost_regularization = RED_Regularization(in_channels=self.feature.out_channels, base_channels=8)
        else:
            self.cost_regularization = nn.ModuleList([RED_Regularization(in_channels=self.feature.out_channels[i],
                                                                 base_channels=self.cr_base_chs[i])
                                                      for i in range(self.num_stage)])
        self.DepthNet = DepthNet()

    def forward(self, imgs, proj_matrices, depth_values):
        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -2].cpu().numpy())
        depth_interval = float(depth_values[0, -1].cpu().numpy())
        depth_range = depth_values[:, 0:-1]

        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):
            img = imgs[:, nview_idx]
            features.append(self.feature(img))

        depth_batch = int(img.shape[0])
        img_h = int(img.shape[2])
        img_w = int(img.shape[3])

        outputs = {}
        depth, cur_depth = None, None
        for stage_idx in range(self.num_stage):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]

            if depth is not None:
                cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1),
                                                [img_h, img_w], mode='bilinear',
                                                align_corners=False).squeeze(1)
            else:
                cur_depth = depth_values
            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                                                        ndepth=self.ndepths[stage_idx],
                                                        depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
                                                        dtype=img[0].dtype,
                                                        device=img[0].device,
                                                        shape=[depth_batch, img_h, img_w],
                                                        max_depth=depth_max,
                                                        min_depth=depth_min)



            dv = F.interpolate(depth_range_samples.unsqueeze(1),
                               [self.ndepths[stage_idx], img.shape[2] // int(stage_scale),
                                img.shape[3] // int(stage_scale)], mode='trilinear', align_corners=False)

            outputs_stage = self.DepthNet(features_stage, proj_matrices_stage,
                                          depth_values=dv.squeeze(1),
                                          num_depth=self.ndepths[stage_idx],
                                          cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx])   # add propagationNet

            depth = outputs_stage['depth']

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        return outputs


##################Inference########################################
# predict Regularization module
class slice_RED_Regularization(nn.Module):
    def __init__(self, in_channels, base_channels = 8):
        super(slice_RED_Regularization, self).__init__()
        self.base_channels = base_channels
        self.conv_gru1 = ConvGRUCell2(in_channels, base_channels, 3)
        self.conv_gru2 = ConvGRUCell2(base_channels * 2 , base_channels * 2, 3)
        self.conv_gru3 = ConvGRUCell2(base_channels * 4, base_channels * 4, 3)
        self.conv_gru4 = ConvGRUCell2(base_channels * 8 , base_channels * 8, 3)
        self.conv1 = ConvReLU(in_channels, base_channels * 2, 3, 2, 1)
        self.conv2 = ConvReLU(base_channels * 2, base_channels * 4, 3, 2, 1)
        self.conv3 = ConvReLU(base_channels * 4, base_channels * 8, 3, 2, 1)
        self.upconv3 = ConvTransReLU(base_channels * 8, base_channels * 4, 3, 2, 1, 1)
        self.upconv2 = ConvTransReLU(base_channels * 4, base_channels * 2, 3, 2, 1, 1)
        self.upconv1 = ConvTransReLU(base_channels * 2, base_channels, 3, 2, 1, 1)
        self.upconv2d = nn.ConvTranspose2d(base_channels, 1, kernel_size=3, stride=1, padding=1, output_padding=0)

    def forward(self, cost, state1, state2, state3, state4):
        # Recurrent Regularization
        conv_cost1 = self.conv1(-cost)
        conv_cost2 = self.conv2(conv_cost1)
        conv_cost3 = self.conv3(conv_cost2)
        reg_cost4, state4 = self.conv_gru4(conv_cost3, state4)
        up_cost3 = self.upconv3(reg_cost4)
        reg_cost3, state3 = self.conv_gru3(conv_cost2, state3)
        up_cost33 = torch.add(up_cost3, reg_cost3)
        up_cost2 = self.upconv2(up_cost33)
        reg_cost2, state2 = self.conv_gru2(conv_cost1, state2)
        up_cost22 = torch.add(up_cost2, reg_cost2)
        up_cost1 = self.upconv1(up_cost22)
        reg_cost1, state1 = self.conv_gru1(-cost, state1)
        up_cost11 = torch.add(up_cost1, reg_cost1)
        reg_cost = self.upconv2d(up_cost11)

        return reg_cost, state1, state2, state3, state4


class InferDepthNet(nn.Module):
    def __init__(self):
        super(InferDepthNet, self).__init__()

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(
            depth_values.shapep[1], num_depth)
        num_views = len(features)

        # step 1. feature extraction
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        b_num, f_num, img_h, img_w = ref_feature.shape
        state1 = torch.zeros((b_num, 8, img_h, img_w)).cuda()
        state2 = torch.zeros((b_num, 16, int(img_h / 2), int(img_w / 2))).cuda()
        state3 = torch.zeros((b_num, 32, int(img_h / 4), int(img_w / 4))).cuda()
        state4 = torch.zeros((b_num, 64, int(img_h / 8), int(img_w / 8))).cuda()

        # initialize variables
        exp_sum = torch.zeros((b_num, 1, img_h, img_w)).cuda()
        depth_image = torch.zeros((b_num, 1, img_h, img_w)).cuda()
        max_prob_image = torch.zeros((b_num, 1, img_h, img_w)).cuda()

        for d in range(num_depth):
            ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, 1, 1, 1)
            depth_value = depth_values[:, d:d + 1]
            volume_sum = ref_volume
            volume_sq_sum = ref_volume ** 2
            del ref_volume
            for src_fea, src_proj in zip(src_features, src_projs):
                # warpped features
                warped_volume = homo_warping_float(src_fea, src_proj, ref_proj, depth_value)
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
                del warped_volume

            # aggregate multiple feature volumes by variance
            volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
            volume_variance = volume_variance.squeeze(2)

            # step 3. Recurrent Regularization
            reg_cost, state1, state2, state3, state4 = cost_regularization(volume_variance, state1, state2, state3, state4)
            prob = reg_cost.exp()

            update_flag_image = (max_prob_image < prob).float()
            new_max_prob_image = update_flag_image * prob + (1 - update_flag_image) * max_prob_image
            new_depth_image = depth_value * prob + depth_image

            max_prob_image = new_max_prob_image
            depth_image = new_depth_image
            exp_sum = exp_sum + prob

        # update the sum_avg
        forward_exp_sum = exp_sum + 1e-10
        forward_depth_map = (depth_image / forward_exp_sum).squeeze(1)
        forward_prob_map = (max_prob_image / forward_exp_sum).squeeze(1)
        return {"depth": forward_depth_map, "photometric_confidence": forward_prob_map}


# predict  CascadeREDNet
class Infer_CascadeREDNet(nn.Module):
    def __init__(self, num_depth=384, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1], share_cr=False, cr_base_chs=[8, 8, 8]):
        super(Infer_CascadeREDNet, self).__init__()
        self.num_depth = num_depth
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        print("**********netphs:{}, depth_intervals_ratio:{}, chs:{}************".format(ndepths,
              depth_interals_ratio, self.cr_base_chs))
        assert len(ndepths) == len(depth_interals_ratio)
        self.stage_infos = {
            "stage1": {
                "scale": 4.0,
            },
            "stage2": {
                "scale": 2.0,
            },
            "stage3": {
                "scale": 1.0,
            }
        }

        self.feature = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stage, arch_mode='unet')
        if self.share_cr:
            self.cost_regularization = slice_RED_Regularization(in_channels=self.feature.out_channels, base_channels=8)
        else:
            self.cost_regularization = nn.ModuleList([slice_RED_Regularization(in_channels=self.feature.out_channels[i],
                                                                 base_channels=self.cr_base_chs[i])
                                                      for i in range(self.num_stage)])
        self.DepthNet = InferDepthNet()

    def forward(self, imgs, proj_matrices, depth_values):

        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / self.num_depth

        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):
            img = imgs[:, nview_idx]
            features.append(self.feature(img))
        img_h = int(img.shape[2])
        img_w = int(img.shape[3])
        outputs = {}
        depth, cur_depth = None, None
        for stage_idx in range(self.num_stage):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]

            if depth is not None:
                cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1),
                                                [img_h, img_w], mode='bilinear',
                                                align_corners=False).squeeze(1)
            else:
                cur_depth = depth_values
            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                                                        ndepth=self.ndepths[stage_idx],
                                                        depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
                                                        dtype=img[0].dtype,
                                                        device=img[0].device,
                                                        shape=[img.shape[0], img_h, img_w],
                                                        max_depth=depth_max,
                                                        min_depth=depth_min)

            dv = F.interpolate(depth_range_samples.unsqueeze(1),
                               [self.ndepths[stage_idx], img.shape[2] // int(stage_scale),
                                img.shape[3] // int(stage_scale)], mode='trilinear', align_corners=False)

            outputs_stage = self.DepthNet(features_stage, proj_matrices_stage,
                                          depth_values=dv.squeeze(1),
                                          num_depth=self.ndepths[stage_idx],
                                          cost_regularization=self.
                                          cost_regularization if self.share_cr else self.cost_regularization[stage_idx])

            depth = outputs_stage['depth']

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        return outputs

