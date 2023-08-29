import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *



def cas_mvs_vis_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"][0:1, :, :]
        pair_results = stage_inputs["pair_result"]

        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_est = F.interpolate(depth_est.unsqueeze(1),
                                  [depth_gt.shape[1], depth_gt.shape[2]], mode='bilinear',
                                  align_corners=False).squeeze(1)

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
        pair_l1_loss = 0
        if len(pair_results) > 0:
            cnt = 0
            for pair_est in pair_results:
                pair_est = F.interpolate(pair_est.unsqueeze(1),
                                          [depth_gt.shape[1], depth_gt.shape[2]], mode='bilinear',
                                          align_corners=False).squeeze(1)
                pair_l1_loss += F.smooth_l1_loss(pair_est[mask], depth_gt[mask], reduction='mean')
                cnt += 1
            pair_l1_loss /= cnt
        else:
            pair_l1_loss = 0

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * pair_l1_loss
            total_loss += depth_loss_weights[stage_idx] * depth_loss
        else:
            total_loss += 1.0 * pair_l1_loss
            total_loss += 1.0 * depth_loss

    return total_loss, depth_loss


class FeatureNet0(nn.Module):
    def __init__(self, base_channels, num_stage=3, stride=4):
        super(FeatureNet0, self).__init__()
        print("*************feature extraction ****************")
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

        self.branch1_1 = nn.Sequential(nn.AvgPool2d((4, 4), stride=(4, 4)),
                                     Conv2d(base_channels * 4, base_channels * 2, 1, stride=1, padding=0, dilation=1))

        self.branch1_2 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     Conv2d(base_channels * 4, base_channels * 2, 1, stride=1, padding=0, dilation=1))

        self.out1 = nn.Conv2d(base_channels * 8, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]

        self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
        self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)

        self.branch2_1 = nn.Sequential(nn.AvgPool2d((4, 4), stride=(4, 4)),
                                       Conv2d(base_channels * 2, base_channels * 1, 1, stride=1, padding=0, dilation=1))

        self.branch2_2 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                       Conv2d(base_channels * 2, base_channels * 1, 1, stride=1, padding=0, dilation=1))

        self.branch3_1 = nn.Sequential(nn.AvgPool2d((4, 4), stride=(4, 4)),
                                       Conv2d(base_channels, base_channels // 2, 1, stride=1, padding=0, dilation=1))

        self.branch3_2 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                       Conv2d(base_channels, base_channels // 2, 1, stride=1, padding=0, dilation=1))



        self.out2 = nn.Conv2d(base_channels * 4, base_channels * 2, 1, bias=False)

        self.out3 = nn.Conv2d(base_channels * 2, base_channels, 1, bias=False)
        self.out_channels.append(2 * base_channels)
        self.out_channels.append(base_channels)


    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}

        output_branch1_1 = self.branch1_1(intra_feat)
        output_branch1_1 = F.upsample(output_branch1_1, (intra_feat.size()[2], intra_feat.size()[3]), mode='bilinear')

        output_branch1_2 = self.branch1_2(intra_feat)
        output_branch1_2 = F.upsample(output_branch1_2, (intra_feat.size()[2], intra_feat.size()[3]), mode='bilinear')

        output_feature1 = torch.cat((output_branch1_1, output_branch1_2, intra_feat), 1)

        out = self.out1(output_feature1)
        outputs["stage1"] = out

        intra_feat = self.deconv1(conv1, intra_feat)

        output_branch2_1 = self.branch2_1(intra_feat)
        output_branch2_1 = F.upsample(output_branch2_1, (intra_feat.size()[2], intra_feat.size()[3]), mode='bilinear')

        output_branch2_2 = self.branch2_2(intra_feat)
        output_branch2_2 = F.upsample(output_branch2_2, (intra_feat.size()[2], intra_feat.size()[3]), mode='bilinear')

        output_feature2 = torch.cat((output_branch2_1, output_branch2_2, intra_feat), 1)

        out = self.out2(output_feature2)
        outputs["stage2"] = out

        intra_feat = self.deconv2(conv0, intra_feat)

        output_branch3_1 = self.branch3_1(intra_feat)
        output_branch3_1 = F.upsample(output_branch3_1, (intra_feat.size()[2], intra_feat.size()[3]), mode='bilinear')

        output_branch3_2 = self.branch3_2(intra_feat)
        output_branch3_2 = F.upsample(output_branch3_2, (intra_feat.size()[2], intra_feat.size()[3]), mode='bilinear')

        output_feature3 = torch.cat((output_branch3_1, output_branch3_2, intra_feat), 1)

        out = self.out3(output_feature3)
        outputs["stage3"] = out

        return outputs


# train & test Regularization module

class CostRegNetRED(nn.Module):
    def __init__(self, in_channels, up=True, base_channels=8):
        super(CostRegNetRED, self).__init__()
        # simple hourglass module
        self.base_channels = base_channels
        self.conv1 = ConvReLU(in_channels, base_channels, 3, 1, 1)
        self.conv_gru1 = ConvGRUCell(base_channels, base_channels, 3)
        self.conv2 = ConvReLU(base_channels, base_channels * 2, 3, 2, 1)
        self.conv_gru2 = ConvGRUCell(base_channels * 2 , base_channels * 2, 3)
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        if up:
            self.upconv2d = nn.ConvTranspose2d(base_channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        else:
            self.upconv2d = nn.Conv2d(base_channels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, volume_variance):
        depth_costs = []
        b_num, f_num, d_num, img_h, img_w = volume_variance.shape
        state1 = torch.zeros((b_num, 8, img_h, img_w)).cuda()
        state2 = torch.zeros((b_num, 16, int(img_h / 2), int(img_w / 2))).cuda()

        cost_list = volume_variance.chunk(d_num, dim=2)
        cost_list = [cost.squeeze(2) for cost in cost_list]

        for cost in cost_list:
            # Recurrent Regularization
            conv_cost1 = self.conv1(cost)
            reg_cost1, state1 = self.conv_gru1(conv_cost1, state1)
            conv_cost2 = self.conv2(reg_cost1)
            reg_cost2, state2 = self.conv_gru2(conv_cost2, state2)
            up_cost1 = self.upconv1(reg_cost2)
            up_cost11 = F.relu(torch.add(up_cost1, reg_cost1), inplace=True)
            reg_cost = self.upconv2d(up_cost11)
            depth_costs.append(reg_cost)

        prob_volume = torch.stack(depth_costs, dim=1)
        prob_volume = prob_volume.squeeze(2)

        return prob_volume


class CostRegNet2D(nn.Module):
    def __init__(self, in_channels, base_channels=8):
        super(CostRegNet2D, self).__init__()
        self.conv0 = ConvBnReLU(in_channels, in_channels)

        self.conv1 = ConvBnReLU(in_channels, in_channels, stride=2)
        self.conv2 = ConvBnReLU(in_channels, in_channels)

        self.conv3 = ConvBnReLU(in_channels, in_channels, stride=2)
        self.conv4 = ConvBnReLU(in_channels, in_channels)

        self.conv5 = ConvBnReLU(in_channels, in_channels, stride=2)
        self.conv6 = ConvBnReLU(in_channels, in_channels)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x


class DepthNet0(nn.Module):
    def __init__(self, in_depths, in_channels, in_up=True, base_channels=8):
        super(DepthNet0, self).__init__()
        self.reg = CostRegNet2D(in_depths, base_channels)
        self.reg_fuse = CostRegNetRED(in_channels, in_up, base_channels)

    def forward(self, features, proj_matrices, depth_values, num_depth, confidence_map=None):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)
        num_views = len(features)

        # step 1. feature extraction
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        b_num, c_num, d_num, h_num, w_num = ref_volume.size()[0], ref_volume.size()[1], ref_volume.size()[2], ref_volume.size()[3], ref_volume.size()[4]

        weight_sum = 0
        fused_interm = 1e-5

        pair_confidence = []
        pair_results = []

        if confidence_map is None:
            for src_fea, src_proj in zip(src_features, src_projs):
                warped_volume = homo_warping_float(src_fea, src_proj, ref_proj, depth_values)
                warped_volume2 = ref_volume*warped_volume
                # warped_volume2 = groupwise_correlation(ref_volume, warped_volume, 8, 1)
                warped_volume1 = warped_volume2.mean(dim=1)
                score_volume = self.reg(warped_volume1)
                prob_volume = F.softmax(score_volume, dim=1)
                del warped_volume1

                photometric_confidence, indices = prob_volume.max(1)
                photometric_confidence = photometric_confidence.unsqueeze(1)
                # photometric_confidence[photometric_confidence<0.05]=1e-5
                est_depth = depth_regression(prob_volume, depth_values=depth_values)

                pair_results.append(est_depth)
                pair_confidence.append(photometric_confidence)

                weight = photometric_confidence.unsqueeze(1)
                weight_sum = weight_sum + weight
                fused_interm = fused_interm + warped_volume2 * weight

            del warped_volume, warped_volume2
            fused_interm /= weight_sum
        else:
            for src_fea, src_proj, confidence in zip(src_features, src_projs, confidence_map):
                warped_volume = homo_warping_float(src_fea, src_proj, ref_proj, depth_values)
                warped_volume2 = ref_volume*warped_volume
                # warped_volume2 = groupwise_correlation(ref_volume, warped_volume, 8, 1)  # [B, 8, D, H, W]
                weight = F.interpolate(confidence, [h_num, w_num], mode='bilinear', align_corners=False)
                weight_sum = weight_sum + weight.unsqueeze(1)  # n11hw
                fused_interm = fused_interm + warped_volume2 * weight.unsqueeze(1)  # n8dhw

            del warped_volume, warped_volume2
            fused_interm /= weight_sum
            pair_confidence = confidence_map

        # step 3. cost volume regularization
        prob_volume = self.reg_fuse(fused_interm)
        prob_volume = F.softmax(prob_volume, dim=1)

        # regression
        depth = depth_regression(prob_volume, depth_values=depth_values)
        photometric_confidence, indices = prob_volume.max(1)

        return {"depth": depth, "photometric_confidence": photometric_confidence, "pair_confidence": pair_confidence, "pair_result": pair_results}


# train & test
class AdaMVSNet(nn.Module):
    def __init__(self, ndepths=[48, 32, 8], depth_intervals_ratio=[4, 2, 1], share_cr=False, cr_base_chs=[8, 8, 8]):
        super(AdaMVSNet, self).__init__()
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_intervals_ratio = depth_intervals_ratio
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        print("**********ndepths:{}, depth_intervals_ratio:{}, chs:{}************".format(ndepths,
              depth_intervals_ratio, self.cr_base_chs))
        assert len(ndepths) == len(depth_intervals_ratio)
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

        self.feature = FeatureNet0(base_channels=8, stride=4, num_stage=self.num_stage) # unet
        self.DepthNet = nn.ModuleList([DepthNet0(in_depths=self.ndepths[0], in_channels=self.feature.out_channels[0]), DepthNet0(in_depths=self.ndepths[0],in_channels=self.feature.out_channels[1]), DepthNet0(in_depths=self.ndepths[0], in_up=False, in_channels=self.feature.out_channels[2])])

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
        pair_confidence = None
        for stage_idx in range(self.num_stage):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]

            if depth is not None:
                cur_depth = depth
                depth_h = int(cur_depth.shape[1])
                depth_w = int(cur_depth.shape[2])
            else:
                cur_depth = depth_range
                depth_h = img_h//int(stage_scale)
                depth_w = img_w//int(stage_scale)

            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                                                          ndepth=self.ndepths[stage_idx],
                                                          depth_inteval_pixel=self.depth_intervals_ratio[
                                                                                  stage_idx] * depth_interval,
                                                          dtype=img[0].dtype,
                                                          device=img[0].device,
                                                          shape=[depth_batch, depth_h, depth_w],
                                                          max_depth=depth_max,
                                                          min_depth=depth_min)

            outputs_stage = self.DepthNet[stage_idx](features_stage, proj_matrices_stage,
                                          depth_values=depth_range_samples, num_depth=self.ndepths[stage_idx], confidence_map=pair_confidence)

            depth = outputs_stage['depth']
            pair_confidence = outputs_stage['pair_confidence']

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        return outputs


# Inference
class SliceCostRegNetRED(nn.Module):
    def __init__(self, in_channels, up=True, base_channels=8):
        super(SliceCostRegNetRED, self).__init__()
        self.base_channels = base_channels
        self.conv1 = ConvReLU(in_channels, base_channels, 3, 1, 1)
        self.conv_gru1 = ConvGRUCell(base_channels, base_channels, 3)
        self.conv2 = ConvReLU(base_channels, base_channels * 2, 3, 2, 1)
        self.conv_gru2 = ConvGRUCell(base_channels * 2, base_channels * 2, 3)
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)
        if up:
            self.upconv2d = nn.ConvTranspose2d(base_channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        else:
            self.upconv2d = nn.Conv2d(base_channels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, cost, state1, state2):
        conv_cost1 = self.conv1(cost)
        reg_cost1, state1 = self.conv_gru1(conv_cost1, state1)
        conv_cost2 = self.conv2(reg_cost1)
        reg_cost2, state2 = self.conv_gru2(conv_cost2, state2)
        up_cost1 = self.upconv1(reg_cost2)
        up_cost11 = F.relu(torch.add(up_cost1, reg_cost1), inplace=True)
        reg_cost = self.upconv2d(up_cost11)

        return reg_cost, state1, state2

class InferDepthNet0(nn.Module):
    def __init__(self, in_depths, in_channels, in_up=True, base_channels=8):
        super(InferDepthNet0, self).__init__()
        self.in_up = in_up
        self.reg = CostRegNet2D(in_depths, base_channels)
        self.reg_fuse = SliceCostRegNetRED(in_channels, in_up, base_channels)

    def forward(self, features, proj_matrices, depth_values, num_depth, confidence_map=None):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)
        num_views = len(features)

        # step 1. feature extraction
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        pair_confidence = []
        pair_results = []

        b_num, f_num, img_h, img_w = ref_feature.shape
        state1 = torch.zeros((b_num, 8, img_h, img_w)).cuda()
        state2 = torch.zeros((b_num, 16, int(img_h / 2), int(img_w / 2))).cuda()

        # initialize variables
        if self.in_up:
            exp_sum = torch.zeros((b_num, 1, img_h*2, img_w*2)).cuda()
            depth_image = torch.zeros((b_num, 1, img_h*2, img_w*2)).cuda()
            max_prob_image = torch.zeros((b_num, 1, img_h*2, img_w*2)).cuda()
        else:
            exp_sum = torch.zeros((b_num, 1, img_h, img_w)).cuda()
            depth_image = torch.zeros((b_num, 1, img_h, img_w)).cuda()
            max_prob_image = torch.zeros((b_num, 1, img_h, img_w)).cuda()

        # stage-1
        if confidence_map is None:
            # ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
            for i, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):
                # # warpped features
                # warped_volume1 = homo_warping_float(src_fea, src_proj, ref_proj, depth_values)
                # warped_volume2 = (ref_volume * warped_volume1).mean(dim=1)

                # split into pieces
                # temp to deal with insufficient memory
                warped_volume2 = []
                for d in range(num_depth):
                    ref_volume1 = ref_feature.unsqueeze(2).repeat(1, 1, 1, 1, 1)
                    depth_value1 = depth_values[:, d:d + 1]
                    warped_volume1 = homo_warping_float(src_fea, src_proj, ref_proj, depth_value1)
                    warped_volume12 = (ref_volume1 * warped_volume1).mean(dim=1)
                    warped_volume2.append(warped_volume12.squeeze(1))
                warped_volume2 = torch.stack(warped_volume2, dim=1)

                score_volume = self.reg(warped_volume2)
                prob_volume = F.softmax(score_volume, dim=1)

                photometric_confidence, indices = prob_volume.max(1)
                view_weight = photometric_confidence.unsqueeze(1)
                # photometric_confidence[photometric_confidence<0.05]=1e-5
                est_depth = depth_regression(prob_volume, depth_values=depth_values)

                pair_results.append(est_depth)
                pair_confidence.append(view_weight)
                confidence_map = pair_confidence

            del warped_volume2, score_volume, prob_volume, warped_volume1

        # stage-2 & stage-3
        for d in range(num_depth):
            similarity_sum = 0
            pixel_wise_weight_sum = 1e-5
            ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, 1, 1, 1)
            depth_value = depth_values[:, d:d + 1]

            for i, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):
                # warpped features
                warped_volume = homo_warping_float(src_fea, src_proj, ref_proj, depth_value)
                warped_volume2 = (warped_volume * ref_volume)
                view_weight = F.interpolate(confidence_map[i], [img_h, img_w], mode='bilinear', align_corners=False)
                pair_confidence.append(view_weight)

                similarity_sum += warped_volume2 * view_weight.unsqueeze(1)
                pixel_wise_weight_sum += view_weight.unsqueeze(1)
                del warped_volume

            similarity = similarity_sum.div_(pixel_wise_weight_sum)

            # step 3. Recurrent Regularization
            reg_cost, state1, state2 = self.reg_fuse(similarity.squeeze(2), state1, state2)
            prob = reg_cost.exp()

            update_flag_image = (max_prob_image < prob).float()
            new_max_prob_image = update_flag_image * prob + (1 - update_flag_image) * max_prob_image

            if self.in_up:
                depth_value = F.interpolate(depth_value, [img_h*2, img_w*2], mode='bilinear', align_corners=False)

            new_depth_image = depth_value * prob + depth_image
            max_prob_image = new_max_prob_image
            depth_image = new_depth_image
            exp_sum = exp_sum + prob

        forward_exp_sum = exp_sum + 1e-10
        forward_depth_map = (depth_image / forward_exp_sum).squeeze(1)
        forward_prob_map = (max_prob_image / forward_exp_sum).squeeze(1)

        return {"depth": forward_depth_map, "photometric_confidence": forward_prob_map, "pair_confidence": pair_confidence, "pair_result": pair_results}


# predict
class Infer_AdaMVSNet(nn.Module):
    def __init__(self, num_depth=384, ndepths=[48, 32, 8], depth_intervals_ratio=[4, 2, 1], share_cr=False, cr_base_chs=[8, 8, 8]):
        super(Infer_AdaMVSNet, self).__init__()
        self.num_depth = num_depth
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_intervals_ratio = depth_intervals_ratio
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        print("**********ndepths:{}, depth_intervals_ratio:{}, chs:{}************".format(ndepths,
              depth_intervals_ratio, self.cr_base_chs))
        assert len(ndepths) == len(depth_intervals_ratio)
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

        self.feature = FeatureNet0(base_channels=8, stride=4, num_stage=self.num_stage)
        self.DepthNet = nn.ModuleList(
            [InferDepthNet0(in_depths=self.ndepths[0], in_channels=self.feature.out_channels[0]),
             InferDepthNet0(in_depths=self.ndepths[0], in_channels=self.feature.out_channels[1]),
             InferDepthNet0(in_depths=self.ndepths[0], in_up=False, in_channels=self.feature.out_channels[2])])

    def forward(self, imgs, proj_matrices, depth_values):

        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / self.num_depth

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
        pair_confidence = None
        for stage_idx in range(self.num_stage):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]

            if depth is not None:
                cur_depth = depth
                depth_h = int(cur_depth.shape[1])
                depth_w = int(cur_depth.shape[2])
            else:
                cur_depth = depth_values
                depth_h = img_h // int(stage_scale)
                depth_w = img_w // int(stage_scale)

            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                                                          ndepth=self.ndepths[stage_idx],
                                                          depth_inteval_pixel=self.depth_intervals_ratio[
                                                                                  stage_idx] * depth_interval,
                                                          dtype=img[0].dtype,
                                                          device=img[0].device,
                                                          shape=[depth_batch, depth_h, depth_w],
                                                          max_depth=depth_max,
                                                          min_depth=depth_min)

            outputs_stage = self.DepthNet[stage_idx](features_stage, proj_matrices_stage,
                                          depth_values=depth_range_samples, num_depth=self.ndepths[stage_idx], confidence_map=pair_confidence)

            depth = outputs_stage['depth']
            pair_confidence = outputs_stage['pair_confidence']

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        return outputs
