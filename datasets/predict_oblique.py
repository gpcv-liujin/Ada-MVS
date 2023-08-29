from torch.utils.data import Dataset
import numpy as np
import os
from datasets.preprocess import *
from datasets.data_io import *
from imageio import imread, imsave, imwrite

"""
# the whu-omvs dataset preprocessed by Jin Liu (only for predict)
"""

class MVSDataset(Dataset):
    def __init__(self, data_folder, view_num, args):
        super(MVSDataset, self).__init__()
        self.data_folder = data_folder
        self.viewpair_path = self.data_folder + '/viewpair.txt'
        self.image_params_path = self.data_folder + '/image_info.txt'
        self.cam_params_path = self.data_folder + '/camera_info.txt'
        self.image_path_path = self.data_folder + '/image_path.txt'

        self.args = args
        self.view_num = view_num
        self.min_interval = args.min_interval
        self.interval_scale = args.interval_scale
        self.num_depth = args.numdepth
        self.counter = 0

        self.cam_params_dict = read_cameras_text(self.cam_params_path)   # dict
        self.image_params_dict = read_images_text(self.image_params_path)  # dict
        self.image_paths, _ = read_images_path_text(self.image_path_path)   # dict
        self.sample_list = read_view_pair_text(self.viewpair_path, self.view_num)  # list [ref_view, src_views]
        self.sample_num = len(self.sample_list)


    def __len__(self):
        return len(self.sample_list)

    def read_img(self, filename):
        img = Image.open(filename)
        return img


    def read_depth(self, filename):

        depth_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        mask_path = filename.replace("depths", "masks")
        mask_path = mask_path.replace(".exr", ".png")
        mask_image = np.array(cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)) / 255.
        mask_image = mask_image < 0.5
        depth_image[mask_image] = 0

        return np.array(depth_image)

    def center_image(self, img, mode='mean'):
        # normalize image input
        if mode == 'standard':
            np_img = np.array(img, dtype=np.float32) / 255.

        elif mode == 'mean':
            img_array = np.array(img)
            img = img_array.astype(np.float32)
            var = np.var(img, axis=(0, 1), keepdims=True)
            mean = np.mean(img, axis=(0, 1), keepdims=True)
            np_img = (img - mean) / (np.sqrt(var) + 0.00000001)

        else:
            raise Exception("{}? Not implemented yet!".format(mode))

        return np_img


    def create_cams(self, image_params, cam_params_dict, num_depth=384, min_interval=0.1):
        """
        read camera txt file  (XrightYup，[Rwc|twc])
        write camera for rednet  (XrightYdown, [Rcw|tcw]
        """

        # read camera txt file
        cam = np.zeros((2, 4, 4), dtype=np.float32)
        extrinsics = np.zeros((4, 4), dtype=np.float32)

        # T
        O_xrightyup = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)
        R = np.matmul(image_params.rotation_matrix, O_xrightyup)  # Rwc, XrightYup to XrightYdown
        t = image_params.project_center      # twc
        extrinsics[0:3, 0:3] = R
        extrinsics[0:3, 3] = t
        extrinsics[3, 3] = 1.0
        extrinsics = np.linalg.inv(extrinsics)  # convert Twc to Tcw
        cam[0, :, :] = extrinsics

        # K
        cam_params = cam_params_dict[image_params.camera_id]
        fx = cam_params.focallength[0]
        fy = cam_params.focallength[1]
        x0 = cam_params.x0y0[0]
        y0 = cam_params.x0y0[1]

        cam[1][0][0] = fx
        cam[1][1][1] = fy
        cam[1][0][2] = x0
        cam[1][1][2] = y0
        cam[1][2][2] = 1

        cam[1][3][0] = image_params.depth[0]  # start
        cam[1][3][1] = (image_params.depth[1]-image_params.depth[0])/num_depth  # interval
        cam[1][3][3] = image_params.depth[1]  # end
        cam[1][3][2] = num_depth  # depth_sample_num
        # cam[1][3][2] = int((cam[1][3][3] - cam[1][3][0]) / cam[1][3][1] / 32 + 1) * 32  # depth_sample_num

        return cam


    def __getitem__(self, idx):
        data = self.sample_list[idx]

        # read input data
        outimage = None
        outcam = None

        centered_images = []
        proj_matrices = []

        for view in range(self.view_num):
            # Images
            image_idx = data[view]  # id
            image = self.read_img(self.image_paths[image_idx])
            image = np.array(image)

            # Cameras
            depth_interval = self.min_interval * self.interval_scale
            image_params = self.image_params_dict[image_idx]
            cam = self.create_cams(image_params, self.cam_params_dict, self.num_depth, depth_interval)

            # determine a proper scale to resize input
            scaled_image, scaled_cam = scale_input(image, cam, scale=self.args.resize_scale)
            # crop to fit network
            croped_image, croped_cam = crop_input(scaled_image, scaled_cam, max_h=self.args.max_h,
                                                  max_w=self.args.max_w, resize_scale=self.args.resize_scale)

            if view == 0:
                ref_img_path = self.image_paths[image_idx]
                outimage = croped_image   # for output
                outcam = croped_cam
                depth_min = croped_cam[1][3][0]
                depth_max = croped_cam[1][3][3]
                image_name = image_params.name
                # vid = image_params.image_id
                h, w = croped_image.shape[0:2]

            # scale cameras for building cost volume
            scaled_cam = scale_camera(croped_cam, scale=self.args.sample_scale)
            # multiply intrinsics and extrinsics to get projection matrix
            extrinsics = scaled_cam[0, :, :]
            intrinsics = scaled_cam[1, 0:3, 0:3]
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])

            proj_matrices.append(proj_mat)
            centered_images.append(self.center_image(croped_image))

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        depth_values = np.array([depth_min, depth_max], dtype=np.float32)

        # ms proj_mats
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, :2, :] = proj_matrices[:, :2, :] / 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, :2, :] = proj_matrices[:, :2, :] / 4

        proj_matrices_ms = {
            "stage1": stage3_pjmats,
            "stage2": stage2_pjmats,
            "stage3": proj_matrices
        }

        name = os.path.splitext(os.path.basename(image_name))[0]
        vid = os.path.dirname(image_name).split("/")[-1]


        return {"imgs": centered_images,
                "proj_matrices": proj_matrices_ms,
                "depth_values": depth_values,
                "outimage": outimage,
                "outcam": outcam,
                "ref_image_path": ref_img_path,
                "out_name": name,
                "out_view": vid}




if __name__ == "__main__":
    # some testing code, just IGNORE it
    pass
