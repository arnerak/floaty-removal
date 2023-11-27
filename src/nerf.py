import sys
import cv2 as cv
import numpy as np
import os 

PYNGP_PATH = '/opt/instant-ngp/build'
INGP_CONFIG_PATH = '/opt/instant-ngp/configs/nerf/base.json'
# import instant ngp python bindings library
sys.path.append(PYNGP_PATH)
import pyngp as ngp


class Nerf:
    def __init__(self, transforms_json_path, shall_train=True):
        self.testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
        self.testbed.reload_network_from_file(INGP_CONFIG_PATH)
        self.testbed.shall_train = True
        self.testbed.load_training_data(transforms_json_path)
        training_image_dimensions = np.flip(self.testbed.nerf.training.dataset.metadata[0].resolution)
        self.im_width = training_image_dimensions[1]
        self.im_height = training_image_dimensions[0]
        self.n_training_images = len(self.testbed.nerf.training.dataset.paths)

    def train_until_convergence(self, min_iterations = 2000):
        print("Training NeRF")
        loss_hist = []
        while self.testbed.frame():
            loss_hist.append(self.testbed.loss)
            if len(loss_hist) % 10 == 0:
                if len(loss_hist) > min_iterations:
                    moving_avg_loss = np.mean(loss_hist[-500:-200])
                    if np.mean(loss_hist[-200:]) >= moving_avg_loss:
                        break
                print(f"\r{self.testbed.training_step+1}", end="")
        print("\nTraining complete with loss ", self.testbed.loss)
        
    def get_training_image(self, n_training_view):
        img_path = self.testbed.nerf.training.dataset.paths[n_training_view]
        im = cv.imread(img_path)
        return im

    # returns fov in radians
    def get_fov(self, n_training_view):
        focal_length = self.testbed.nerf.training.dataset.metadata[n_training_view].focal_length[1]
        return 2 * np.arctan(self.im_height / (2 * focal_length))

    def get_filename(self, n_training_view):
        return os.path.basename(self.testbed.nerf.training.dataset.paths[n_training_view])

    def get_aspect_ratio(self, n_training_view):
        return self.im_width / self.im_height

    def get_depth_image(self, n_training_view):
        self.testbed.set_camera_to_training_view(n_training_view)
        self.testbed.render_mode = ngp.RenderMode.Depth
        depth = self.testbed.render(self.im_width, self.im_height, 1, True)
        #depth = self.testbed.render(self.im_width, self.im_height, 1, True)
        # remove alpha channel
        depth = depth[:, :, 0:3]
        depth = cv.resize(depth, (self.im_width, self.im_height))
        return depth
    
    def render(self, n_training_view, spp=2):
        self.testbed.render_mode = ngp.RenderMode.Shade
        self.testbed.set_camera_to_training_view(n_training_view)
        img = self.testbed.render(self.im_width, self.im_height, spp, False)
        img = cv.cvtColor(img, cv.COLOR_RGBA2BGRA)
        return (img * 255).astype(np.uint8)
    
    def render_eval(self, eval_frame, spp=2):
        self.testbed.render_mode = ngp.RenderMode.Shade
        self.testbed.set_camera_to_training_view(0)
        self.testbed.set_nerf_camera_matrix(np.array(eval_frame["transform_matrix"])[0:3])
        img = self.testbed.render(self.im_width, self.im_height, spp, False)
        img = cv.cvtColor(img, cv.COLOR_RGBA2BGRA)
        return (img * 255).astype(np.uint8)
