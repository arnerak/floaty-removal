import cv2
import numpy as np
import os
import sys
import json
import copy
from skimage.metrics import structural_similarity as ski_ssim
from floatyremoval import sscs, cluster, visualize_ngpgrid, visualize_clusters
from ngpgrid import NgpGrid
from nerf import Nerf

AABB_SCALES = [4, 8, 16]

def convert_deblur_dataset(path, aabb_scales=AABB_SCALES):
    transforms = os.path.join(path, "transforms.json")
    eval_frames = []
    with open(transforms, "r") as file:
        t = json.load(file)
        frames = []
        dof_t = copy.deepcopy(t)
        t["frames"].sort(key=lambda x: x["file_path"], reverse=False)
        for i,frame in enumerate(t["frames"]):
            im_path = frame["file_path"]
            im_file = os.path.basename(im_path)
            # skip eval frames
            if "_1_" in im_file:
                eval_frames.append(frame)
                continue
            new_frame = copy.deepcopy(frame)
            frames.append(new_frame)
        dof_t["frames"] = frames

        for aabb_scale in aabb_scales:
            dof_t["aabb_scale"] = aabb_scale
            with open(os.path.join(path, "transforms_eval_%d.json" % aabb_scale), "w") as file:
                json.dump(dof_t, file, indent=2)
    return eval_frames


def compute_masked_psnr(image1, image2):
    mask = image1[:, :, 3]
    image1 = image1[:, :, 0:3]
    mask = np.where(mask == 255, 1, 0).astype(np.uint8)
    #im1[mask==0] = 0
    #im2[mask==0] = 0
    mask = np.expand_dims(mask, axis=2)
    mask = np.repeat(mask, 3, axis=2)
    # compute the squared difference between the images
    diff = (image1.astype(np.int32) - image2.astype(np.int32)) ** 2
    # apply the mask to the difference image
    masked_diff = diff * mask
    # compute the mean squared error (MSE) within the masked region
    mse = np.sum(masked_diff) / np.count_nonzero(mask)
    # compute the PSNR using the MSE and the maximum pixel value
    max_value = 255
    psnr = 20 * np.log10(max_value) - 10 * np.log10(mse)
    return psnr


def compute_masked_ssim(image1, image2):
    mask = image1[:, :, 3]
    image1 = image1[:, :, 0:3]
    mask = np.where(mask == 255, 1, 0).astype(np.uint8)
    mask = np.expand_dims(mask, axis=2)
    mask = np.repeat(mask, 3, axis=2)
    # convert the images to the range [0, 255] if necessary
    if image1.dtype == np.float32:
        image1 = np.uint8(image1 * 255)
    if image2.dtype == np.float32:
        image2 = np.uint8(image2 * 255)
    # apply the mask to both images
    masked_image1 = image1 * mask
    masked_image2 = image2 * mask
    score, diff = ski_ssim(masked_image1, masked_image2, full=True, multichannel=True, channel_axis=2) 
    return np.sum(diff * mask) / (np.count_nonzero(mask))


def coverage(image1):
    mask = image1[:, :, 3]
    image1 = image1[:, :, 0:3]
    return np.count_nonzero(mask) / (image1.shape[0] * image1.shape[1])


def eval_scene(scene_path):
    eval_frames = convert_deblur_dataset(scene_path)
    # render GT images
    if not os.path.exists(os.path.join(scene_path, "gt")):
        os.mkdir(os.path.join(scene_path, "gt"))
        transforms_path = os.path.join(scene_path, "transforms.json")
        nerf = Nerf(transforms_path)
        nerf.train_until_convergence(4000)
        # render GT eval views and write them to disk
        for i in range(nerf.n_training_images):
            if "_1_" in nerf.get_filename(i):
                im = nerf.render(i)
                cv2.imwrite(os.path.join(scene_path, "gt", nerf.get_filename(i)), im)

    # get density grids for multiple aabb_scales for SSCS
    if not os.path.exists(os.path.join(scene_path, "grids")):
        os.mkdir(os.path.join(scene_path, "grids"))
        for aabb_scale in AABB_SCALES:
            transforms_path = os.path.join(scene_path, f"transforms_eval_{aabb_scale}.json")
            nerf = Nerf(transforms_path)
            nerf.train_until_convergence()
            density_grid_dump = nerf.testbed.nerf.get_density_grid()
            with open(os.path.join(scene_path, "grids", str(aabb_scale)), "wb") as file:
                file.write(density_grid_dump.tobytes())

    # generate post-process density grid
    if not os.path.exists(os.path.join(scene_path, "grids", "pp")):
        density_grids = []
        for aabb_scale in AABB_SCALES:
            density_grid = NgpGrid(os.path.join(scene_path, "grids", str(aabb_scale)))
            density_grids.append(density_grid)
        # post-process density grids
        density_grids[0].density_points = sscs(density_grids)
        density_grids[0].density_points = cluster(density_grids[0])
        with open(os.path.join(scene_path, "grids", "pp"), "wb") as file:
            file.write(density_grids[0].serialized())

    # render eval images    
    if not os.path.exists(os.path.join(scene_path, "eval")):
        os.mkdir(os.path.join(scene_path, "eval"))
        # train a new nerf for evaluation
        transforms_path = os.path.join(scene_path, "transforms_eval_16.json")
        nerf = Nerf(transforms_path)
        nerf.testbed.background_color = [0, 0, 0, 0]
        nerf.train_until_convergence(4000)
        # override ingp's density grid with the post-processed density grid
        pp_density_grid = np.fromfile(os.path.join(scene_path, "grids", "pp"), dtype=np.byte)
        nerf.testbed.nerf.set_density_grid(pp_density_grid)
        # render eval views and write them to disk
        for frame in eval_frames:
            im = nerf.render_eval(frame)
            fn = os.path.basename(frame["file_path"])
            cv2.imwrite(os.path.join(scene_path, "eval", fn), im)

    # compute metrics
    sum_psnr = 0
    sum_ssim = 0
    sum_cvg = 0
    num = 0
    for frame in eval_frames:
        fn = os.path.basename(frame["file_path"])
        p1 = os.path.join(scene_path, "eval", fn)
        p2 = os.path.join(scene_path, "gt", fn)
        im1 = cv2.imread(p1, cv2.IMREAD_UNCHANGED)
        im2 = cv2.imread(p2)
        psnr = compute_masked_psnr(im1, im2)
        ssim = compute_masked_ssim(im1, im2)
        cvg = coverage(im1)
        sum_psnr += psnr
        sum_ssim += ssim
        sum_cvg += cvg
    
    print("PSNR:", sum_psnr / len(eval_frames))
    print("SSIM:", sum_ssim / len(eval_frames))
    print("Cvg:", sum_cvg / len(eval_frames))




if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 eval.py PATH/TO/SCENE")

    scene_path = sys.argv[1]
    eval_scene(scene_path)

