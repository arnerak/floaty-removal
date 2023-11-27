import numpy as np
import matplotlib.colors as mcolors
import open3d as o3d
import sys
from ngpgrid import NgpGrid

def visualize_point_cloud(point_cloud, color_values, point_size=5):
    """
    Visualizes a point cloud with color values.

    Args:
        point_cloud (numpy.ndarray): A Nx3 numpy array of point coordinates.
        color_values (numpy.ndarray): A Nx3 numpy array of RGB color values.
    """

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()

    # Set the point coordinates and colors
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(color_values / 255.0)  # Normalize color values to [0, 1]

    # Create a render option and set the point size
    render_option = o3d.visualization.RenderOption()
    render_option.point_size = point_size

    # Create a visualizer, add the point cloud, and set the render option
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name='Point Cloud Visualization', width=679, height=436)
    visualizer.add_geometry(pcd)
    visualizer.get_render_option().point_size = point_size

    # ctr = visualizer.get_view_control()
    # parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2023-06-02-11-09-52.json")
    # ctr.convert_from_pinhole_camera_parameters(parameters)

    # Run the visualizer
    visualizer.run()
    visualizer.destroy_window()


def generate_distinguishable_colors(n_colors):
    """
    Generate a distinguishable color palette as a numpy array.

    Args:
        n_colors (int): Number of colors to generate.

    Returns:
        numpy.ndarray: An Nx3 numpy array of RGB color values.
    """
    color_indices = np.linspace(0, 1, n_colors)
    hsv_colors = np.zeros((n_colors, 3), dtype=np.float32)
    hsv_colors[:, 0] = color_indices
    hsv_colors[:, 1] = 1.0
    hsv_colors[:, 2] = 1.0

    rgb_colors = (mcolors.hsv_to_rgb(hsv_colors) * 255).astype(np.uint8)
    return rgb_colors


def generate_point_cloud(indices):
    return np.array([(np.array([x,y,z])/128-0.5)*2**mip for x,y,z,mip in indices])

def get_confident_density_voxels_from_multiple_grids(filepaths):
	# (x,y,z,mip) -> score
    scores = dict()
    final_point_set = set()

    min_score = 1 / len(filepaths)
    max_mip_lvl = 6
	
    for filepath in filepaths:
        ngp_grid = NgpGrid(filepath)
        for point in ngp_grid.density_points:
            x,y,z,mip = point
            max_dist_from_origin = max( abs(x-64), abs(y-64), abs(z-64) )
            min_mip_level_containing_point = np.ceil( max( np.log2(max_dist_from_origin) - 6 + mip, 0) )
            score = max(min_score, 1 / (max_mip_lvl - min_mip_level_containing_point + 1))
            if point in scores:
                scores[point] += score
            else:
                scores[point] = score
	
    for point in scores:
        if scores[point] > 0.9:
            final_point_set.add(point)

    return final_point_set

def visualize_clusters(clusters):
    #finalCluster = []
    #numTotalPoints = sum([len(cluster) for cluster in clusters])
    #for cluster in clusters:
    #    finalCluster.extend(cluster)
    #    if len(finalCluster) >= 0.95 * numTotalPoints:
    #        break    

    color_values = []
    point_cloud = []
    color_palette = generate_distinguishable_colors(10)
    for i,cluster in enumerate(clusters):
        color_values.extend(len(cluster) * [color_palette[i % 10]])
        point_cloud.extend(generate_point_cloud(cluster))

    color_values = np.array(color_values)
    point_cloud = np.array(point_cloud)

    # Visualize the point cloud with the generated color values
    visualize_point_cloud(point_cloud, color_values, 2)

def visualize_ngpgrid(ngp_grid):
    pc = generate_point_cloud(ngp_grid.density_points)
    c = generate_distinguishable_colors(1)
    visualize_point_cloud(pc, c, 2)

def sscs(ngp_grids):
    # (x,y,z,mip) -> score
    scores = dict()
    final_point_set = set()

    min_score = 1 / len(ngp_grids)
    max_mip_lvl = 6
	
    for ngp_grid in ngp_grids:
        for point in ngp_grid.density_points:
            x,y,z,mip = point
            max_dist_from_origin = max( abs(x-64), abs(y-64), abs(z-64) )
            min_mip_level_containing_point = np.ceil( max( np.log2(max_dist_from_origin) - 6 + mip, 0) )
            score = max(min_score, 1 / (max_mip_lvl - min_mip_level_containing_point + 1))
            if point in scores:
                scores[point] += score
            else:
                scores[point] = score
	
    for point in scores:
        if scores[point] > 0.9:
            final_point_set.add(point)

    return final_point_set

def cluster(ngp_grid):
    clusters, noise = ngp_grid.cluster()
    clusters = sorted(clusters, key=lambda cluster: len(cluster), reverse=True)
    finalCluster = []
    numTotalPoints = sum([len(cluster) for cluster in clusters])
    for cluster in clusters:
        finalCluster.extend(cluster)
        if len(finalCluster) > 0.8 * numTotalPoints:
            break
    return finalCluster

if __name__ == "__main__":
    filepaths = sys.argv[1:]
    filedir = '/'.join(filepaths[0].split('/')[0:-1]) or "."

    # density grid 1: only clustering
    # TODO: don't hardcode filepath
    ngp_grid = NgpGrid(filepaths[1])
    clusters, noise = ngp_grid.cluster()
    print(len(clusters), len(noise), len(clusters) + len(noise))
    clusters = sorted(clusters, key=lambda cluster: len(cluster), reverse=True)
    NgpGrid.serialize_data(filedir + "/density_cluster", clusters[0])
    #quit()

    # density grid 2: only multi scale filtering
    confident_density_points = get_confident_density_voxels_from_multiple_grids(filepaths)
    NgpGrid.serialize_data(filedir + "/density_scale_filter", confident_density_points)
	
    # density grid 3: both
    ngp_grid.density_points = confident_density_points
    clusters, noise = ngp_grid.cluster()
    clusters = sorted(clusters, key=lambda cluster: len(cluster), reverse=True)
    finalCluster = []
    numTotalPoints = sum([len(cluster) for cluster in clusters])
    for cluster in clusters:
        finalCluster.extend(cluster)
        if len(finalCluster) > 0.8 * numTotalPoints:
            break
    NgpGrid.serialize_data(filedir + "/density_both2", finalCluster)

    #visualize_clusters([clusters[0]])
