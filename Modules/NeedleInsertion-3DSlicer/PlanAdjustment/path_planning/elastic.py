import matplotlib.pyplot as plt
import numpy as np
import torch
import nrrd 
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from torch.distributions import Normal
from tqdm import tqdm
import logging

from matplotlib import rc
font = {'size'   : 14}
rc('font', **font)

# Air, 0
# Fat, label value 1
# Prostate, label value 2
# Muscle, label value 3
# Blubospongiosus m. , label value 4
# Ischiocavernosus m. , label value 5
# Crus of the Penis / Corpus Cavernosum, label value 6
# Rectum, label value 8
# Obturator internus m. , label value 10
# Pubic Arc, label value 11
# Transverse Perineal m. , label value 13
# Bulb of the Penus / Corpus Spongiosum, label value 16

def get_tissue_label(coords, pca, readdata):
    coords = pca.inverse_transform(coords)
    # offset by the "space origin" from the nrrd file
    coords = coords - [-114.22564115161549, -131.18068302734218,-108.96788724692369]
    # translate voxel size by "space directions" from the nrrd file
    try:
        return readdata[(coords[:,1]/.875).round().astype(int), (coords[:,0]/.875).round().astype(int), (coords[:,2]/4).round().astype(int)]
    except:
        return np.zeros(coords.shape[0])

def replace_values(arr, val0, val1, val2, val3, val4, val5, val6, val7, val8, val9, val10):
    result = np.copy(arr)
    result[result == 0] = val0
    result[result == 2] = val1
    result[result == 3] = val2
    result[result == 4] = val3
    result[result == 5] = val4
    result[result == 6] = val5
    result[result == 8] = val6
    result[result == 10] = val7
    result[result == 11] = val8
    result[result == 13] = val9
    result[result == 16] = val10
    return result

def load_anatomy(entry_pt):
    '''Loads a .nrrd file and creates a numpy array with voxels 1mm on each side.'''
    anatomy_filename = 'anatomy-label.nrrd'
    readdata, header = nrrd.read(anatomy_filename)

    pca = PCA(n_components=3)
    pca.fit(entry_pt)

    # Sets the start point to (0, 50, 50) in the PCA space
    x, y, z = np.mgrid[0:120, 0:100, 0:100]
    y-=50
    z-=50
    coordinates = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    labels = get_tissue_label(coordinates, pca, readdata)
    x_range = 120  # x varies from 0 to 120
    y_range = 100  # y varies from -50 to 50
    z_range = 100  # z varies from -50 to 50
    labels = labels.reshape((x_range, y_range, z_range))
    geo = labels
    return geo, pca



def predict_deflection(geo, reg,
                       start_point,
                       target_point,
                       n_samples = 5000):

    # geo[target_point[0],target_point[1],target_point[2]] = 10
    material_vec = np.zeros((126,))
    # compute the same unit‐vector and step‐indices
    dist  = np.linalg.norm(target_point - start_point)
    u_vec = (target_point - start_point) / dist

    n     = dist.astype(int)

    # build a (126 × 3) array of floating‐point sampling coordinates
    steps     = np.arange(n)[:, None]           # shape (n,1)
    positions = start_point + steps * u_vec[None, :]  # (n,3)

    # turn them into integer grid‐indices
    #    since everything here is non‐negative, astype(int) acts like floor()
    ix, iy, iz = positions.astype(int).T          # each is shape (n,)

    # handle out‐of‐bounds by clipping to the valid range
    ix = np.clip(ix, 0, geo.shape[0] - 1)
    iy = np.clip(iy, 0, geo.shape[1] - 1)
    iz = np.clip(iz, 0, geo.shape[2] - 1)

    # gather all values
    material_vec[126-n:] = geo[ix, iy, iz]                # shape (126,)
    # print(material_vec)
    # handle out‐of‐bounds by replacing with the last valid value
    valid_mask = (
        (positions[:,0] >= 0) & (positions[:,0] < geo.shape[0]) &
        (positions[:,1] >= 0) & (positions[:,1] < geo.shape[1]) &
        (positions[:,2] >= 0) & (positions[:,2] < geo.shape[2])
    )
    if not valid_mask.all():
        first_bad = np.where(~valid_mask)[0][0]
        material_vec[first_bad:] = material_vec[first_bad - 1]


    depth = int(target_point[0]-start_point[0])
    material_vec[-1] = depth/10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # means and standard deviations, in the same order as replace_values expects:
    #   [Air, Fat, Prostate, Muscle, Blubospongiosus,
    #    Ischiocavernosus, Corpus Cavernosum, Rectum,
    #    Obturator internus, Pubic Arc, Transverse Perineal,
    #    Corpus Spongiosum]
    means = torch.tensor([0.01, .005, 0.043, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        device=device)
    stds   = torch.tensor([0.005, .01, 0.08, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                        device=device)

    # create tissue properties distribution
    dist = Normal(means, stds)

    # sample and clamp to enforce lower bound = 0
    # -> samples.shape == (n_samples, 11)
    samples = dist.sample((n_samples,)).clamp(min=0.0)

    # material_vec (length 126) and samples (n_samples×11)
    #    material_vec[-1] already = depth/10
    #    samples is a torch.Tensor of shape (n_samples, 11) on `device`
    n_samples = samples.shape[0]
    device = samples.device

    #build label -> slot mapping
    label_list = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 13, 16]
    label_to_idx = {lbl: i for i, lbl in enumerate(label_list)}

    # make an index array for the first 125 entries
    base_labels = material_vec[:-1].astype(int)              # shape (125,)
    gather_idx = torch.tensor(
        [label_to_idx[lbl] for lbl in base_labels],
        dtype=torch.long,
        device=device
    )   # shape (125,)

    # Apply the tissue properties according to the anatomy labels
    inputs_base = samples[:, gather_idx]                     # (n_samples, 125)

    # make a column of the fixed last value = material_vec[-1]
    last_val   = float(material_vec[-1])
    last_col   = torch.full((n_samples, 1), last_val, device=device)

    # concatenate -> (n_samples, 126)
    NN_input_batch = torch.cat([inputs_base, last_col], dim=1)

    # run predictor on the batch
    predictions = reg.predict(NN_input_batch)           # -> shape (n_samples,)

    # post‐process
    predictions = np.array(predictions) + start_point[1]
    tip_pos = int(depth)

    return predictions, tip_pos

def plot_distribution(predictions, geo, tip_pos, start_pos):
    """
    Plot the distribution of deflection predictions.
    """

    L = 141
    plt.imshow(geo[:,:,start_pos[2]].T, aspect='auto', alpha=.6)
    # Calculate the mean at each point of the predictions
    means = np.mean(predictions, axis=0)

    # Define the points at which these predictions are evaluated
    x_points = np.arange(-L-3, 0, 3) + 3 + tip_pos

    # Calculate percentiles for each point
    percentiles_50 = np.percentile(predictions, [25, 75], axis=0)  # 50% CI
    percentiles_90 = np.percentile(predictions, [5, 95], axis=0)   # 90% CI
    percentiles_95 = np.percentile(predictions, [2.5, 97.5], axis=0)  # 95% CI
    percentiles_99 = np.percentile(predictions, [0.5, 99.5], axis=0)  # 99% CI

    # Plotting the mean line
    plt.plot(x_points, means, label='Mean Prediction', color='black')  # Mean line in black for contrast

    # Adding shading for different confidence intervals with distinct colors
    plt.fill_between(x_points, percentiles_99[0], percentiles_99[1], color='red', alpha=.07)
    plt.fill_between(x_points, percentiles_95[0], percentiles_95[1], color='red', alpha=.1)
    plt.fill_between(x_points, percentiles_90[0], percentiles_90[1], color='red', alpha=.2)
    plt.fill_between(x_points, percentiles_50[0], percentiles_50[1], color='red', alpha=.3)

    colors = [(1, 0, 0, 0.2), (1, 0, 0, 0.4), (1, 0, 0, 0.6), (1, 0, 0, 0.8)]  # RGBA tuples
    labels = ['99% CI', '95% CI', '90% CI', '50% CI']

    # Create a list of patches with alpha values
    patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
    # plt.scatter(data_pca[:-1,0], -data_pca[:-1,1]+50, c='b', label = 'Actual Needle Shape')
    # plt.legend()
    scatter_legend = Line2D([0], [0], marker='o', color='b', label='Actual Needle Shape')
                            # markerfacecolor='b', markersize=10)

    # Add the scatter legend to the patches list
    patches.append(scatter_legend)

    # Add the custom legend to the plot
    ax = plt.gca()
    legend = ax.legend(handles=patches, loc='upper left')

    plt.tight_layout()
    # Additional plot settings
    plt.xlabel('Insertion Depth (mm)')
    plt.ylabel('y Position (mm)')
    # plt.title('Distribution of Deflection Predictions')
    plt.ylim(70, 40)
    plt.grid(True)
    plt.show()

def find_entry(geo, reg, start_point, target_point, tolerance_range=1, search_dist=10, search_samples=20):
    """
    Find the entry point for the needle based on likelihood of reaching the target point.
    geo: The geometry of the tissue.
    reg: The regression model for predicting deflection.
    start_point: The initial point to start searching from.
    target_point: The point we want to reach.
    tolerance_range: The acceptable range of deviation from the target point.
    search_dist: The radius to search around the start point.
    search_samples: The number of samples to take in the search.

    Returns the best start point, the tolerance count for each sample, and the index of the best start point.
    """

    start_points = np.tile(start_point,(search_samples, 1))
    start_points[:,1] = np.linspace(start_point[1] - search_dist, start_point[1] + search_dist, search_samples)
    tolerance_count = np.zeros((search_samples,))

    for i in tqdm(range(start_points.shape[0])):
        predictions, tip_pos = predict_deflection(geo, reg,
                                              start_point = start_points[i],
                                              target_point = target_point)
        # plot_distribution(predictions, geo, tip_pos, start_points[i])
        tolerance_count[i] = np.sum(np.abs(predictions[:,-1] - target_point[1]) < tolerance_range)
    best_index = np.argmax(tolerance_count)
    best_start_point = start_points[best_index]

    return best_start_point, tolerance_count, best_index

def plan_path(geo, reg, entry_pt, target_pt, tolerance_range=1):
    """
    Plan the path for the needle from entry point to target point.
    geo: The geometry of the tissue.
    reg: The regression model for predicting deflection.
    entry_pt: The entry point for the needle.
    target_pt: The target point we want to reach.
    tolerance_range: The range of tip position predictions over which 
        the average (the returned path) is calculated.

    Returns the planned path as a list of points.
    """
    
    predictions, tip_pos = predict_deflection(geo, reg,
                                        start_point = entry_pt,
                                        target_point = target_pt)
    
    prediction_mask = np.abs(predictions[:,-1] - target_pt[1]) < tolerance_range
    masked_predictions = predictions[prediction_mask]
    mean_path = np.mean(masked_predictions, axis=0) # The planned path will be the mean of the predictions that fall within the tolerance range
    x = np.linspace(target_pt[0]-141, target_pt[0], num=mean_path.shape[0])
    z = np.full_like(x, target_pt[2])

    planned_path = np.vstack((x, mean_path, z)).T  
    planned_path = planned_path[planned_path[:,0] >= entry_pt[0]]  # Remove points before the entry point
    return planned_path
