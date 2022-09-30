# ragheb ghazi 314892506
import sys
import numpy as np
import matplotlib.pyplot as plt

image_fname, centroids_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3]
z = np.loadtxt(centroids_fname)  # load centroids
orig_pixels = plt.imread(image_fname)
pixels = orig_pixels.astype(float) / 255
pixels = pixels.reshape(-1, 3)  # reshape the image


# distance function is calculating the distance between point and every center and return the minimum distance
def distance(cNum, distance, arr, cUpdated):
    Range = cNum[0]
    for i in range(Range):
        distance[:, i] = np.linalg.norm(arr - cUpdated[i], axis=1)
    clusters = np.argmin(distance, axis=1)
    return clusters


# Calculate mean and update centroids
def mean(cNum, arr, clusters, cUpdated, cNotUpdated):
    Range = cNum[0]
    for i in range(Range):
        if arr[clusters == i].shape[0] < 0:
            cUpdated[i] = cNotUpdated[i]
        else:
            cUpdated[i] = np.mean(arr[clusters == i], axis=0).round(4)


def main():
    out_put_file = open(out_fname, "w")  # opening the file to write
    pixels_array = np.array(pixels)  # pixels array
    iteration, check = 0, 0
    centroids_number = z.shape  # number of centroids
    updated_centroids = np.copy(z)  # centroids up to date
    points = pixels_array.shape  # points to train
    point_centroid_distance = np.zeros((points[0], centroids_number[0]))
    while iteration < 20 and check != 1:
        iteration += 1
        clusters = distance(centroids_number, point_centroid_distance, pixels_array, updated_centroids)
        # update original centroids to updated centroids
        not_updated_centroids = np.copy(updated_centroids)
        mean(centroids_number, pixels_array, clusters, updated_centroids, not_updated_centroids)
        # check if there is no updates then stop the loop by setting check = 1
        if np.linalg.norm(updated_centroids - not_updated_centroids) != 0:
            check = 0
        else:
            check += 1
        out_put_file.write(f"[iter {iteration - 1}]:{','.join([str(i) for i in updated_centroids])}\n")
    out_put_file.close()


if __name__ == "__main__":
    main()
