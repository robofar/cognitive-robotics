import numpy as np

        
def pose_from_odometry(u, x_prev):
    x_cur = np.zeros(x_prev.shape)
    x_cur[0] = x_prev[0] + u[0] * np.cos(np.deg2rad(x_prev[2]) + np.deg2rad(u[1]))
    x_cur[1] = x_prev[1] + u[0] * np.sin(np.deg2rad(x_prev[2]) + np.deg2rad(u[1]))
    x_cur[2] = x_prev[2] + u[1] + u[2]

    return x_cur

def poses_from_odometries(error, error_combinations, x_prev, u):
    x = np.zeros(x_prev.shape)
    for i in range(0, len(error_combinations)):
        x[i] = pose_from_odometry((error_combinations[i] * error) + u, x_prev[i])
    
    return x

def generate_spread(x, u, error, binary_digits):
    poses = np.zeros(binary_digits.shape)
    for i in range(0,len(binary_digits)):
        u_new = u + binary_digits[i]*error
        resulting_pose = pose_from_odometry(u_new, x)
        poses[i] = resulting_pose
    return poses

def marginalize_orientation(poses):
    # Initialize a dictionary to store groups of (x, y) pairs
    grouped_dict = {}
    # Group the arrays by (x, y) pairs
    for pose in poses:
        x, y, theta = pose
        if (x, y) not in grouped_dict:
            grouped_dict[(x, y)] = []
        grouped_dict[(x, y)].append(theta)
        
    # Calculate the mean z value for each group and update the original array
    new_poses = poses.copy()
    for (x, y), thetas in grouped_dict.items():
        mean_theta = np.mean(thetas)
        for pose in new_poses:
            if pose[0] == x and pose[1] == y:
                pose[2] = mean_theta
    
    return new_poses
