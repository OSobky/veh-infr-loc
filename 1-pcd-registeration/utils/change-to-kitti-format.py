import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as R


def combine_matrices(rot_path, trans_path, output_path):
    # Read rotation matrices
    rotations = []
    for file_name in sorted(os.listdir(rot_path)):
        # with open(os.path.join(rot_path, file_name), 'r') as rot_file:
            rot_mtx = R.from_quat(np.loadtxt(
                os.path.join(rot_path, file_name),
                delimiter=",",
            )).as_matrix().flatten().tolist()
            rotations.append(rot_mtx)
            # rotations.append([line.strip() for line in rot_file.readlines()])
    
    # Read translation vectors
    translations = []
    for file_name in sorted(os.listdir(trans_path)):
        with open(os.path.join(trans_path, file_name), 'r') as trans_file:
            translations.append([line.strip() for line in trans_file.readlines()]) 

    # Combine matrices and vectors
    combined = []
    for i in range(len(rotations)):
        matrix = rotations[i]
        vector = translations[i]
        matrix.insert(3, vector[0])
        matrix.insert(7, vector[1])
        matrix.insert(11, vector[2])
        combined.append(' '.join(map(str,matrix)))

    # Write combined output to file
    with open(output_path, 'w') as output_file:
        output_file.write('\n'.join(combined))
    print(f"Combined matrices and vectors saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine rotation matrices and translation vectors.')
    parser.add_argument('--rot', type=str, required=True, help='Path to rotation matrices')
    parser.add_argument('--trans', type=str, required=True, help='Path to translation vectors')
    parser.add_argument('--output', type=str, required=True, help='Path to output file')
    args = parser.parse_args()

    combine_matrices(args.rot, args.trans, args.output)
