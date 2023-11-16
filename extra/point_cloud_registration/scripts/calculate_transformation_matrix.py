import numpy as np
from eulerangles import euler2matrix

# 1. transformation matrix ouster_north to ouster_south
roll = -0.1006728081
pitch = 0.3248542883
yaw = -14.32052765
rotation_angles_euler = np.array([roll,pitch,yaw])
rotation_matrix=euler2matrix(rotation_angles_euler,axes='xyz',intrinsic=True,right_handed_rotation=True).T
print("rotation",rotation_matrix)
#[[ 0.9689116   0.2473422   0.00566975]
# [-0.24735544  0.96892322  0.00175704]
# [-0.00505896 -0.00310486  0.99998238]]
# transposed
#[[ 0.9689116  -0.24735544 -0.00505896]
# [ 0.2473422   0.96892322 -0.00310486]
# [ 0.00566975  0.00175704  0.99998238]]


# manually calculated
#rotation_matrix=np.array([[  0.9689116,  0.2473422,  0.0056697],
#  [-0.2473554,  0.9689232,  0.0017570],
#  [-0.0050590, -0.0031049,  0.9999824 ]],dtype=float).T

translation_vector = np.array([1.34,13.632,-0.167], dtype=float)

# rotate the translation vector, then stack it
translation_vector_rotated = np.matmul(rotation_matrix, translation_vector)
transformation_matrix = np.zeros((4, 4))
transformation_matrix[3,3]=1
transformation_matrix[0:3, 0:3] = rotation_matrix
transformation_matrix[0:3, 3] = -translation_vector_rotated
print("transformation_matrix:",transformation_matrix)
# transformation matrix ouster_north to ouster_south
#[[ 9.68911600e-01  2.47342200e-01  5.66970000e-03 -4.66916357e+00]
# [-2.47355400e-01  9.68923200e-01  1.75700000e-03 -1.28766114e+01]
# [-5.05900000e-03 -3.10490000e-03  9.99982400e-01  2.16102118e-01]]

