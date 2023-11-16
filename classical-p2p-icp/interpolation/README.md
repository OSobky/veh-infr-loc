# Interpolation between two Transformation matrices 

``` 
The algorithom will be divided into two parts:

1. Rotation Matrix Interpolation using SLERP
    - SLERP only works with Quaternion. That way we need the rotations as Quaterion
    - then scipy.spatial.transform.Slerp will be used for interpolation


2. Translation Vector Interpolation using Linear or SLERP 
    -  scipy.interpolate.interp1d will be used as in this example https://stackoverflow.com/a/48854356

```