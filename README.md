# Face Swap
Face swap algorithm using geometric approaches - warping with Triangulation and Thin Plate Splines. Achieved warping using full 3D mesh of faces from a supervised encoder-decoder neural network.

### Test data source image:
![src](Data/Rambo.jpg)

### Test data destination video:
![dest](Data/face_swap_test_video.gif)

### Output after using Thin plate spline for warping and applying Kalman Filter for smooth transition between frames:
![output](Data/face_swap.gif)
