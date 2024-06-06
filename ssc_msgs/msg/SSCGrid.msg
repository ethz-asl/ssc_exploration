std_msgs/Header header

# origin of occupancy grid
float32 origin_x
float32 origin_y
float32 origin_z

# reference frame of the grid
string frame

# dimensions of grid
uint32 width
uint32 height
uint32 depth

# Voxel data
float32[] data
