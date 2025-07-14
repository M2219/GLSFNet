import glob
import os

## training

root_im = "/datasets/kittiraw/2011_09_26/"
root_s = "/datasets/data_depth_velodyne/"
root_gt = "/datasets/data_depth_annotated/"

list_gt_dirs = os.listdir(os.path.join(root_gt, 'train'))
list_im_dirs = os.listdir(root_im)

list_im_dirs_new = []
for item in list_gt_dirs:
    if item in list_im_dirs:
        list_im_dirs_new.append(item)


train_seq_gt_list = []
train_seq_left_list = []
train_seq_right_list = []
train_seq_s_list = []

for p in list_im_dirs_new:

    train_seq_gt = sorted(glob.glob(os.path.join(root_gt, "train", p, 'proj_depth/groundtruth/image_02/*')))
    tail = [os.path.split(path)[1] for path in train_seq_gt]

    train_seq_s = sorted(glob.glob(os.path.join(root_s, "train", p, 'proj_depth/velodyne_raw/image_02/*')))

    left_p = os.path.join(root_im, p, 'image_02/data/')
    train_seq_left = [os.path.join(left_p, t) for t in tail]

    right_p = os.path.join(root_im, p, 'image_03/data/')
    train_seq_right = [os.path.join(right_p, t) for t in tail]


    train_seq_gt_list = train_seq_gt_list +  train_seq_gt
    train_seq_left_list = train_seq_left_list + train_seq_left
    train_seq_right_list = train_seq_right_list + train_seq_right
    train_seq_s_list = train_seq_s_list + train_seq_s

print("----------------")
print(len(train_seq_left_list))
print(len(train_seq_s_list))
print(len(train_seq_right_list))
print(len(train_seq_gt_list))

for i in range(len(train_seq_left_list)):

    train_seq_gt_list[i] = train_seq_gt_list[i].split(root_gt)[1]
    train_seq_s_list[i] = train_seq_s_list[i].split(root_s)[1]
    train_seq_left_list[i] = train_seq_left_list[i].split(root_im)[1]
    train_seq_right_list[i] = train_seq_right_list[i].split(root_im)[1]

with open('kitti_depth_train.txt', 'w') as f:

    for i in range(len(train_seq_left_list)):
        f.write(train_seq_left_list[i] + " " + train_seq_right_list[i] + " " + train_seq_s_list[i]  + " " +train_seq_gt_list[i])
        f.write('\n')


## validation

root_im = "/datasets/kittiraw/2011_09_26/"
root_gt = "/datasets/data_depth_annotated/"
root_s = "/datasets/data_depth_velodyne/"

list_gt_dirs = os.listdir(os.path.join(root_gt, 'val'))
list_im_dirs = os.listdir(root_im)

list_im_dirs_new = []
for item in list_gt_dirs:
    if item in list_im_dirs:
        list_im_dirs_new.append(item)


val_seq_gt_list = []
val_seq_s_list = []
val_seq_left_list = []
val_seq_right_list = []

for p in list_im_dirs_new:

    val_seq_gt = sorted(glob.glob(os.path.join(root_gt, "val", p, 'proj_depth/groundtruth/image_02/*')))
    tail = [os.path.split(path)[1] for path in val_seq_gt]

    val_seq_s = sorted(glob.glob(os.path.join(root_s, "val", p, 'proj_depth/velodyne_raw/image_02/*')))

    left_p = os.path.join(root_im, p, 'image_02/data/')
    val_seq_left = [os.path.join(left_p, t) for t in tail]

    right_p = os.path.join(root_im, p, 'image_03/data/')
    val_seq_right = [os.path.join(right_p, t) for t in tail]

    val_seq_gt_list = val_seq_gt_list +  val_seq_gt
    val_seq_s_list = val_seq_s_list +  val_seq_s
    val_seq_left_list = val_seq_left_list + val_seq_left
    val_seq_right_list = val_seq_right_list + val_seq_right

print("----------------")
print(len(val_seq_left_list))
print(len(val_seq_right_list))
print(len(val_seq_gt_list))
print(len(val_seq_s_list))

for i in range(len(val_seq_left_list)):

    val_seq_gt_list[i] = val_seq_gt_list[i].split(root_gt)[1]
    val_seq_s_list[i] = val_seq_s_list[i].split(root_s)[1]
    val_seq_left_list[i] = val_seq_left_list[i].split(root_im)[1]
    val_seq_right_list[i] = val_seq_right_list[i].split(root_im)[1]

with open('kitti_depth_val.txt', 'w') as f:

    for i in range(len(val_seq_left_list)):
        f.write(val_seq_left_list[i] + " " + val_seq_right_list[i] + " " + val_seq_s_list[i]  + " " +val_seq_gt_list[i])
        f.write('\n')


