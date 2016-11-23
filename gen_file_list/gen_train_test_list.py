import file_io
import os

data_dir = "/home/geoff/eye_data/2016_11_18_14_17_2_extract/"

train_file_name = "/home/geoff/eye_data/file_list/train_list1.txt"
test_file_name = "/home/geoff/eye_data/file_list/test_list1.txt"


def get_image_list(file_name):
    f_list = file_io.read_file(file_name)
    return_list = list()
    for f in f_list:
        f_l = f.split(" ")
        b_name = os.path.basename(f_l[0])
        st_name = data_dir + b_name
        img_name = data_dir + b_name.replace("_st.data", "_1.jpg")
        return_list.append(st_name + " " + img_name)
    return return_list

train_list = get_image_list(train_file_name)
test_list = get_image_list(test_file_name)

train_save_name = "../file_list/train_list1.txt"
test_save_name = "../file_list/test_list1.txt"

file_io.save_file(train_list, train_save_name)
file_io.save_file(test_list, test_save_name)
