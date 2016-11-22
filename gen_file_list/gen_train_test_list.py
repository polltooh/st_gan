import file_io

train_file_name = "/home/mscvadmin/eye_img_st/file_list/train_list1.txt"
test_file_name = "/home/mscvadmin/eye_img_st/file_list/train_list2.txt"


def get_image_list(file_name):
    f_list = file_io.read_file(file_name)
    return_list = list()
    for f in f_list:
        f_l = f.split(" ")
        return_list.append(f_l[0] + " " + f_l[1])
    return return_list

train_list = get_image_list(train_file_name)
test_list = get_image_list(test_file_name)

train_save_name = "../file_list/train_list1.txt"
test_save_name = "../file_list/test_list1.txt"

file_io.save_file(train_list, train_save_name)
file_io.save_file(test_list, test_save_name)
