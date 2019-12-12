import base64
import re
import sys
import os
import random
import time
import json
import pandas as pd
from copy import deepcopy

Config = '/label_config'
Result = '/label_result'
Pair_Order_Key = 'pair_order'


def eye_info(file_path):
    ret = os.path.exists(file_path)
    if ret == False:
        print('config file lack')

    fsize = os.path.getsize(file_path)
    if fsize == 0:
        info_dic = {}
    else:
        with open(file_path, "r", encoding='utf-8') as label_config_file:
            tmp_str = label_config_file.read()
            info_dic = json.loads(tmp_str)
    return info_dic


def get_info(dir_path):
    ret_list = []

    ret = os.path.exists(dir_path + Config)
    if ret == False:
        print('config file lack')

    ret = os.path.exists(dir_path + Result)
    if ret == False:
        print('config file lack')

    order_list = []
    with open(dir_path + Config, "r", encoding='utf-8') as label_config_file:
        f_data_list = label_config_file.readlines()

        for a in f_data_list:
            order_list.append(a.strip())  # 去掉\n strip去掉头尾默认空格或换行符

    fsize = os.path.getsize(dir_path + Result)
    if fsize == 0:
        info_dic = {}
    else:
        with open(dir_path + Result, "r", encoding='utf-8') as label_config_file:
            tmp_str = label_config_file.read()
            de = base64.b64decode(tmp_str).decode("utf-8")
            info_dic = json.loads(de)

    front_key = ''
    key = ''
    for order_key in order_list:
        front_key = key
        key = order_key
        if order_key not in info_dic:
            if front_key == '':
                key = order_list[0]
                info_dic[key] = {}
                break;

            pair_wise_info_dic = info_dic[front_key]
            if len(pair_wise_info_dic[Pair_Order_Key]) != (len(pair_wise_info_dic) - 1):
                key = front_key
                break
            info_dic[key] = {}
            break
    pair_wise_info_dic = info_dic[key]
    if key == order_list[len(order_list) - 1] and len(pair_wise_info_dic) != 0 and (
            len(pair_wise_info_dic[Pair_Order_Key]) == (len(pair_wise_info_dic) - 1)):
        return False, 'Work finished', ret_list

    ret_list.append(order_list)
    ret_list.append(info_dic)
    sub_key = -1
    if len(pair_wise_info_dic) == 0:
        ret_list.append(key)
        ret_list.append(sub_key)
    elif (len(pair_wise_info_dic) - 1) < len(pair_wise_info_dic[Pair_Order_Key]):
        ret_list.append(key)
        sub_key = len(pair_wise_info_dic) - 1
        ret_list.append(sub_key)

    return True, "Initialize successfully.", ret_list


def eye_type(left_image_name, right_image_name, left_image_label, right_image_label, eye_info_sub_dic):
    if eye_info_sub_dic[left_image_name] == left_image_label and eye_info_sub_dic[right_image_name] == left_image_label:
        return None
    elif eye_info_sub_dic[left_image_name] == right_image_label and eye_info_sub_dic[
        right_image_name] == right_image_label:
        return None
    elif eye_info_sub_dic[left_image_name] == left_image_label and eye_info_sub_dic[
        right_image_name] == right_image_label:
        return 1
    elif eye_info_sub_dic[left_image_name] == right_image_label and eye_info_sub_dic[
        right_image_name] == left_image_label:
        return -1
    else:
        return None


def eye_info_preprocess(eye_info_dic, info_dict):
    for key in info_dict.keys():
        eye_key = key.replace('-repeat', '')
        if eye_key not in eye_info_dic:
            eye_info_dic[eye_key] = {}
        if Pair_Order_Key not in info_dict[key].keys():
            continue
        for image_pair in info_dict[key][Pair_Order_Key]:
            if image_pair[0] not in eye_info_dic[eye_key].keys():
                eye_info_dic[eye_key][image_pair[0]] = -1
            if image_pair[1] not in eye_info_dic[eye_key].keys():
                eye_info_dic[eye_key][image_pair[1]] = -1

    return eye_info_dic


def get_corr_dict(info_dict, eye_info_dic):
    dict_head = {'整图美感评价': [], '姿态美感': [], '面部美感': []}
    label_dic = {'open': 1, 'close': 0, 'none': -1}
    key_list = ['open', 'close', 'none']
    dict_list = []
    for left_key_index in range(0, len(key_list) - 1):
        for right_key_index in range(left_key_index + 1, len(key_list)):
            tmp_dict = {}
            new_key = key_list[left_key_index] + '-' + key_list[right_key_index]
            tmp_dict[new_key] = deepcopy(dict_head)
            tmp_dict[new_key][new_key] = []
            for key in info_dict.keys():
                for sub_key, sub_val in info_dict[key].items():
                    if sub_key == Pair_Order_Key:
                        continue

                    left_image_name = info_dict[key][Pair_Order_Key][int(sub_key)][0]
                    right_image_name = info_dict[key][Pair_Order_Key][int(sub_key)][1]
                    left_image_label = label_dic[key_list[left_key_index]]
                    right_image_label = label_dic[key_list[right_key_index]]
                    eye_key = key.replace('-repeat', '')
                    eye_info_sub_dic = eye_info_dic[eye_key]
                    ret_type = eye_type(left_image_name, right_image_name, left_image_label, right_image_label,
                                        eye_info_sub_dic)
                    if ret_type == None:
                        continue

                    for index_num in range(len(sub_val)):
                        if index_num > 2:
                            break

                        if sub_val[index_num] == 'A':
                            sub_val[index_num] = 1
                        elif sub_val[index_num] == '0':
                            sub_val[index_num] = 0
                        else:
                            sub_val[index_num] = -1

                    tmp_dict[new_key]['整图美感评价'].append(sub_val[0])
                    tmp_dict[new_key]['姿态美感'].append(sub_val[1])
                    tmp_dict[new_key]['面部美感'].append(sub_val[2])
                    tmp_dict[new_key][new_key].append(ret_type)
            dict_list.append(tmp_dict)

    return dict_list


def main():
    file_list = ['C:\\Users\\19938\\PycharmProjects\\label_GUI\\标注结果\\数据标注前20组20191203\\张亮20',
                 'C:\\Users\\19938\\PycharmProjects\\label_GUI\\标注结果\\数据标注前20组20191203\\许松20',
                 'C:\\Users\\19938\\PycharmProjects\\label_GUI\\标注结果\\数据标注前20组20191203\\肖力玮20',
                 'C:\\Users\\19938\\PycharmProjects\\label_GUI\\标注结果\\数据标注前20组20191203\\王剑峰20',
                 'C:\\Users\\19938\\PycharmProjects\\label_GUI\\标注结果\\数据标注前20组20191203\\柯飞龙20']

    eye_info_dic = eye_info(r'C:\Users\19938\PycharmProjects\Eye_Detection\eye_predict_dic')
    for file_path in file_list:
        ret, ret_str, ret_list = get_info(file_path)
        eye_info_dic = eye_info_preprocess(eye_info_dic, ret_list[1])

    df_list = []
    # argv = 'consistency'
    argv = 'corr'
    if argv == 'corr':
        df_dic = {}
        index = 0
        key_list = []
        for file_path in file_list:
            ret, ret_str, ret_list = get_info(file_path)
            corr_dic = get_corr_dict(ret_list[1], eye_info_dic)
            for val in corr_dic:
                for key in val.keys():
                    save_path = './data_analyse/' + file_path[file_path.rfind('\\') + 1:] + '-' + key + '.csv'
                    if key not in key_list:
                        key_list.append(key)
                        df_dic[key] = []
                    df = pd.DataFrame(data=val[key])
                    df_dic[key].append(df)
                    df.corr().to_csv(save_path, encoding='utf_8_sig')
                    print(file_path[file_path.rfind('\\') + 1:]+':'+key+'\t'+str(len(val[key]['整图美感评价'])))
        for key, df_list in df_dic.items():
            save_path = './data_analyse/' + 'hol_corr' + '-' + key + '.csv'
            hol_df = pd.concat(df_list)
            hol_df.corr().to_csv(save_path, encoding='utf_8_sig')


if __name__ == '__main__':
    main()
