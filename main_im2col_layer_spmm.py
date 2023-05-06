import copy
import csv


class compiler_backend():
    csv_filename = 'D:\\file\\pg\\xjw\\yolov5_excel\\excel\\875_736_1312.csv'
    cpp_filename = '875.cpp'
    csv_file_path = '"D:\\\\file\\\\pg\\\\backup\\\\yolo\\\\yolov5\\\\csv\\\\conv\\\\'
    # csv_file_path = '"csv\\\\conv\\\\'

    weight_segment_start_addr = 65536 * 32 * 96
    #### computed after weight is stored
    bias_segment_start_addr = 0
    feature_segment_start_addr = 65536 * 32
    max_feature_length = 80 * 480 * 640
    data = []
    ####tool list
    output_resue_flag_table = []
    output_store_requirement_table = []
    storage_used_flag = []
    feature_addr_options_list = []

    max_number_of_storage_blocks = 0
    first_input_resue_flag_table = [0]

    weight_addr_list = []
    bias_addr_list = []
    input_feature_addr_list = []
    output_feature_addr_list = []

    def __init__(self):
        self.csv_filename = 'D:\\file\\pg\\xjw\\yolov5_excel\\excel\\yolov5s.csv'
        self.cpp_filename = 'yolov5s.cpp'
        self.weight_segment_start_addr = 65536 * 32 * 96
        self.feature_segment_start_addr = 65536 * 32
        self.max_feature_length = 80 * 480 * 640

    def read_csv(self):
        with open(self.csv_filename) as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            # header = next(csv_reader)        # 读取第一行每一列的标题
            for row in csv_reader:  # 将csv 文件中的数据保存到data中
                self.data.append(row)  # 选择某一列加入到data数组中
        print("csv data:", len(self.data))

    def count_layers_output_resue(self):
        for i, row in enumerate(self.data):
            if (i == 0):
                continue
            else:
                self.output_resue_flag_table.append([i - 1])
            if (i == 0 or i == 1):
                continue
            elif (row[0].strip()[0:4] == 'conv'):
                if (len(row[0]) > 4):
                    self.output_resue_flag_table[int(row[0].strip()[5:-1]) - 2].append(i - 1)
                ## i'th layer output is used as input in (i+1)'th layer
                else:
                    self.output_resue_flag_table[i - 1 - 1].append(i - 1)
            elif (row[0].strip()[0:3] == 'act'):
                self.output_resue_flag_table[i - 1 - 1].append(i - 1)
            elif (row[0].strip()[0:3] == 'add'):
                row_splite = []
                for j in row[0].strip().split(' '):
                    if j != '':
                        row_splite.append(j)
                self.output_resue_flag_table[int(row_splite[0][4:]) - 2].append(i - 1)
                self.output_resue_flag_table[int(row_splite[1][0:-1]) - 2].append(i - 1)
            elif (row[0].strip()[0:10] == 'upsampling'):
                self.output_resue_flag_table[i - 1 - 1].append(i - 1)
            elif (row[0].strip()[0:7] == 'maxpool'):
                # print('test:',row[0].strip()[8:-1])
                self.output_resue_flag_table[int(row[0].strip()[8:-1]) - 2].append(i - 1)
            elif (row[0].strip()[0:5] == 'slice'):
                self.output_resue_flag_table[i - 1 - 1].append(i - 1)
            elif (row[0].strip()[0:3] == 'cat'):
                if (row[0].strip().find('-') != -1):
                    self.output_resue_flag_table[int(row[0].strip().split('-')[0][4:]) - 2].append(i - 1)
                    self.output_resue_flag_table[int(row[0].strip().split('-')[0][4:]) - 2 + 1].append(i - 1)
                    self.output_resue_flag_table[int(row[0].strip().split('-')[0][4:]) - 2 + 2].append(i - 1)
                    self.output_resue_flag_table[int(row[0].strip().split('-')[0][4:]) - 2 + 3].append(i - 1)
                else:
                    row_splite = []
                    for j in row[0].strip().split(' '):
                        if j != '':
                            row_splite.append(j)
                    self.output_resue_flag_table[int(row_splite[0][4:]) - 2].append(i - 1)
                    self.output_resue_flag_table[int(row_splite[1][0:-1]) - 2].append(i - 1)
        for i in range(len(self.output_resue_flag_table)):
            ## this is a leaf node, save this layer's  output
            if (len(self.output_resue_flag_table[i]) == 1):
                self.output_resue_flag_table[i].append(len(self.output_resue_flag_table))
        # print(len(output_resue_flag_table))
        for i in range(len(self.output_resue_flag_table)):
            print(self.output_resue_flag_table[i])

    def count_layers_storage_requirement_table(self):
        for i in range(len(self.output_resue_flag_table)):
            if i == 0:
                self.output_store_requirement_table.append([0 for j in range(len(self.output_resue_flag_table))])
            else:
                self.output_store_requirement_table.append(copy.deepcopy((self.output_store_requirement_table[i - 1])))
            self.output_store_requirement_table[i][i] = 1
            for j in range(0, i - 1):
                if (self.output_store_requirement_table[i][j] == 1):
                    flag = 0
                    ####this is leaf node
                    if (len(self.output_resue_flag_table[j]) == 0):
                        flag = 1
                    for k in self.output_resue_flag_table[j]:
                        if i < k:
                            flag = 1
                            break
                    if (flag == 0):
                        self.output_store_requirement_table[i][j] = 0
        # print("output_store_requirement_table")
        # for i in range(len(self.output_store_requirement_table)):
        #     sum = 0
        #     for j in range(len(self.output_store_requirement_table)):
        #         sum += self.output_store_requirement_table[i][j]
        #     # print(i,output_store_requirement_table[i])
        #     print(i, sum)
        #### compute max storage blocks need、
        for i in range(len(self.output_store_requirement_table)):
            sum = 0
            for j in range(len(self.output_store_requirement_table)):
                sum += self.output_store_requirement_table[i][j]
            if (sum > self.max_number_of_storage_blocks):
                self.max_number_of_storage_blocks = sum
        self.max_number_of_storage_blocks += 1
        print("max_number_of_storage_blocks")
        print(self.max_number_of_storage_blocks)

    def arrange_weight_addr(self):
        count = 0
        #### compute weight addr
        temp_addr = self.weight_segment_start_addr
        for i, row in enumerate(self.data):
            if (row[0].strip()[0:4] == 'conv'):
                self.weight_addr_list.append(temp_addr)
                Co_pad=int(row[2])
                if(int(row[2])%32!=0):
                    Co_pad = int(int(int(row[2])/32)+1)*32
                Ci_pad=int(row[1])
                if(int(row[1])%32!=0):
                    Ci_pad = int(int(int(row[1])/32)+1)*32
                # print(row, int(row[1]) * int(row[2]) * int(row[7]) * int(row[7]),count, temp_addr)
                temp_addr = temp_addr + Ci_pad * Co_pad* int(row[7]) * int(row[7])
                count += 1
        self.bias_segment_start_addr = temp_addr

    def arrange_bias_addr(self):
        #### compute bias addr
        count = 0
        temp_addr = self.bias_segment_start_addr
        for i, row in enumerate(self.data):
            if (row[0].strip()[0:4] == 'conv'):
                self.bias_addr_list.append(temp_addr)
                Co_pad=int(row[2])
                if(int(row[2])%32!=0):
                    Co_pad = int(int(int(row[2])/32)+1)*32
                # print(int(row[2]), count,temp_addr)
                temp_addr = temp_addr + Co_pad
                count += 1
        # print(len(bias_segment_start_addr))

    def arrange_feature_addr(self):
        #### compute bias addr
        self.storage_used_flag = [-2 for i in range(self.max_number_of_storage_blocks)]
        self.storage_used_flag[0] = -1

        for i in range(self.max_number_of_storage_blocks):
            self.feature_addr_options_list.append(self.feature_segment_start_addr + i * self.max_feature_length)

        ####  temproer modify output_resue_flag_table
        print(self.output_resue_flag_table)
        for i, row in enumerate(self.data):
            print(row, self.storage_used_flag)
            if (row[0].strip()[0:4] == 'conv'):
                ####find the input feature addr
                if (len(row[0]) > 4):
                    layer_index_of_input_src = int(row[0].strip()[5:-1]) - 2
                else:
                    layer_index_of_input_src = i - 1 - 1
                flag = False
                for j in range(self.max_number_of_storage_blocks):
                    #### find the src input stored in ddr
                    if (self.storage_used_flag[j] == layer_index_of_input_src):
                        self.input_feature_addr_list.append([self.feature_addr_options_list[j]])
                        flag = True
                        break
                if (not flag):
                    print("wrong!! couldn't find layer %d's input in ddr" % (i - 1))

                ####malloc output  feature addr
                flag = False
                for j in range(self.max_number_of_storage_blocks):
                    #### find the src input stored in ddr
                    if (self.storage_used_flag[j] == -2):
                        self.output_feature_addr_list.append(self.feature_addr_options_list[j])
                        self.storage_used_flag[j] = i - 1
                        flag = True
                        break
                if (not flag):
                    print("wrong!! couldn't find storage space for layer %d's output in ddr" % (i - 1))

                ####free the space
                for j in range(self.max_number_of_storage_blocks):
                    if (self.storage_used_flag[j] == -1):
                        if (self.first_input_resue_flag_table[0] == i - 1):
                            self.storage_used_flag[j] = -2
                    elif (self.storage_used_flag[j] != -2):
                        if (self.output_resue_flag_table[self.storage_used_flag[j]][-1] == i - 1):
                            self.storage_used_flag[j] = -2

            elif (row[0].strip()[0:3] == 'act'):
                layer_index_of_input_src = i - 1 - 1
                flag = False
                for j in range(self.max_number_of_storage_blocks):
                    #### find the src input stored in ddr
                    if (self.storage_used_flag[j] == layer_index_of_input_src):
                        self.input_feature_addr_list.append([self.feature_addr_options_list[j]])
                        self.output_feature_addr_list.append(self.feature_addr_options_list[j])
                        self.storage_used_flag[j] = i - 1
                        flag = True
                        break
                if (not flag):
                    print("wrong!! couldn't find layer %d's input in ddr" % (i - 1))
                ####in place operations, cantains no free process

                ####free the space
                for j in range(self.max_number_of_storage_blocks):
                    if (self.storage_used_flag[j] == -1):
                        if (self.first_input_resue_flag_table[0] == i - 1):
                            self.storage_used_flag[j] = -2
                    elif (self.storage_used_flag[j] != -2):
                        if (self.output_resue_flag_table[self.storage_used_flag[j]][-1] == i - 1):
                            self.storage_used_flag[j] = -2

            elif (row[0].strip()[0:3] == 'add'):
                ####find the input feature addr
                layer_index_of_input_src_list = []
                row_splite = []
                for j in row[0].strip().split(' '):
                    if j != '':
                        row_splite.append(j)
                layer_index_of_input_src_list.append(int(row_splite[0][4:]) - 2)
                layer_index_of_input_src_list.append(int(row_splite[1][0:-1]) - 2)
                temp_input_addr_list = []
                flag = False
                for k in range(len(layer_index_of_input_src_list)):
                    for j in range(self.max_number_of_storage_blocks):
                        #### find the src input stored in ddr
                        if (self.storage_used_flag[j] == layer_index_of_input_src_list[k]):
                            temp_input_addr_list.append(self.feature_addr_options_list[j])
                            flag = True
                            break
                if (not flag):
                    print("wrong!! couldn't find layer %d's input in ddr" % (i - 1))
                else:
                    self.input_feature_addr_list.append(temp_input_addr_list)

                ####malloc output  feature addr
                flag = False
                for j in range(self.max_number_of_storage_blocks):
                    #### find the src input stored in ddr
                    if (self.storage_used_flag[j] == -2):
                        self.output_feature_addr_list.append(self.feature_addr_options_list[j])
                        self.storage_used_flag[j] = i - 1
                        flag = True
                        break
                if (not flag):
                    print("wrong!! couldn't find storage space for layer %d's output in ddr" % (i - 1))

                ####free the space
                for j in range(self.max_number_of_storage_blocks):
                    if (self.storage_used_flag[j] == -1):
                        if (self.first_input_resue_flag_table[0] == i - 1):
                            self.storage_used_flag[j] = -2
                    elif (self.storage_used_flag[j] != -2):
                        if (self.output_resue_flag_table[self.storage_used_flag[j]][-1] == i - 1):
                            self.storage_used_flag[j] = -2


            elif (row[0].strip()[0:10] == 'upsampling'):
                ####find the input feature addr
                layer_index_of_input_src = i - 1 - 1
                flag = False
                for j in range(self.max_number_of_storage_blocks):
                    #### find the src input stored in ddr
                    if (self.storage_used_flag[j] == layer_index_of_input_src):
                        self.input_feature_addr_list.append([self.feature_addr_options_list[j]])
                        flag = True
                        break
                if (not flag):
                    print("wrong!! couldn't find layer %d's input in ddr" % (i - 1))

                ####malloc output  feature addr
                flag = False
                for j in range(self.max_number_of_storage_blocks):
                    #### find the src input stored in ddr
                    if (self.storage_used_flag[j] == -2):
                        self.output_feature_addr_list.append(self.feature_addr_options_list[j])
                        self.storage_used_flag[j] = i - 1
                        flag = True
                        break
                if (not flag):
                    print("wrong!! couldn't find storage space for layer %d's output in ddr" % (i - 1))

                ####free the space
                for j in range(self.max_number_of_storage_blocks):
                    if (self.storage_used_flag[j] == -1):
                        if (self.first_input_resue_flag_table[0] == i - 1):
                            self.storage_used_flag[j] = -2
                    elif (self.storage_used_flag[j] != -2):
                        if (self.output_resue_flag_table[self.storage_used_flag[j]][-1] == i - 1):
                            self.storage_used_flag[j] = -2

            elif (row[0].strip()[0:7] == 'maxpool'):
                layer_index_of_input_src = int(row[0].strip()[8:-1]) - 2
                flag = False
                for j in range(self.max_number_of_storage_blocks):
                    #### find the src input stored in ddr
                    if (self.storage_used_flag[j] == layer_index_of_input_src):
                        self.input_feature_addr_list.append([self.feature_addr_options_list[j]])
                        flag = True
                        break
                if (not flag):
                    print("wrong!! couldn't find layer %d's input in ddr" % (i - 1))

                ####malloc output  feature addr
                flag = False
                for j in range(self.max_number_of_storage_blocks):
                    #### find the src input stored in ddr
                    if (self.storage_used_flag[j] == -2):
                        self.output_feature_addr_list.append(self.feature_addr_options_list[j])
                        self.storage_used_flag[j] = i - 1
                        flag = True
                        break
                if (not flag):
                    print("wrong!! couldn't find storage space for layer %d's output in ddr" % (i - 1))

                ####free the space
                for j in range(self.max_number_of_storage_blocks):
                    if (self.storage_used_flag[j] == -1):
                        if (self.first_input_resue_flag_table[0] == i - 1):
                            self.storage_used_flag[j] = -2
                    elif (self.storage_used_flag[j] != -2):
                        if (self.output_resue_flag_table[self.storage_used_flag[j]][-1] == i - 1):
                            self.storage_used_flag[j] = -2


            elif (row[0].strip()[0:5] == 'slice'):
                layer_index_of_input_src = i - 1 - 1
                flag = False
                for j in range(self.max_number_of_storage_blocks):
                    #### find the src input stored in ddr
                    if (self.storage_used_flag[j] == layer_index_of_input_src):
                        self.input_feature_addr_list.append([self.feature_addr_options_list[j]])
                        flag = True
                        break
                if (not flag):
                    print("wrong!! couldn't find layer %d's input in ddr" % (i - 1))

                ####malloc output  feature addr
                flag = False
                for j in range(self.max_number_of_storage_blocks):
                    #### find the src input stored in ddr
                    if (self.storage_used_flag[j] == -2):
                        self.output_feature_addr_list.append(self.feature_addr_options_list[j])
                        self.storage_used_flag[j] = i - 1
                        flag = True
                        break
                if (not flag):
                    print("wrong!! couldn't find storage space for layer %d's output in ddr" % (i - 1))

                ####free the space
                for j in range(self.max_number_of_storage_blocks):
                    if (self.storage_used_flag[j] == -1):
                        if (self.first_input_resue_flag_table[0] == i - 1):
                            self.storage_used_flag[j] = -2
                    elif (self.storage_used_flag[j] != -2):
                        if (self.output_resue_flag_table[self.storage_used_flag[j]][-1] == i - 1):
                            self.storage_used_flag[j] = -2

            elif (row[0].strip()[0: 3] == 'cat'):
                layer_index_of_input_src_list = []
                if (row[0].strip().find('-') != -1):
                    layer_index_of_input_src_list.append(int(row[0].strip().split('-')[0][4:]) - 2)
                    layer_index_of_input_src_list.append(int(row[0].strip().split('-')[0][4:]) - 2 + 1)
                    layer_index_of_input_src_list.append(int(row[0].strip().split('-')[0][4:]) - 2 + 2)
                    layer_index_of_input_src_list.append(int(row[0].strip().split('-')[0][4:]) - 2 + 3)
                else:
                    row_splite = []
                    for j in row[0].strip().split(' '):
                        if j != '':
                            row_splite.append(j)
                    layer_index_of_input_src_list.append(int(row_splite[0][4:]) - 2)
                    layer_index_of_input_src_list.append(int(row_splite[1][0:-1]) - 2)
                temp_input_addr_list = []
                flag = False
                for k in range(len(layer_index_of_input_src_list)):
                    for j in range(self.max_number_of_storage_blocks):
                        #### find the src input stored in ddr
                        if (self.storage_used_flag[j] == layer_index_of_input_src_list[k]):
                            temp_input_addr_list.append(self.feature_addr_options_list[j])
                            flag = True
                            break
                if (not flag):
                    print("wrong!! couldn't find layer %d's input in ddr" % (i - 1))
                else:
                    self.input_feature_addr_list.append(temp_input_addr_list)

                ####malloc output  feature addr
                flag = False
                for j in range(self.max_number_of_storage_blocks):
                    #### find the src input stored in ddr
                    if (self.storage_used_flag[j] == -2):
                        self.output_feature_addr_list.append(self.feature_addr_options_list[j])
                        self.storage_used_flag[j] = i - 1
                        flag = True
                        break
                if (not flag):
                    print("wrong!! couldn't find storage space for layer %d's output in ddr" % (i - 1))

                ####free the space
                for j in range(self.max_number_of_storage_blocks):
                    if (self.storage_used_flag[j] == -1):
                        if (self.first_input_resue_flag_table[0] == i - 1):
                            self.storage_used_flag[j] = -2
                    elif (self.storage_used_flag[j] != -2):
                        if (self.output_resue_flag_table[self.storage_used_flag[j]][-1] == i - 1):
                            self.storage_used_flag[j] = -2

    def generate_code(self):
        lines = []
        print(len(self.input_feature_addr_list))
        print(len(self.output_feature_addr_list))
        print(len(self.weight_addr_list))
        lines.append('store_fea_from_csv_2_ddr(ddr,' + self.csv_file_path + 'input\\\\input0.csv",' +
                     self.data[1][1] + ',' + self.data[1][3] + ',' + self.data[1][5] + ',' + str(
            self.input_feature_addr_list[0][0]) + ');\n')

        conv_layer_index = 0
        for i, row in enumerate(self.data):
            if (row[0].strip()[0:4] == 'conv'):
                # csv_file_path='"D:\\\\file\\\\pg\\\\backup\\\\yolo\\\\yolov5\\\\csv\\\\conv\\\\'
                Co_pad=int(row[2])
                if(int(row[2])%32!=0):
                    Co_pad = int(int(int(row[2])/32)+1)*32
                lines.append('store_weight_from_csv_2_ddr(ddr,' + self.csv_file_path + 'weight\\\\weight'
                             + str(conv_layer_index) + '.csv",' + row[2] + ',' + row[1] + ',' + row[7] + ',' + str(
                    self.weight_addr_list[conv_layer_index]) + ');\n')

                lines.append('store_bias_from_csv_2_ddr(ddr,' + self.csv_file_path + 'bias\\\\bias'
                             + str(conv_layer_index) + '.csv",' + row[2]  + ',' + str(
                    self.bias_addr_list[conv_layer_index]) + ');\n')
                conv_layer_index += 1

        conv_layer_index = 0
        for i, row in enumerate(self.data):
            if (row[0].strip()[0:4] == 'conv'):
                Co_pad=int(row[2])
                if(int(row[2])%32!=0):
                    Co_pad = int(int(int(row[2])/32)+1)*32
                Ci_pad=int(row[1])
                if(int(row[1])%32!=0):
                    Ci_pad = int(int(int(row[1])/32)+1)*32
                if(conv_layer_index>=59):
                    lines.append('smvar_conv_code_nCi(ddr,'
                                 + str(self.input_feature_addr_list[i - 1][0]) + ',' + str(
                        self.output_feature_addr_list[i - 1]) + ',' + str(
                        self.weight_addr_list[conv_layer_index]) + ',' + str(self.bias_addr_list[conv_layer_index])
                                 + ',MatSram0,MatSram1,VecSram0,VecSram1,VecRegs0,VecRegs1,SumSram,ResVSram,ResMSram,vecReg,matReg,res_bus,sum,biasSram,'
                                 + str(Co_pad) + ',' + str(Ci_pad) + ',' + row[3] + ',' + row[5] + ',' + row[7] + ',' +
                                 row[
                                     9] + ',' + row[8] + ',1' + ',0' + ',isa_idx,isa_ddr);\n')
                else:
                    lines.append('smvar_conv_code_nCi(ddr,'
                             + str(self.input_feature_addr_list[i - 1][0]) + ',' + str(
                    self.output_feature_addr_list[i - 1]) + ',' + str(
                    self.weight_addr_list[conv_layer_index]) + ',' + str(self.bias_addr_list[conv_layer_index])
                             + ',MatSram0,MatSram1,VecSram0,VecSram1,VecRegs0,VecRegs1,SumSram,ResVSram,ResMSram,vecReg,matReg,res_bus,sum,biasSram,'
                             + str(Co_pad) + ',' + str(Ci_pad) + ',' + row[3] + ',' + row[5] + ',' + row[7] + ',' + row[
                                 9] + ',' + row[8] + ',1' + ',4' + ',isa_idx,isa_ddr);\n')

                # lines.append('smvar_bias_sim(ddr,' + str(self.output_feature_addr_list[i - 1]) + ',' + str(
                #     self.bias_addr_list[conv_layer_index]) + ','
                #              + str(Co_pad) + ',' + row[4] + ',' + row[6] + ');\n')
                conv_layer_index += 1
            elif (row[0].strip()[0:3] == 'act'):
                continue
                # lines.append('smvar_act_code(ddr,'
                #              + str(self.input_feature_addr_list[i - 1][0]) + ',' + row[1] + ',' + row[3] + ',' + row[
                #                  5] + ');\n')
            elif (row[0].strip()[0:3] == 'add'):
                lines.append('smvar_add_code(ddr,'
                             + str(self.input_feature_addr_list[i - 1][0]) + ',' + str(
                    self.input_feature_addr_list[i - 1][1]) + ',' + str(self.output_feature_addr_list[i - 1])
                             + ',' + row[1] + ',' + row[3] + ',' + row[5] + ');\n')
            elif (row[0].strip()[0:10] == 'upsampling'):
                lines.append('smvar_upsample_code(ddr,'
                             + str(self.input_feature_addr_list[i - 1][0]) + ',' + str(
                    self.output_feature_addr_list[i - 1])
                             + ',' + row[1] + ',' + row[3] + ',' + row[5] + ');\n')
            elif (row[0].strip()[0:7] == 'maxpool'):
                lines.append('smvar_maxpool_code(ddr,'
                             + str(self.input_feature_addr_list[i - 1][0]) + ',' + str(
                    self.output_feature_addr_list[i - 1])
                             + ',' + row[1] + ',' + row[3] + ',' + row[5] + ',' + row[7] + ',' + row[8] + ');\n')
            elif (row[0].strip()[0:5] == 'slice'):
                lines.append('smvar_slice_code(ddr,'
                             + str(self.input_feature_addr_list[i - 1][0]) + ',' + str(
                    self.output_feature_addr_list[i - 1])
                             + ',' + row[1] + ',' + row[3] + ',' + row[5] + ');\n')
            elif (row[0].strip()[0:3] == 'cat'):
                if (row[0].strip().find('-') != -1):
                    lines.append('smvar_cat4_code(ddr,'
                                 + str(self.input_feature_addr_list[i - 1][0]) + ',' + str(
                        self.input_feature_addr_list[i - 1][1])
                                 + ',' + str(self.input_feature_addr_list[i - 1][2]) + ',' + str(
                        self.input_feature_addr_list[i - 1][3]) + ',' + str(self.output_feature_addr_list[i - 1])
                                 + ',' + row[1] + ',' + row[3] + ',' + row[5] + ');\n')
                else:
                    lines.append('smvar_cat_code(ddr,'
                                 + str(self.input_feature_addr_list[i - 1][0]) + ',' + str(
                        self.input_feature_addr_list[i - 1][1]) + ',' + str(self.output_feature_addr_list[i - 1])
                                 + ',' + row[1] + ',' + row[3] + ',' + row[5] + ');\n')
        file = open(self.cpp_filename, 'w')
        file.writelines(lines)
        file.close()
        # for i in range(len(self.input_feature_addr_list)):
        #     print(self.input_feature_addr_list[i],self.output_feature_addr_list[i])

    def addr_compute(self):
        self.read_csv()
        self.count_layers_output_resue()
        ##represent each layer's output' need for storage when i'th layer is computed,a row represent a layer is computed
        self.count_layers_storage_requirement_table()

        self.arrange_weight_addr()
        self.arrange_bias_addr()
        ####res addr
        self.arrange_feature_addr()
        self.generate_code()


c = compiler_backend()
c.addr_compute()
