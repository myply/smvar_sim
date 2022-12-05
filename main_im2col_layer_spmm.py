import copy
import csv
class compiler_backend():
    csv_filename = 'D:\\file\\pg\\xjw\\yolov5_excel\\excel\\yolov5s76.csv'
    csv_filename = 'yolov5s76.cpp'
    weight_segment_start_addr = 65536 * 32 * 28
    #### computed after weight is stored
    bias_segment_start_addr=0
    feature_segment_start_addr = 65536 * 32 * 16 + 16384
    max_feature_length = 1024 * 32 * 32
    data = []
    ####tool list
    output_resue_flag_table = []
    output_store_requirement_table = []
    storage_used_flag = []
    feature_addr_options_list = []

    max_number_of_storage_blocks=0
    first_input_resue_flag_table = [0]

    weight_addr_list=[]
    bias_addr_list = []
    input_feature_addr_list = []
    output_feature_addr_list = []
    def __init__(self):
        self.csv_filename = 'D:\\file\\pg\\xjw\\yolov5_excel\\excel\\yolov5s76.csv'
        self.weight_segment_start_addr = 65536 * 32 * 28
        self.feature_segment_start_addr = 65536 * 32 * 16 + 16384
        self.max_feature_length = 1024 * 32 * 32
        self.cpp_filename=''
    def read_csv(self):
        with open(self.csv_filename) as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            # header = next(csv_reader)        # 读取第一行每一列的标题
            for row in csv_reader:  # 将csv 文件中的数据保存到data中
                self.data.append(row)  # 选择某一列加入到data数组中
        print("csv data:",len(self.data))
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
                self.output_resue_flag_table[int(row[0].split(' ')[1][4:]) - 2].append(i - 1)
                self.output_resue_flag_table[int(row[0].split(' ')[3][0:-1]) - 2].append(i - 1)
            elif (row[0].strip()[0:10] == 'upsampling'):
                self.output_resue_flag_table[i - 1 - 1].append(i - 1)
            elif (row[0].strip()[0:7] == 'maxpool'):
                self.output_resue_flag_table[i - 1 - 1].append(i - 1)
            elif (row[0].strip()[0:5] == 'slice'):
                self.output_resue_flag_table[i - 1 - 1].append(i - 1)
            elif (row[0].strip()[0:3] == 'cat'):
                if (row[0].strip().find('-') != -1):
                    self.output_resue_flag_table[int(row[0].strip().split('-')[0][4:]) - 2].append(i - 1)
                    self.output_resue_flag_table[int(row[0].strip().split('-')[0][4:]) - 2 + 1].append(i - 1)
                    self.output_resue_flag_table[int(row[0].strip().split('-')[0][4:]) - 2 + 2].append(i - 1)
                    self.output_resue_flag_table[int(row[0].strip().split('-')[0][4:]) - 2 + 3].append(i - 1)
                else:
                    self.output_resue_flag_table[int(row[0].strip().split(' ')[0][4:]) - 2].append(i - 1)
                    self.output_resue_flag_table[int(row[0].strip().split(' ')[2][0:-1]) - 2].append(i - 1)

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
    def arrange_weight_addr(self):
        count = 0
        #### compute weight addr
        temp_addr = self.weight_segment_start_addr
        for i, row in enumerate(self.data):
            if (row[0].strip()[0:4] == 'conv'):
                self.weight_addr_list.append(temp_addr)
                # print(row, int(row[1]) * int(row[2]) * int(row[7]) * int(row[7]),count, temp_addr)
                temp_addr = temp_addr + int(row[1]) * int(row[2]) * int(row[7]) * int(row[7])
                count += 1
        self.bias_segment_start_addr=temp_addr
    def arrange_bias_addr(self):
        #### compute bias addr
        count = 0
        temp_addr = self.bias_segment_start_addr
        for i, row in enumerate(self.data):
            if (row[0].strip()[0:4] == 'conv'):
                self.bias_addr_list.append(temp_addr)
                # print(int(row[2]), count,temp_addr)
                temp_addr = temp_addr + int(row[2])
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
                        self.input_feature_addr_list.append(self.feature_addr_options_list[j])
                        flag = True
                        break
                if (not flag):
                    print("wrong!! couldn't find layer %d's input in ddr" % (i - 1))

                ####free the space
                for j in range(self.max_number_of_storage_blocks):
                    if (self.storage_used_flag[j] == -1):
                        if (self.first_input_resue_flag_table[0] == i - 1):
                            self.storage_used_flag[j] = -2
                    elif (self.storage_used_flag[j] != -2):
                        if (self.output_resue_flag_table[self.storage_used_flag[j]][-1] == i - 1):
                            self.storage_used_flag[j] = -2

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



            elif (row[0].strip()[0:3] == 'act'):
                layer_index_of_input_src = i - 1 - 1
                flag = False
                for j in range(self.max_number_of_storage_blocks):
                    #### find the src input stored in ddr
                    if (self.storage_used_flag[j] == layer_index_of_input_src):
                        self.input_feature_addr_list.append(self.feature_addr_options_list[j])
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
                layer_index_of_input_src1 = int(row[0].split(' ')[1][4:]) - 2
                layer_index_of_input_src2 = int(row[0].split(' ')[3][0:-1]) - 2
                flag = False
                for j in range(self.max_number_of_storage_blocks):
                    #### find the src input stored in ddr
                    if (self.storage_used_flag[j] == layer_index_of_input_src1):
                        self.input_feature_addr_list.append(self.feature_addr_options_list[j])
                        flag = True
                        break
                if (not flag):
                    print("wrong!! couldn't find layer %d's input src1 in ddr" % (i - 1))
                flag = False
                for j in range(self.max_number_of_storage_blocks):
                    if (self.storage_used_flag[j] == layer_index_of_input_src2):
                        self.input_feature_addr_list.append(self.feature_addr_options_list[j])
                        flag = True
                        break
                if (not flag):
                    print("wrong!! couldn't find layer %d's input src2 in ddr" % (i - 1))

                ####free the space
                for j in range(self.max_number_of_storage_blocks):
                    if (self.storage_used_flag[j] == -1):
                        if (self.first_input_resue_flag_table[0] == i - 1):
                            self.storage_used_flag[j] = -2
                    elif (self.storage_used_flag[j] != -2):
                        if (self.output_resue_flag_table[self.storage_used_flag[j]][-1] == i - 1):
                            self.storage_used_flag[j] = -2

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


            elif (row[0].strip()[0:10] == 'upsampling'):
                ####find the input feature addr
                layer_index_of_input_src = i - 1 - 1
                flag = False
                for j in range(self.max_number_of_storage_blocks):
                    #### find the src input stored in ddr
                    if (self.storage_used_flag[j] == layer_index_of_input_src):
                        self.input_feature_addr_list.append(self.feature_addr_options_list[j])
                        flag = True
                        break
                if (not flag):
                    print("wrong!! couldn't find layer %d's input in ddr" % (i - 1))

                ####free the space
                for j in range(self.max_number_of_storage_blocks):
                    if (self.storage_used_flag[j] == -1):
                        if (self.first_input_resue_flag_table[0] == i - 1):
                            self.storage_used_flag[j] = -2
                    elif (self.storage_used_flag[j] != -2):
                        if (self.output_resue_flag_table[self.storage_used_flag[j]][-1] == i - 1):
                            self.storage_used_flag[j] = -2

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

            elif (row[0].strip()[0:7] == 'maxpool'):
                layer_index_of_input_src = i - 1 - 1
                flag = False
                for j in range(self.max_number_of_storage_blocks):
                    #### find the src input stored in ddr
                    if (self.storage_used_flag[j] == layer_index_of_input_src):
                        self.input_feature_addr_list.append(self.feature_addr_options_list[j])
                        flag = True
                        break
                if (not flag):
                    print("wrong!! couldn't find layer %d's input in ddr" % (i - 1))

                ####free the space
                for j in range(self.max_number_of_storage_blocks):
                    if (self.storage_used_flag[j] == -1):
                        if (self.first_input_resue_flag_table[0] == i - 1):
                            self.storage_used_flag[j] = -2
                    elif (self.storage_used_flag[j] != -2):
                        if (self.output_resue_flag_table[self.storage_used_flag[j]][-1] == i - 1):
                            self.storage_used_flag[j] = -2

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


            elif (row[0].strip()[0:5] == 'slice'):
                layer_index_of_input_src = i - 1 - 1
                flag = False
                for j in range(self.max_number_of_storage_blocks):
                    #### find the src input stored in ddr
                    if (self.storage_used_flag[j] == layer_index_of_input_src):
                        self.input_feature_addr_list.append(self.feature_addr_options_list[j])
                        flag = True
                        break
                if (not flag):
                    print("wrong!! couldn't find layer %d's input in ddr" % (i - 1))

                ####free the space
                for j in range(self.max_number_of_storage_blocks):
                    if (self.storage_used_flag[j] == -1):
                        if (self.first_input_resue_flag_table[0] == i - 1):
                            self.storage_used_flag[j] = -2
                    elif (self.storage_used_flag[j] != -2):
                        if (self.output_resue_flag_table[self.storage_used_flag[j]][-1] == i - 1):
                            self.storage_used_flag[j] = -2

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

            elif (row[0].strip()[0:3] == 'cat'):
                layer_index_of_input_src_list = []
                if (row[0].strip().find('-') != -1):
                    layer_index_of_input_src_list.append(int(row[0].strip().split('-')[0][4:]) - 2)
                    layer_index_of_input_src_list.append(int(row[0].strip().split('-')[0][4:]) - 2 + 1)
                    layer_index_of_input_src_list.append(int(row[0].strip().split('-')[0][4:]) - 2 + 2)
                    layer_index_of_input_src_list.append(int(row[0].strip().split('-')[0][4:]) - 2 + 3)
                else:
                    layer_index_of_input_src_list.append(int(row[0].strip().split(' ')[0][4:]) - 2)
                    layer_index_of_input_src_list.append(int(row[0].strip().split(' ')[2][0:-1]) - 2)

                flag = False
                for k in range(len(layer_index_of_input_src_list)):
                    for j in range(self.max_number_of_storage_blocks):
                        #### find the src input stored in ddr
                        if (self.storage_used_flag[j] == layer_index_of_input_src_list[k]):
                            self.input_feature_addr_list.append(self.feature_addr_options_list[j])
                            flag = True
                            break
                if (not flag):
                    print("wrong!! couldn't find layer %d's input in ddr" % (i - 1))

                ####free the space
                for j in range(self.max_number_of_storage_blocks):
                    if (self.storage_used_flag[j] == -1):
                        if (self.first_input_resue_flag_table[0] == i - 1):
                            self.storage_used_flag[j] = -2
                    elif (self.storage_used_flag[j] != -2):
                        if (self.output_resue_flag_table[self.storage_used_flag[j]][-1] == i - 1):
                            self.storage_used_flag[j] = -2

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
    def generate_code(self):
        lines=[]
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




c=compiler_backend()
c.addr_compute()