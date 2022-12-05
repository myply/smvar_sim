import copy
import csv

filename = 'D:\\file\\pg\\xjw\\yolov5_excel\\excel\\yolov5s76.csv'
weight_segment_start_addr=65536*32*28
feature_segment_start_addr=65536*32*16+16384
max_feature_length=1024*32*32

data = []
with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    # header = next(csv_reader)        # 读取第一行每一列的标题
    for row in csv_reader:  # 将csv 文件中的数据保存到data中
        data.append(row)  # 选择某一列加入到data数组中
print(len(data))
# for i,row in  enumerate(data):
#     print(i+1,row)
    #### arrange the weight space
output_resue_flag_table=[]
for i,row in  enumerate(data):
    if(i==0):
        continue
    else:
        output_resue_flag_table.append([i-1])
    if(i==0 or i==1):
        continue
    elif(row[0].strip()[0:4]=='conv'):
        if (len(row[0])>4):
            output_resue_flag_table[int(row[0].strip()[5:-1])-2].append(i-1)
        ## i'th layer output is used as input in (i+1)'th layer
        else:
            output_resue_flag_table[i-1-1].append(i-1)
    elif(row[0].strip()[0:3]=='act'):
        output_resue_flag_table[i-1-1].append(i-1)
    elif (row[0].strip()[0:3] == 'add'):
        output_resue_flag_table[int(row[0].split(' ')[1][4:])-2].append(i-1)
        output_resue_flag_table[int(row[0].split(' ')[3][0:-1]) - 2].append(i-1)
    elif (row[0].strip()[0:10] == 'upsampling'):
        output_resue_flag_table[i -1- 1].append(i-1)
    elif (row[0].strip()[0:7] == 'maxpool'):
        output_resue_flag_table[i-1 - 1].append(i-1)
    elif (row[0].strip()[0:5] == 'slice'):
        output_resue_flag_table[i-1 - 1].append(i-1)
    elif (row[0].strip()[0:3] == 'cat'):
        if(row[0].strip().find('-')!=-1):
            output_resue_flag_table[int(row[0].strip().split('-')[0][4:]) - 2].append(i-1)
            output_resue_flag_table[int(row[0].strip().split('-')[0][4:]) - 2+1].append(i-1)
            output_resue_flag_table[int(row[0].strip().split('-')[0][4:]) - 2+2].append(i-1)
            output_resue_flag_table[int(row[0].strip().split('-')[0][4:]) - 2+3].append(i-1)
        else:
            output_resue_flag_table[int(row[0].strip().split(' ')[0][4:]) - 2].append(i-1)
            output_resue_flag_table[int(row[0].strip().split(' ')[2][0:-1]) - 2].append(i-1)

# print(len(output_resue_flag_table))
for i in range(len(output_resue_flag_table)):
        print(output_resue_flag_table[i])
##represent each layer's output' need for storage when i'th layer is computed,a row represent a layer is computed
output_store_requirement_table=[]
for i in range(len(output_resue_flag_table)):
    if i==0:
        output_store_requirement_table.append([0 for j in range(len(output_resue_flag_table))])
    else:
        output_store_requirement_table.append(copy.deepcopy((output_store_requirement_table[i-1])))
    output_store_requirement_table[i][i]=1
    for j in range(0,i-1):
        if(output_store_requirement_table[i][j]==1):
            flag=0
            ####this is leaf node
            if(len(output_resue_flag_table[j])==0):
                flag = 1
            for k in output_resue_flag_table[j]:
                if i<k:
                    flag=1
                    break
            if(flag==0):
                output_store_requirement_table[i][j] =0
print("output_store_requirement_table")
for i in range(len(output_store_requirement_table)):
    sum=0
    for j in range(len(output_store_requirement_table)):
        sum+=output_store_requirement_table[i][j]
    # print(i,output_store_requirement_table[i])
    print(i, sum)


weight_addr_list=[]
count=0

#### compute weight addr
temp_addr=weight_segment_start_addr
for i,row in  enumerate(data):
    if(row[0].strip()[0:4]=='conv'):
        weight_addr_list.append(temp_addr)
        # print(row, int(row[1]) * int(row[2]) * int(row[7]) * int(row[7]),count, temp_addr)
        temp_addr=temp_addr+int(row[1])*int(row[2])*int(row[7])*int(row[7])
        count += 1

#### compute bias addr
bias_addr_list=[]
count=0
for i,row in  enumerate(data):
    if(row[0].strip()[0:4]=='conv'):
        bias_addr_list.append(temp_addr)
        # print(int(row[2]), count,temp_addr)
        temp_addr=temp_addr+int(row[2])
        count += 1
print(len(bias_addr_list))


####res addr
#### compute max storage blocks need
max_number_of_storage_blocks=0
for i in range(len(output_store_requirement_table)):
    sum=0
    for j in range(len(output_store_requirement_table)):
        sum+=output_store_requirement_table[i][j]
    if(sum>max_number_of_storage_blocks):
        max_number_of_storage_blocks=sum

storage_used_flag=[-2 for i in range(max_number_of_storage_blocks)]
storage_used_flag[0]=-1
feature_addr_options_list=[]
for i in  range(max_number_of_storage_blocks):
    feature_addr_options_list.append(feature_segment_start_addr+i*max_feature_length)

####  temproer modify output_resue_flag_table
first_input_resue_flag_table=[0]
input_feature_addr_list=[]
output_feature_addr_list=[]
print(output_resue_flag_table)
for i,row in  enumerate(data):
    print(row,storage_used_flag)
    if(row[0].strip()[0:4]=='conv'):
        ####find the input feature addr
        if (len(row[0]) > 4):
            layer_index_of_input_src=int(row[0].strip()[5:-1]) - 2
        else:
            layer_index_of_input_src=i - 1-1
        flag=False
        for j in range(max_number_of_storage_blocks):
        #### find the src input stored in ddr
            if (storage_used_flag[j]==layer_index_of_input_src):
                input_feature_addr_list.append(feature_addr_options_list[j])
                flag=True
                break
        if(not flag):
            print("wrong!! couldn't find layer %d's input in ddr"%(i-1))

        ####free the space
        for j in range(max_number_of_storage_blocks):
            if(storage_used_flag[j]==-1):
                if(first_input_resue_flag_table[0]==i-1):
                    storage_used_flag[j]=-2
            elif(storage_used_flag[j]!=-2):
                if(output_resue_flag_table[storage_used_flag[j]][-1]==i-1):
                    storage_used_flag[j]=-2

        ####malloc output  feature addr
        flag=False
        for j in range(max_number_of_storage_blocks):
        #### find the src input stored in ddr
            if (storage_used_flag[j]==-2):
                output_feature_addr_list.append(feature_addr_options_list[j])
                storage_used_flag[j]=i-1
                flag=True
                break
        if(not flag):
            print("wrong!! couldn't find storage space for layer %d's output in ddr"%(i-1))



    elif (row[0].strip()[0:3] == 'act'):
        layer_index_of_input_src=i - 1-1
        flag=False
        for j in range(max_number_of_storage_blocks):
        #### find the src input stored in ddr
            if (storage_used_flag[j]==layer_index_of_input_src):
                input_feature_addr_list.append(feature_addr_options_list[j])
                output_feature_addr_list.append(feature_addr_options_list[j])
                storage_used_flag[j]=i-1
                flag=True
                break
        if(not flag):
            print("wrong!! couldn't find layer %d's input in ddr"%(i-1))
        ####in place operations, cantains no free process

        ####free the space
        for j in range(max_number_of_storage_blocks):
            if(storage_used_flag[j]==-1):
                if(first_input_resue_flag_table[0]==i-1):
                    storage_used_flag[j]=-2
            elif(storage_used_flag[j]!=-2):
                if(output_resue_flag_table[storage_used_flag[j]][-1]==i-1):
                    storage_used_flag[j]=-2

    elif (row[0].strip()[0:3] == 'add'):
        ####find the input feature addr
        layer_index_of_input_src1=int(row[0].split(' ')[1][4:]) - 2
        layer_index_of_input_src2 = int(row[0].split(' ')[3][0:-1]) - 2
        flag=False
        for j in range(max_number_of_storage_blocks):
        #### find the src input stored in ddr
            if (storage_used_flag[j]==layer_index_of_input_src1):
                input_feature_addr_list.append(feature_addr_options_list[j])
                flag=True
                break
        if(not flag):
            print("wrong!! couldn't find layer %d's input src1 in ddr"%(i-1))
        flag=False
        for j in range(max_number_of_storage_blocks):
            if (storage_used_flag[j]==layer_index_of_input_src2):
                input_feature_addr_list.append(feature_addr_options_list[j])
                flag=True
                break
        if(not flag):
            print("wrong!! couldn't find layer %d's input src2 in ddr"%(i-1))

        ####free the space
        for j in range(max_number_of_storage_blocks):
            if(storage_used_flag[j]==-1):
                if(first_input_resue_flag_table[0]==i-1):
                    storage_used_flag[j]=-2
            elif(storage_used_flag[j]!=-2):
                if(output_resue_flag_table[storage_used_flag[j]][-1]==i-1):
                    storage_used_flag[j]=-2

        ####malloc output  feature addr
        flag=False
        for j in range(max_number_of_storage_blocks):
        #### find the src input stored in ddr
            if (storage_used_flag[j]==-2):
                output_feature_addr_list.append(feature_addr_options_list[j])
                storage_used_flag[j]=i-1
                flag=True
                break
        if(not flag):
            print("wrong!! couldn't find storage space for layer %d's output in ddr"%(i-1))


    elif (row[0].strip()[0:10] == 'upsampling'):
        ####find the input feature addr
        layer_index_of_input_src=i - 1-1
        flag=False
        for j in range(max_number_of_storage_blocks):
        #### find the src input stored in ddr
            if (storage_used_flag[j]==layer_index_of_input_src):
                input_feature_addr_list.append(feature_addr_options_list[j])
                flag=True
                break
        if(not flag):
            print("wrong!! couldn't find layer %d's input in ddr"%(i-1))

        ####free the space
        for j in range(max_number_of_storage_blocks):
            if(storage_used_flag[j]==-1):
                if(first_input_resue_flag_table[0]==i-1):
                    storage_used_flag[j]=-2
            elif(storage_used_flag[j]!=-2):
                if(output_resue_flag_table[storage_used_flag[j]][-1]==i-1):
                    storage_used_flag[j]=-2

        ####malloc output  feature addr
        flag=False
        for j in range(max_number_of_storage_blocks):
        #### find the src input stored in ddr
            if (storage_used_flag[j]==-2):
                output_feature_addr_list.append(feature_addr_options_list[j])
                storage_used_flag[j]=i-1
                flag=True
                break
        if(not flag):
            print("wrong!! couldn't find storage space for layer %d's output in ddr"%(i-1))

    elif (row[0].strip()[0:7] == 'maxpool'):
        layer_index_of_input_src=i - 1-1
        flag=False
        for j in range(max_number_of_storage_blocks):
        #### find the src input stored in ddr
            if (storage_used_flag[j]==layer_index_of_input_src):
                input_feature_addr_list.append(feature_addr_options_list[j])
                flag=True
                break
        if(not flag):
            print("wrong!! couldn't find layer %d's input in ddr"%(i-1))

        ####free the space
        for j in range(max_number_of_storage_blocks):
            if(storage_used_flag[j]==-1):
                if(first_input_resue_flag_table[0]==i-1):
                    storage_used_flag[j]=-2
            elif(storage_used_flag[j]!=-2):
                if(output_resue_flag_table[storage_used_flag[j]][-1]==i-1):
                    storage_used_flag[j]=-2

        ####malloc output  feature addr
        flag=False
        for j in range(max_number_of_storage_blocks):
        #### find the src input stored in ddr
            if (storage_used_flag[j]==-2):
                output_feature_addr_list.append(feature_addr_options_list[j])
                storage_used_flag[j]=i-1
                flag=True
                break
        if(not flag):
            print("wrong!! couldn't find storage space for layer %d's output in ddr"%(i-1))


    elif (row[0].strip()[0:5] == 'slice'):
        layer_index_of_input_src=i - 1-1
        flag=False
        for j in range(max_number_of_storage_blocks):
        #### find the src input stored in ddr
            if (storage_used_flag[j]==layer_index_of_input_src):
                input_feature_addr_list.append(feature_addr_options_list[j])
                flag=True
                break
        if(not flag):
            print("wrong!! couldn't find layer %d's input in ddr"%(i-1))


        ####free the space
        for j in range(max_number_of_storage_blocks):
            if(storage_used_flag[j]==-1):
                if(first_input_resue_flag_table[0]==i-1):
                    storage_used_flag[j]=-2
            elif(storage_used_flag[j]!=-2):
                if(output_resue_flag_table[storage_used_flag[j]][-1]==i-1):
                    storage_used_flag[j]=-2

        ####malloc output  feature addr
        flag=False
        for j in range(max_number_of_storage_blocks):
        #### find the src input stored in ddr
            if (storage_used_flag[j]==-2):
                output_feature_addr_list.append(feature_addr_options_list[j])
                storage_used_flag[j]=i-1
                flag=True
                break
        if(not flag):
            print("wrong!! couldn't find storage space for layer %d's output in ddr"%(i-1))

    elif (row[0].strip()[0:3] == 'cat'):
        layer_index_of_input_src_list=[]
        if (row[0].strip().find('-') != -1):
            layer_index_of_input_src_list.append(int(row[0].strip().split('-')[0][4:]) - 2)
            layer_index_of_input_src_list.append(int(row[0].strip().split('-')[0][4:]) - 2+1)
            layer_index_of_input_src_list.append(int(row[0].strip().split('-')[0][4:]) - 2+2)
            layer_index_of_input_src_list.append(int(row[0].strip().split('-')[0][4:]) - 2+3)
        else:
            layer_index_of_input_src_list.append(int(row[0].strip().split(' ')[0][4:]) - 2)
            layer_index_of_input_src_list.append(int(row[0].strip().split(' ')[2][0:-1]) - 2)


        flag=False
        for k in range(len(layer_index_of_input_src_list)):
            for j in range(max_number_of_storage_blocks):
        #### find the src input stored in ddr
                if (storage_used_flag[j]==layer_index_of_input_src_list[k]):
                    input_feature_addr_list.append(feature_addr_options_list[j])
                    flag=True
                    break
        if(not flag):
            print("wrong!! couldn't find layer %d's input in ddr"%(i-1))

        ####free the space
        for j in range(max_number_of_storage_blocks):
            if(storage_used_flag[j]==-1):
                if(first_input_resue_flag_table[0]==i-1):
                    storage_used_flag[j]=-2
            elif(storage_used_flag[j]!=-2):
                if(output_resue_flag_table[storage_used_flag[j]][-1]==i-1):
                    storage_used_flag[j]=-2

        ####malloc output  feature addr
        flag=False
        for j in range(max_number_of_storage_blocks):
        #### find the src input stored in ddr
            if (storage_used_flag[j]==-2):
                output_feature_addr_list.append(feature_addr_options_list[j])
                storage_used_flag[j]=i-1
                flag=True
                break
        if(not flag):
            print("wrong!! couldn't find storage space for layer %d's output in ddr"%(i-1))
