import copy
import csv

filename = 'D:\\file\\pg\\xjw\\yolov5_excel\\excel\\yolov5s76.csv'
data = []
with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    # header = next(csv_reader)        # 读取第一行每一列的标题
    for row in csv_reader:  # 将csv 文件中的数据保存到data中
        data.append(row)  # 选择某一列加入到data数组中
print(len(data))
for i,row in  enumerate(data):
    print(i+1,row)
    #### arrange the weight space
output_resue_flag_table=[]
for i,row in  enumerate(data):
    if(i==0):
        continue
    else:
        output_resue_flag_table.append([])
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
        print(row[0].strip())
        if(row[0].strip().find('-')!=-1):
            print(row[0].strip().split('-'))
            output_resue_flag_table[int(row[0].strip().split('-')[0][4:]) - 2].append(i-1)
            output_resue_flag_table[int(row[0].strip().split('-')[0][4:]) - 2+1].append(i-1)
            output_resue_flag_table[int(row[0].strip().split('-')[0][4:]) - 2+2].append(i-1)
            output_resue_flag_table[int(row[0].strip().split('-')[0][4:]) - 2+3].append(i-1)
        else:
            output_resue_flag_table[int(row[0].strip().split(' ')[0][4:]) - 2].append(i-1)
            output_resue_flag_table[int(row[0].strip().split(' ')[2][0:-1]) - 2].append(i-1)

print(len(output_resue_flag_table))
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
# lines=[]
# lines.append("import torch.nn as nn\n")
# lines.append("import torch\n")
# lines.append("from functions import Slice\n")
# lines.append("@torch.no_grad()\n")
# lines.append("def yolov5():\n")
# lines.append("    layer0input=torch.zeros(1,3,192,320).detach()\n")
# ###i start from headers start from 1,and  layers start form 0,excel layers start from 2, so layer=i-1, or excelnumber-2
# for i,row in  enumerate(data):
#     outputName = "layer" + str(i-1) + "output"
#     if(row[0].strip()[0:4]=='conv'):
#         if (len(row[0])>4):
#             inputName = "layer" + str(int(row[0].strip()[5:-1])-2) + "output"
#         else:
#             inputName = "layer" + str(i-1-1) + "output"
#         lines.append('    '+outputName + '=nn.Conv2d('+'in_channels='+row[1]+',out_channels='+row[2]+',kernel_size='+row[7]+',stride='+row[9]+',padding='+row[8]+').forward('+inputName+')\n')
#     elif(row[0].strip()[0:3]=='act'):
#         inputName = "layer" + str(i-1-1) + "output"
#         lines.append('    ' + outputName + '=nn.SiLU().forward(' +inputName + ')' + '\n')
#     elif (row[0].strip()[0:3] == 'add'):
#         inputName1= "layer" + str(int(row[0].split(' ')[1][4:])-2)+ "output"
#         inputName2 = "layer" + str(int(row[0].split(' ')[3][0:-1]) - 2) + "output"
#         lines.append('    ' + outputName + '=' + inputName1 + '+' + inputName2 + '\n')
#     elif (row[0].strip()[0:10] == 'upsampling'):
#         inputName = "layer" + str(i - 1 - 1) + "output"
#         lines.append('    ' + outputName + '=nn.Upsample(size=None, scale_factor=2, mode=\'nearest\').forward(' + inputName + ')' + '\n')
#     elif (row[0].strip()[0:7] == 'maxpool'):
#         inputName = "layer" + str(int(row[0].strip()[8:-1])-2) + "output"
#         lines.append('    ' + outputName +'=nn.MaxPool2d(kernel_size=' + row[7] + ', stride='+ row[9] +', padding=' + row[8]  + ').forward(' + inputName+')' + '\n')
#     elif (row[0].strip()[0:5] == 'slice'):
#         inputName = "layer" + str(i - 1) + "input"
#         lines.append('    ' + outputName + '=Slice().forward(' + inputName + ')' + '\n')
#     elif (row[0].strip()[0:3] == 'cat'):
#         print(row[0].strip())
#         if(row[0].strip().find('-')!=-1):
#             print(row[0].strip().split('-'))
#             inputName1 = "layer" + str(int(row[0].strip().split('-')[0][4:]) - 2) + "output"
#             inputName2 = "layer" + str(int(row[0].strip().split('-')[0][4:]) - 2+1) + "output"
#             inputName3 = "layer" + str(int(row[0].strip().split('-')[0][4:]) - 2+2) + "output"
#             inputName4 = "layer" + str(int(row[0].strip().split('-')[0][4:]) - 2+3) + "output"
#             lines.append('    ' + outputName + '=' + 'torch.cat((' + inputName1 + ',' + inputName2 + ','+ inputName3+ ','+ inputName4 + '),1)' + '\n')
#         else:
#             inputName1 = "layer" + str(int(row[0].strip().split(' ')[0][4:]) - 2) + "output"
#             inputName2 = "layer" + str(int(row[0].strip().split(' ')[2][0:-1]) - 2) + "output"
#             lines.append('    ' + outputName + '=' +'torch.cat(('+inputName1+','+inputName2+'),1)'+'\n')
# lines.append('    ' + 'print(layer144output.shape)' + '\n')
# lines.append('    ' + 'print(layer145output.shape)' + '\n')
# lines.append('    ' + 'print(layer146output.shape)' + '\n')
# lines.append('yolov5()' + '\n')
# print(len(lines))
# file =open('gen_result_spmm.py','w')
# file.writelines(lines)
# file.close()