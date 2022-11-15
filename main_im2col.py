
import random
import numpy as np
import math
import torch
import torch.nn as nn
Ci=80
Co=160
# Hi=368
# Wi=640
Hi=46
Wi=80
kernel_size=3
padding=1
stride=2
# Ho=int((Hi- kernel_size + 2 * m.conv.padding[1]) / m.conv.stride[1]) + 1
Ho=int((Hi- kernel_size + 2 * padding) / stride) + 1
Wo=int((Wi- kernel_size + 2 * padding) / stride) + 1

# M=1500
# N=128-63
# K=96*3-49
# n=3
# ###12 for im2col and 9 for spmm
# numer_of_32_n_12_in_a_vec_blocks=12

M=Co
N=kernel_size*kernel_size*Ci
K=Ho*Wo
n=kernel_size*kernel_size
###12 for im2col and 9 for spmm
numer_of_32_n_12_in_a_vec_blocks=12
print("M:%d N:%d K:%d"%(M,N,K))
sp_dtype=np.dtype([('row',np.int32),('col',np.int32),('val',np.float32)])
class DDR:
    input_feature=np.random.random((1,Ci,Hi,Wi))
    weight = np.random.random((Co,Ci, kernel_size, kernel_size))
    # bias=np.random.random((Co))
    bias = np.zeros((Co))
    output_feature=np.zeros((1,Co,Ho,Wo),dtype=np.float32)
    output_feature_correct = np.zeros((1,Co, Ho, Wo), dtype=np.float32)
    conv=nn.Conv2d(Ci,Co,kernel_size,stride,padding)


    mat1=np.empty((M,N),dtype=sp_dtype).tolist()
    mat2=np.empty((N,K),dtype=np.float32).tolist()
    ##divided to N/32 blocks, each block size is M*32
    mat1sparse=np.empty((math.ceil(M/1024),math.ceil(N/32)),dtype=sp_dtype).tolist()
    def __init__(self):

        weight_matrix_temp = self.weight.reshape(Co, Ci * kernel_size * kernel_size)
        for i in range(M):
            for j in range(N):
                self.mat1[i][j] = (i, j, weight_matrix_temp[i][j])

        self.mat2=self.im2col().tolist()

        for i in range(M):
            for j in range(N):
                if self.mat1[i][j][2]!=0:
                    if(isinstance(self.mat1sparse[int(i/1024)][int(j/ 32)],tuple)):
                        self.mat1sparse[int(i/1024)][int(j/ 32)]=[self.mat1[i][j]]
                    else:
                        self.mat1sparse[int(i/1024)][int(j/32)].append(self.mat1[i][j])
        ###get correct conv result
        self.conv.weight.data=torch.tensor(self.weight)
        self.conv.bias.data=torch.tensor(self.bias)
        self.output_feature_correct=self.conv.forward(torch.tensor(self.input_feature)).detach().numpy()
    def im2col(self):
        input=torch.tensor(self.input_feature)
        p_input = torch.nn.ZeroPad2d(padding).forward(input)
        input_matrix_temp = torch.rand(Ho*Wo,Ci *kernel_size*kernel_size )
        for i in range(Ho):
            for j in range(Wo):
                for k in range(Ci):
                    input_matrix_temp[i * Wo + j][k * kernel_size*kernel_size :(k + 1) * kernel_size*kernel_size ] = p_input[0, k,
                                                                                                     i * stride:i * stride + kernel_size,
                                                                                                     j * stride:j * stride + kernel_size].flatten()
        return input_matrix_temp.transpose(0,1).numpy()
    def show_mat1(self):
        for i in range(math.ceil(M/1024)):
            for j in range(math.ceil(N/32)):
                print("number of vals in (%d,%d) M*32 mat1 block is %d"%(i,j,len(self.mat1sparse[i][j])))
    def get_mat1_blocks_by_group(self,group_number,index_of_block_count_by_by_row):
        #mat1sparse layout is (N/32)*(M*32), each time a group of size n*(m*32) ,group_number is 0,1,
        return self.mat1sparse[index_of_block_count_by_by_row][group_number*n:(group_number+1)*n]
    def get_mat2_blocks(self,addr,block_size=32*n,block_bumber=12*numer_of_32_n_12_in_a_vec_blocks):
        #each time 96 blocks, each block is 32n,
        row_addr=int(addr/K)
        col_addr=addr-row_addr*K
        return np.array(self.mat2)[row_addr:row_addr+block_size,col_addr:col_addr+block_bumber].tolist()

    def get_correct_result(self):
        # a=np.zeros((M,N))
        # for i in range(M):
        #     for j in range(N):
        #         a[i][j]=self.mat1[i][j][2]
        #
        # b=np.array(self.mat2)
        # return np.matmul(a,b).tolist()

        return self.output_feature_correct.reshape(Co, Ho*Wo)


class MatRAM:
    data=[]
    def store_mat1_blocks_by_group(self,spmat_group):
        for i in spmat_group:
            self.data.append(i)
    def get_32_nozero_elements(self,addr,block_index):
        ##MatRAM data layout is (N/32)*(M*32)
        return self.data[block_index][addr:addr+32]


    def del_mat1_blocks_by_group(self):
        self.data.clear()
    def show_mat1(self):
        print("number of M*32 mat1 blocks is %d" % (len(self.data)))
        for i in self.data:
            print("number of vals in this M*32 mat1 block is %d"%(len(i)))

    def get_number_of_vals_in_one_block(self,block_index):
        if(block_index<len(self.data)):
            return len(self.data[block_index])
        else:
            return 0
class VecRAM:
    ##layout  (32*n)*96
    data=[]
    def store_vec_blocks(self, vec_blocks):
        for i in vec_blocks:
            self.data.append(i)
        ###padding to the times of 32n
        padding_row_size=math.ceil(len(vec_blocks)%(32*n))*32*n-len(vec_blocks)
        ### vec_blocks[0] is 96 except for the last cols
        if(padding_row_size>0):
            for i in range(padding_row_size):
                self.data.append([0 for j in range(len(vec_blocks[0]))])

    def get_data(self):
        return self.data

    def clear_vec_ram(self):
        self.data.clear()
    def show(self):
        print("the vec ram size si %d*%d"%(len(self.data),len(self.data[0])))
class VecRegs:
    ### data layout 12*8*(32*n)
    cu_list=[[] for i in range(12)]
    def store_vec_regs(self, vec_blocks):
        vec_blocks=np.array(vec_blocks)
        for i in range(12):
        #  vec_blocks layout (32*n)*96
        ## need reshape from (32*n)*8  8*(32*n)
        ## most of the time block_numbers=96
            block_numbers=len(vec_blocks[0])
            self.cu_list[i] = vec_blocks[:, i:i + block_numbers:12].transpose().tolist()
        max_12_in_blocks =0
        for i in range(12):
            if(len(self.cu_list[i])>max_12_in_blocks):
                max_12_in_blocks=len(self.cu_list[i])
        ## number_of_32_in_blocks is 96 except for the last block
        number_of_32_in_blocks=len(vec_blocks)

        for i in range(12):
            if(len(self.cu_list[i])<max_12_in_blocks):
                for j in range((max_12_in_blocks-len(self.cu_list[i]))):
                    self.cu_list[i].append([0 for k in range(number_of_32_in_blocks)])

    def get_12_32_vec(self,j,k):
        temp=np.array(self.cu_list)
        ### reshape from (12,1,32) to (12,32)
        return temp[:,j,k*32:(k+1)*32].reshape(12, 32).tolist()
    def clear_vec_reg(self):
        for i in range(len(self.cu_list)):
            self.cu_list[i].clear()
    def show(self):
        for i in range(12):
            print("len of culist[%d]" % (i))
            # print("len of culist[%d]: %d*%d"%(i,len(self.cu_list[i]),len(self.cu_list[i][-1])))
            print(np.array(self.cu_list[i]).shape)
class CUs:
    #32 none-zeros
    CU_src1=[]
    #12*32
    CU_src2=[]
    # x*32
    CU_src1_dense=[]
    ### res size should be 1024*96
    res_reg=np.zeros((M,K),dtype=np.int32).tolist()
    number_of_rows=0
    min_row_index=0
    max_row_index=0
    def store_src1(self,CU_src1):
        self.CU_src1=CU_src1

        self.sparse2dense()
    def store_src2(self,CU_src2):
        self.CU_src2=CU_src2

    def sparse2dense(self):
        self.number_of_rows = 0
        self.min_row_index = 0
        self.max_row_index = 0
        for i in CU_src1:
            if(0==self.number_of_rows):
                self.CU_src1_dense=[[0 for j in range(32)]]
                self.CU_src1_dense[-1][i[1]%32]=i[2]
                self.number_of_rows+=1
                self.min_row_index=i[0]
                self.max_row_index=i[0]
            elif (i[0]== self.max_row_index):
                self.CU_src1_dense[i[0]-self.min_row_index][i[1]%32] = i[2]
            elif(i[0]> self.max_row_index):
                for j in range(i[0]-self.max_row_index):
                    self.CU_src1_dense.append([0 for j in range(32)])
                    self.number_of_rows+=1
                self.max_row_index=i[0]
                self.CU_src1_dense[-1][i[1] % 32] = i[2]
            else:
                ##row  1024 to 0
                pass
    def CUs_compute(self,coladdr):
        temp1=np.array(self.CU_src1_dense)
        temp2 = np.array(self.CU_src2)
        for i in range(len(self.CU_src1_dense)):
            for j in range(12):
                ##  COMPUTE THE PADDING PART BUT NOT ADD
                if (coladdr+j<K):
                    self.res_reg[self.min_row_index+i][coladdr+j]+=np.inner(temp1[i],temp2[j])

    def show_src1(self):
        print("shape of dense src1:")
        print(np.array(self.CU_src1_dense).shape)
    def get_res(self):
        return self.res_reg

if __name__ == '__main__':
    DDR=DDR()
    MatRAM=MatRAM()
    VecRAM=VecRAM()
    VecRegs=VecRegs()
    CUs=CUs()
    # DDR.show_mat1()
    print("maxg:%d, maxh:%d, maxi:%d"%(math.ceil(M/1024),math.ceil(K / (12*numer_of_32_n_12_in_a_vec_blocks)),math.ceil(N / (32 * n))))
    for g in range(math.ceil(M/1024)):
        for h in range(math.ceil(K / (12*numer_of_32_n_12_in_a_vec_blocks))):
            for i in range(math.ceil(N / (32 * n))):
                print("g:%d, h:%d, i:%d"%(g,h,i))
                MatRAM.store_mat1_blocks_by_group(DDR.get_mat1_blocks_by_group(i,g))

                ##change when come to the loop of K/96
                vec_blocks = DDR.get_mat2_blocks(i * 32 * n * K + h * 12*numer_of_32_n_12_in_a_vec_blocks, block_size=32 * n, block_bumber=12*numer_of_32_n_12_in_a_vec_blocks)
                VecRAM.store_vec_blocks(vec_blocks)
                VecRegs.store_vec_regs(VecRAM.get_data())
                # VecRegs.show()
                number_of_12_in_one_blocks = math.ceil(len(vec_blocks[0]) / 12)
                ## number_of_12_in_one_blocks is less than 8 when come to the last block
                for j in range(number_of_12_in_one_blocks):
                    for k in range(n):
                        CU_src2 = VecRegs.get_12_32_vec(j, k)
                        CUs.store_src2(CU_src2)
                        number_of_vals_in_one_block = MatRAM.get_number_of_vals_in_one_block(k)
                        ###Round Up and the last need padding
                        for l in range(0, number_of_vals_in_one_block, 32):
                            CU_src1 = MatRAM.get_32_nozero_elements(addr=l, block_index=k)
                            CUs.store_src1(CU_src1)
                            CUs.CUs_compute(coladdr=h * 12*numer_of_32_n_12_in_a_vec_blocks + j * 12)
                MatRAM.del_mat1_blocks_by_group()
                VecRAM.clear_vec_ram()
                VecRegs.clear_vec_reg()

    result=CUs.get_res()
    correct_result=DDR.get_correct_result()
    for i in range(M):
        for j in range(k):
            if( (result[i][j]-correct_result[i][j])>0.0001 or (correct_result[i][j]-result[i][j])>0.0001 ):
                print("i:%d j:%d not equal! result is %d correct result is %d"%(i,j,result[i][j],correct_result[i][j]))


