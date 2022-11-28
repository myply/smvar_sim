
import random
import numpy as np
import math
import torch
import torch.nn as nn

class DDR:

    def __init__(self,args):
        self.args=args
        self.input_feature = np.random.random((1, args['Ci'], args['Hi'], args['Wi']))
        self.weight = np.random.random((args['Co'], args['Ci'], args['kernel_size'], args['kernel_size']))
        # bias=np.random.random((args['Co))
        self.bias = np.zeros((args['Co']))
        self.output_feature = np.zeros((1, args['Co'], args['Ho'], args['Wo']), dtype=np.float32)
        self.output_feature_correct = np.zeros((1, args['Co'], args['Ho'], args['Wo']), dtype=np.float32)
        self.conv = nn.Conv2d(args['Ci'], args['Co'], args['kernel_size'], args['stride'], args['padding'])

        self.mat1 = np.empty((args['M'], args['N']), dtype=args['sp_dtype']).tolist()
        self.mat2 = np.empty((args['N'], args['K']), dtype=np.float32).tolist()
        ##divided to N/32 blocks, each block size is M*32
        self.mat1sparse = np.empty((math.ceil(args['M']/ 1024), math.ceil(args['N'] / 32)), dtype=args['sp_dtype']).tolist()
        weight_matrix_temp = self.weight.reshape(args['Co'], args['Ci'] * args['kernel_size'] * args['kernel_size'])
        for i in range(args['M']):
            for j in range(args['N']):
                self.mat1[i][j] = (i, j, weight_matrix_temp[i][j])

        self.mat2=self.im2col().tolist()

        for i in range(args['M']):
            for j in range(args['N']):
                if self.mat1[i][j][2]!=0:
                    if(isinstance(self.mat1sparse[int(i/1024)][int(j/ 32)],tuple)):
                        self.mat1sparse[int(i/1024)][int(j/ 32)]=[self.mat1[i][j]]
                    else:
                        self.mat1sparse[int(i/1024)][int(j/32)].append(self.mat1[i][j])
        ###get correct conv result
        self.conv.weight.data=torch.tensor(self.weight)
        self.conv.bias.data=torch.tensor(self.bias)
        self.output_feature_correct=self.conv.forward(torch.tensor(self.input_feature)).detach().numpy()
        # print('ouputfeaturemap:')
        # print(self.output_feature_correct)
    def im2col(self):
        input=torch.tensor(self.input_feature)
        p_input = torch.nn.ZeroPad2d(self.args['padding']).forward(input)
        input_matrix_temp = torch.rand(self.args['Ho']*self.args['Wo'],self.args['Ci'] *self.args['kernel_size']*self.args['kernel_size'] )
        for i in range(self.args['Ho']):
            for j in range(self.args['Wo']):
                for k in range(self.args['Ci']):
                    input_matrix_temp[i * self.args['Wo'] + j][k * self.args['kernel_size']*self.args['kernel_size'] :(k + 1) * self.args['kernel_size']*self.args['kernel_size'] ] = p_input[0, k,
                                                                                                     i * self.args['stride']:i * self.args['stride'] + self.args['kernel_size'],
                                                                                                     j * self.args['stride']:j * self.args['stride'] + self.args['kernel_size']].flatten()
        return input_matrix_temp.transpose(0,1).numpy()
    def show_mat1(self):
        for i in range(math.ceil(self.args['M']/1024)):
            for j in range(math.ceil(self.args['N']/32)):
                print("number of vals in (%d,%d) M*32 mat1 block is %d"%(i,j,len(self.mat1sparse[i][j])))
    def get_mat1_blocks_by_group(self,group_number,index_of_block_count_by_by_row):
        #mat1sparse layout is (N/32)*(M*32), each time a group of size n*(m*32) ,
        return self.mat1sparse[index_of_block_count_by_by_row][group_number*self.args['n']:(group_number+1)*self.args['n']]
    def get_mat2_blocks(self,addr,block_size,block_bumber):
        #each time 96 blocks, each block is 32n,
        row_addr=int(addr/self.args['K'])
        col_addr=addr-row_addr*self.args['K']
        return np.array(self.mat2)[row_addr:row_addr+block_size,col_addr:col_addr+block_bumber].tolist()

    def get_correct_result(self):
        # a=np.zeros((M,N))
        # for i in range(M):
        #     for j in range(N):
        #         a[i][j]=self.mat1[i][j][2]
        #
        # b=np.array(self.mat2)
        # return np.matmul(a,b).tolist()
        return self.output_feature_correct.reshape(self.args['Co'], self.args['Ho']*self.args['Wo'])


class MatRAM:
    def __init__(self, args):
        self.data=[]
        self.args=args
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
    def __init__(self, args):
    ##layout  (32*n)*96 for spmm , (32*n)*(12*12) for conv
        self.data=[]
        self.args = args
    def store_vec_blocks(self, vec_blocks):
        for i in vec_blocks:
            self.data.append(i)
        ###padding to the times of 32n  may have problem
        padding_row_size=math.ceil(len(vec_blocks)/(32*self.args['n']))*32*self.args['n']-len(vec_blocks)
        # print('math.ceil(len(vec_blocks)')
        # print(len(vec_blocks))
        # print(math.ceil(len(vec_blocks)/(32*self.args['n'])))
        ### vec_blocks[0] is 96 except for the last cols
        if(padding_row_size>0):
            for i in range(padding_row_size):
                self.data.append([0 for j in range(len(vec_blocks[0]))])

    def get_data(self,index_of_12_clos_in_one_block):
        data=np.array(self.data)[:,index_of_12_clos_in_one_block*12:(index_of_12_clos_in_one_block+1)*12]
        return data.tolist()

    def clear_vec_ram(self):
        self.data.clear()
    def show(self):
        print("the vec ram size si %d*%d"%(len(self.data),len(self.data[0])))
class VecRegs:
    def __init__(self, args):
        ### data layout 12*8*(32*n)
        ### data layout change to 12*(32*n)
        self.cu_list=[[] for i in range(12)]
        self.args=args
    def store_vec_regs(self, vec_blocks):
        vec_blocks=np.array(vec_blocks)
        for i in range(12):
        #  vec_blocks layout (32*n)*96
        ## need reshape from (32*n)*8  8*(32*n)
        ## most of the time block_numbers=96


        ### change layout
        #  vec_blocks layout (32*n)*12
        ## need reshape from (32*n)  (32*n) in each reg
        ## most of the time block_numbers=12

            block_numbers=len(vec_blocks[0])
            # print('culist[%d]'%(i))
            # print(vec_blocks[:, i:i + block_numbers:12].transpose().shape)
            self.cu_list[i] = vec_blocks[:, i:i + block_numbers:12].transpose().tolist()


        ### the padding part is asme for conv and spmm,it can padding each cu into (1,288)
        ##for conv max_lenth_in_12_regs  is 1 after change the VecReg size,but some reg length mat be less than 1 at last
        max_lenth_in_12_regs =0
        for i in range(12):
            if(len(self.cu_list[i])>max_lenth_in_12_regs):
                max_lenth_in_12_regs=len(self.cu_list[i])
        # ## number_of_val_in_one_reg is always 32*n because the padding is done in VecRam

        number_of_vals_in_one_reg=len(vec_blocks)

        for i in range(12):
            if(len(self.cu_list[i])<max_lenth_in_12_regs):

                for j in range((max_lenth_in_12_regs-len(self.cu_list[i]))):
                    self.cu_list[i].append([0 for k in range(number_of_vals_in_one_reg)])

    def get_12_32_vec(self,j,k):
        temp=np.array(self.cu_list)
        ### reshape from (12,1,32) to (12,32)
        # print('temp.shape')
        # print(temp.shape)
        return temp[:,0,k*32:(k+1)*32].reshape(12, 32).tolist()
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
    def __init__(self, args):
        self.CU_src1=[]
        #12*32
        self.CU_src2=[]
        # x*32
        self.CU_src1_dense=[]
        ### res size should be 1024*96
        self.res_reg=np.zeros((args['M'],args['K']),dtype=np.int32).tolist()
        self.number_of_rows=0
        self.min_row_index=0
        self.max_row_index=0
        self.args=args
    def store_src1(self,CU_src1):
        self.CU_src1=CU_src1

        self.sparse2dense()
    def store_src2(self,CU_src2):
        self.CU_src2=CU_src2

    def sparse2dense(self):
        self.number_of_rows = 0
        self.min_row_index = 0
        self.max_row_index = 0
        for i in self.CU_src1:
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
                if (coladdr+j<self.args['K']):
                    self.res_reg[self.min_row_index+i][coladdr+j]+=np.inner(temp1[i],temp2[j])

    def show_src1(self):
        print("shape of dense src1:")
        print(np.array(self.CU_src1_dense).shape)
    def get_res(self):
        return self.res_reg

# if __name__ == '__main__':
class ConvLayer:
    def __init__(self):
        self.Ci = 80
        self.Co = 160
        self.Hi = 46
        self.Wi = 80
        self.kernel_size = 3
        self.padding = 1
        self.stride = 2
        # Ho=int((Hi- kernel_size + 2 * m.conv.padding[1]) / m.conv.stride[1]) + 1
        self.Ho = int((self.Hi - self.kernel_size + 2 * self.padding) / self.stride) + 1
        self.Wo = int((self.Wi - self.kernel_size + 2 * self.padding) / self.stride) + 1
        self.M = self.Co
        self.N = self.kernel_size * self.kernel_size * self.Ci
        self.K = self.Ho * self.Wo
        self.n = self.kernel_size * self.kernel_size
        ###12 for im2col and 9 for spmm
        self.numer_of_32_n_12_in_a_vec_blocks = 12
        print("M:%d N:%d K:%d" % (self.M, self.N, self.K))
        self.sp_dtype = np.dtype([('row', np.int32), ('col', np.int32), ('val', np.float32)])
        self.args = {"Ci": self.Ci, "Co": self.Co, "Hi": self.Hi, "Wi": self.Wi, "kernel_size": self.kernel_size, "padding": self.padding,
                "stride": self.stride, "Ho": self.Ho, "Wo": self.Wo, "M": self.M, "N": self.N, "K": self.K, "n": self.n,
                "numer_of_32_n_12_in_a_vec_blocks": self.numer_of_32_n_12_in_a_vec_blocks, 'sp_dtype': self.sp_dtype}

        self.DDR=DDR(self.args)
        self.MatRAM=MatRAM(self.args)
        self.VecRAM=VecRAM(self.args)
        self.VecRegs=VecRegs(self.args)
        self.CUs=CUs(self.args)

    def conv_layer_compute(self):
        print("maxg:%d, maxh:%d, maxi:%d" % (
            math.ceil(self.M / 1024), math.ceil(self.K / (12 * self.numer_of_32_n_12_in_a_vec_blocks)),
            math.ceil(self.N / (32 * self.n))))
        for g in range(math.ceil(self.M / 1024)):
            for h in range(math.ceil(self.K / (12 * self.numer_of_32_n_12_in_a_vec_blocks))):
                for i in range(math.ceil(self.N / (32 * self.n))):
                    print("g:%d, h:%d, i:%d" % (g, h, i))
                    self.MatRAM.store_mat1_blocks_by_group(self.DDR.get_mat1_blocks_by_group(i, g))

                    ##change when come to the loop of K/96
                    vec_blocks = self.DDR.get_mat2_blocks(i * 32 * self.n * self.K + h * 12 * self.numer_of_32_n_12_in_a_vec_blocks,
                                                     block_size=32 * self.n,
                                                     block_bumber=12 * self.numer_of_32_n_12_in_a_vec_blocks)
                    self.VecRAM.store_vec_blocks(vec_blocks)
                    ### delete if change is finished all
                    ##self.VecRegs.store_vec_regs(self.VecRAM.get_data())
                    # VecRegs.show()
                    number_of_12_cols_in_one_vec_block = math.ceil(len(vec_blocks[0]) / 12)
                    # print('number_of_12_cols_in_one_vec_block',number_of_12_cols_in_one_vec_block)
                    ## for spmm number_of_12_cols_in_one_vec_block is less than 8 when come to the last block,for conv is 12
                    for j in range(number_of_12_cols_in_one_vec_block):
                        ##for conv VecReg  save 12*9*32 each time (12*9*32*32 before,same size as vecSram)
                        self.VecRegs.store_vec_regs(self.VecRAM.get_data(j))
                        for k in range(self.n):
                            CU_src2 = self.VecRegs.get_12_32_vec(j,k)
                            self.CUs.store_src2(CU_src2)
                            number_of_vals_in_one_M32_mat_block = self.MatRAM.get_number_of_vals_in_one_block(k)
                            # print('number_of_vals_in_one_M32_mat_block',number_of_vals_in_one_M32_mat_block)
                            ###Round Up and the last need padding
                            for l in range(0, number_of_vals_in_one_M32_mat_block, 32):
                                CU_src1 = self.MatRAM.get_32_nozero_elements(addr=l, block_index=k)
                                self.CUs.store_src1(CU_src1)
                                self.CUs.CUs_compute(coladdr=h * 12 * self.numer_of_32_n_12_in_a_vec_blocks + j * 12)
                    self.MatRAM.del_mat1_blocks_by_group()
                    self.VecRAM.clear_vec_ram()
                    self.VecRegs.clear_vec_reg()

        result = self.CUs.get_res()
        correct_result = self.DDR.get_correct_result()
        for i in range(self.M):
            for j in range(self.K):
                if ((result[i][j] - correct_result[i][j]) > 0.0001 or (correct_result[i][j] - result[i][j]) > 0.0001):
                    print("i:%d j:%d not equal! result is %f correct result is %f" % (
                    i, j, result[i][j], correct_result[i][j]))
layer0=ConvLayer()
layer0.conv_layer_compute()