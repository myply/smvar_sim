# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import random
import numpy as np
import math
M=511
N=128
K=96*3-49
n=2
sp_dtype=np.dtype([('row',np.int32),('col',np.int32),('val',np.int32)])
class DDR:
    mat1=np.empty((M,N),dtype=sp_dtype).tolist()
    mat2=np.empty((N,K),dtype=np.int32).tolist()
    ##divided to N/32 blocks, each block size is M*32
    mat1sparse=np.empty(int(N/32),dtype=sp_dtype).tolist()
    def __init__(self):
        ##init mat1 with 50% sparsity
        for i in range(M):
            for j in range(N):
                self.mat1[i][j]=(i,j,random.randint(0,10)*random.choice([0,0,0,0,0,0,0,1]))
        for i in  range(N):
           for j in range(K):
               self.mat2[i][j]=random.randint(1,10)
        for i in range(M):
            for j in range(N):
                if self.mat1[i][j][2]!=0:
                    if(isinstance(self.mat1sparse[int(j/ 32)],tuple)):
                        self.mat1sparse[int(j/ 32)]=[self.mat1[i][j]]
                    else:
                        self.mat1sparse[int(j/32)].append(self.mat1[i][j])
    def show_mat1(self):
        print("number of M*32 mat1 blocks is %d"%(len(self.mat1sparse)))
        for i in range(int(N/32)):
            print("number of vals in this M*32 mat1 block is %d"%(len(self.mat1sparse[i])))
    def get_mat1_blocks_by_group(self,group_number):
        #mat1sparse layout is (N/32)*(M*32), each time a group of size n*(m*32) ,group_number is 0,1,
        return self.mat1sparse[group_number*n:(group_number+1)*n]
    def get_mat2_blocks(self,addr,block_size=32*n,block_bumber=96):
        #each time 96 blocks, each block is 32n,
        row_addr=int(addr/K)
        col_addr=addr-row_addr*K
        return np.array(self.mat2)[row_addr:row_addr+block_size,col_addr:col_addr+block_bumber].tolist()

    def get_correct_result(self):
        a=np.zeros((M,N))
        for i in range(M):
            for j in range(N):
                a[i][j]=self.mat1[i][j][2]

        b=np.array(self.mat2)[:,0:96]
        # print(len(a[0]))
        # print(a[0])
        # print(len(b[0:N, 0]))
        # print(b[0:N,0])
        # print(np.inner(a[0],b[0:N,0]))
        return np.matmul(a,b).tolist()
class MatRAM:
    data=[]
    def store_mat1_blocks_by_group(self,spmat_group):
        for i in spmat_group:
            self.data.append(i)
    def get_32_nozero_elements(self,addr,block_index):
        ##MatRAM data layout is (N/32)*(M*32)

        # if (len(self.data[block_index][addr:addr+32]) != 32):
        #     print("loading lenr:%d" % (len(self.data[block_index][addr:addr+32])))
        #     print("number of vals:%d" % (len( self.data[block_index])))
        #     print("fetch addr:%d" % (addr))
        # print(len(self.data[block_index]))
        # print("addr:[%d,%d]"%(addr,addr+32))
        # print("real len:%d"%(len(self.data[block_index][addr:addr+32])))

        return self.data[block_index][addr:addr+32]


    def del_mat1_blocks_by_group(self):
        self.data.clear()
    def show_mat1(self):
        print("number of M*32 mat1 blocks is %d" % (len(self.data)))
        for i in self.data:
            print("number of vals in this M*32 mat1 block is %d"%(len(i)))

    def get_number_of_vals_in_one_block(self,block_index):
        return len(self.data[block_index])
class VecRAM:
    ##layout  (32*n)*96
    data=[]
    def store_vec_blocks(self, vec_blocks):
        for i in vec_blocks:
            self.data.append(i)

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
        # print("vec_block.shape:",vec_blocks.shape)
        # print(vec_blocks[:,0:96:12])
        # print(vec_blocks[:, 0:96:12].transpose())
        for i in range(12):
        #  vec_blocks layout (32*n)*96
        ## need reshape from (32*n)*8  8*(32*n)
        ## block_numbers=96
            block_numbers=len(vec_blocks[0])
            self.cu_list[i] = vec_blocks[:, i:i + block_numbers:12].transpose().tolist()
            if(len(self.cu_list[i])<8):
                for j in range((8-len(self.cu_list[i]))):
                    self.cu_list[i].append([0 for j in range(n*32)])

    def get_12_32_vec(self,j,k):
        # for j in range(8):
        #     for k in range(n):
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
        # print(self.CU_src1)
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
                # if ((self.min_row_index + i) == 313 and (coladdr + j) == 32):
                #     print(self.res_reg[self.min_row_index + i][coladdr + j])
                #     print('temp1')
                #     print(temp1)
                #     print('temp2')
                #     print(temp2)
                ##  COMPUTE THE PADDING PART BUT NOT ADD
                if (coladdr+j<K):
                    self.res_reg[self.min_row_index+i][coladdr+j]+=np.inner(temp1[i],temp2[j])
                # if((self.min_row_index+i)==313 and (coladdr+j)==62):
                #     print("i:%d,j:%d"%(i,j))
                #     print("vector:")
                #     print(temp1[i])
                #     print(temp2[j])
                #     print("result:")
                #     print(self.res_reg[self.min_row_index+i][coladdr+j])

    def show_src1(self):
        print("shape of dense src1:")
        print(np.array(self.CU_src1_dense).shape)
    def get_res(self):
        # print("res:")
        # for i in range(len(self.res_reg)):
        #     if(i<=33):
        #         print(self.res_reg[i])
        return self.res_reg
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DDR=DDR()
    MatRAM=MatRAM()
    VecRAM=VecRAM()
    VecRegs=VecRegs()
    CUs=CUs()
    DDR.show_mat1()
    for h in range(math.ceil(K/96)):
        for i in range(int(N / (32 * n))):
            MatRAM.store_mat1_blocks_by_group(DDR.get_mat1_blocks_by_group(i))
            # MatRAM.show_mat1()
            ##change when come to the loop of K/96
            vec_blocks=DDR.get_mat2_blocks(i * 32 * n * K+h*96, 32 * n, 96)
            VecRAM.store_vec_blocks(vec_blocks)
            VecRegs.store_vec_regs(VecRAM.get_data())
            # VecRegs.show()
            number_of_12_in_one_blocks=math.ceil(len(vec_blocks[0])/12)
            ## number_of_12_in_one_blocks is less than 8 when come to the last block
            ## CU_src1  32 none-zeros
            ## CU_src2
            for j in range(number_of_12_in_one_blocks):
                for k in range(n):
                    CU_src2 = VecRegs.get_12_32_vec(j, k)
                    CUs.store_src2(CU_src2)

                    number_of_vals_in_one_block = MatRAM.get_number_of_vals_in_one_block(k)
                    ###Round Up and the last need padding
                    for l in range(0, number_of_vals_in_one_block, 32):
                        CU_src1 = MatRAM.get_32_nozero_elements(addr=l, block_index=k)
                        # if(len(CU_src1)!=32):
                        #     print("*loading src1 len:%d"%(len(CU_src1)))
                        #     print("* of vals:%d"%(number_of_vals_in_one_block))
                        #     print("*fetch addr:%d" % (32*int(number_of_vals_in_one_block / 32)))
                        CUs.store_src1(CU_src1)
                        # CUs.show_src1()
                        CUs.CUs_compute(coladdr=h*96+j * 12)
            MatRAM.del_mat1_blocks_by_group()
            VecRAM.clear_vec_ram()
            VecRegs.clear_vec_reg()

    result=CUs.get_res()
    correct_result=DDR.get_correct_result()
    for i in range(M):
        for j in range(k):
            if(result[i][j]!=correct_result[i][j]):
                print("i:%d j:%d not equal! result is %d correct result is %d"%(i,j,result[i][j],correct_result[i][j]))
