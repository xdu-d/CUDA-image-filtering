#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <iostream> 
 
__global__ void addKernel(float **C,  float **A, int RR, int CC)     //主函数
{
	//threadIdx(.x/.y/.z代表几维索引)：线程所在block中各个维度上的线程号
	//blockDim.x代表块中x轴上的线程数量，blockDim.y代表块中y轴上的线程数量，blockDim.z代表块中z轴上的线程数量
	//blockIdx(.x/.y/.z代表几维索引)：块所在grid中各个维度上的块号

	int idx = threadIdx.x + blockDim.x * blockIdx.x; 
	int idy = threadIdx.y + blockDim.y * blockIdx.y;	

	float tem;
	int m;
	int n;
	int win;
	win=5;

	//对矩阵A进行均值滤波
	if (idx < CC-win && idy < RR-win && idx>=win && idy>=win)
	{
	  tem=0;
	  for (  m =-win; m <= win ; m++)
	   		for ( n = -win; n <= win; n++)
				{
					tem = tem + A[idy+m][idx+n];
				}
	//idx idy可以看做是坐标，在属于该if范围的所有坐标内扫描，这里看做每个坐标取一个窗
			C[idy][idx] =  tem/(2*win+1)/(2*win+1);

	}
}
__global__ void addKernel2(float **C, float **A, int RR, int CC)     //主函数
{
	//threadIdx(.x/.y/.z代表几维索引)：线程所在block中各个维度上的线程号
	//blockDim.x代表块中x轴上的线程数量，blockDim.y代表块中y轴上的线程数量，blockDim.z代表块中z轴上的线程数量
	//blockIdx(.x/.y/.z代表几维索引)：块所在grid中各个维度上的块号
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	float temp;
	int m;
	int n;
	int win;
	win = 5;
	float value[9];

	//对矩阵A进行中值滤波
	if (idx < CC - win && idy < RR - win && idx >= win && idy >= win)
	{
		value[0] = A[idy - 1][idx - 1];
		value[1] = A[idy - 1][idx];
		value[2] = A[idy - 1][idx + 1];
		value[3] = A[idy][idx - 1];
		value[4] = A[idy][idx];
		value[5] = A[idy][idx + 1];
		value[6] = A[idy + 1][idx - 1];
		value[7] = A[idy + 1][idx];
		value[8] = A[idy + 1][idx + 1];
		for (m = 0; m < 9; m++)
		{
			for (n = 0; n < 9 - m - 1; n++)
			{
				if (value[n] > value[n + 1])
				{
					temp = value[n];
					value[n] = value[n + 1];
					value[n + 1] = temp;
				}
			}
		}
		//idx idy可以看做是坐标，在属于该if范围的所有坐标内扫描，这里看做每个坐标取一个窗
		C[idy][idx] = value[4];
	}
}



void main()
{	
	//记录起始时间
	clock_t start, finish;
	float costtime;
	start = clock();	
				
    //定义矩阵行列值
	int Row, Col, a, b;
	Row= 1024;
	Col= 1024;
	a = 16;
	b = 16;
	//定义变量并分配CPU内存
	float **A = (float **)malloc(sizeof(float*) * Row);
	float **C = (float **)malloc(sizeof(float*) * Row);
	float *dataA = (float *)malloc(sizeof(float) * Row * Col);
	float *dataC = (float *)malloc(sizeof(float) * Row * Col);


	//定义变量
	float **d_A;
	float **d_C;
	float *d_dataA;
	float *d_dataC;

	float *temp,*temp1;
	FILE *ff;
	FILE *fid;

	
	//分配GPU内存
	cudaMalloc((void**)&d_A, sizeof(float **) * Row);
	cudaMalloc((void**)&d_C, sizeof(float **) * Row);
	cudaMalloc((void**)&d_dataA, sizeof(float) *Row*Col);
	cudaMalloc((void**)&d_dataC, sizeof(float) *Row*Col);

	//分配CPU内存
	temp = (float *)calloc(Row * Col, sizeof(float));
	temp1 = (float *)calloc(Col, sizeof(float));

	//初始化向量
    for (int i=0;i<Row * Col;i++)
		 temp[i]=0;

	//读文本数据
	fid = fopen("C:\\Users\\LENOVO\\Desktop\\Cuda编程相关2\\Meanfiltering\\11\\inputimage4", "rb");
	fread(temp, sizeof(float), Row * Col, fid);

	//将文本数值赋给dataA
	for (int i = 0; i<Row * Col; i++)
	{
		dataA[i]= temp[i];
	}
    fclose(fid);


	//将主机指针A指向设备数据位置，目的是让设备二级指针能够指向设备数据一级指针
	//A和dataA 都传到了设备上，建立二者的对应关系
	for (int i = 0; i < Row; i++)
	 {
		A[i] = d_dataA + Col * i;
		C[i] = d_dataC + Col * i;
	}
	
	//将CPU的数据上传到GPU中
	cudaMemcpy(d_A, A, sizeof(float*) * Row, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, C, sizeof(float*) * Row, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dataA, dataA, sizeof(float) * Row * Col, cudaMemcpyHostToDevice);

	//确定grid、block、thread的大小
	dim3 threadPerBlock(a,b); 
	dim3 blockNumber( (Col + threadPerBlock.x - 1)/ threadPerBlock.x, (Row + threadPerBlock.y - 1) / threadPerBlock.y );
	printf("Block(%d,%d)   Grid(%d,%d).\n", threadPerBlock.x, threadPerBlock.y, blockNumber.x, blockNumber.y);

	//把d_C、d_A、Row、Col的值输进主函数中运算
	//addKernel << <blockNumber, threadPerBlock >> > (d_C, d_A, Row, Col);
	addKernel2 << <blockNumber, threadPerBlock >> > (d_C, d_A, Row, Col);
	
	//拷贝计算数据-一级数据指针
	cudaMemcpy(dataC, d_dataC, sizeof(float) * Row * Col, cudaMemcpyDeviceToHost);


	//将处理好的数据写成文本格式输出
	ff = fopen("C:\\Users\\LENOVO\\Desktop\\Cuda编程相关2\\Meanfiltering\\11\\midd2.dat", "wb");
	for (int i = 0; i < Row; i++)
	{
		for (int j = 0;j < Col;j++)
		{
			temp1[j] = dataC[i * Col +j];
		}
		fwrite(temp1, sizeof(float), Col, ff);

	}
	fclose(ff);


	//获得结束时间,并显示计时结果
	finish = clock();         
	costtime = (float)(finish - start) / CLOCKS_PER_SEC; 
	printf("1024*1024图像均值滤波GPU处理Time=(%3.6f s)\n",costtime);

	//释放CPU、GPU内存
	free(A);
	free(C);
	free(dataA);
	free(dataC);
	cudaFree(d_A);
	cudaFree(d_C);
	cudaFree(d_dataA);
	cudaFree(d_dataC);

	getchar();
}
