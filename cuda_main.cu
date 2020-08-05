#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <iostream> 
 
__global__ void addKernel(float **C,  float **A, int RR, int CC)     //������
{
	//threadIdx(.x/.y/.z����ά����)���߳�����block�и���ά���ϵ��̺߳�
	//blockDim.x�������x���ϵ��߳�������blockDim.y�������y���ϵ��߳�������blockDim.z�������z���ϵ��߳�����
	//blockIdx(.x/.y/.z����ά����)��������grid�и���ά���ϵĿ��

	int idx = threadIdx.x + blockDim.x * blockIdx.x; 
	int idy = threadIdx.y + blockDim.y * blockIdx.y;	

	float tem;
	int m;
	int n;
	int win;
	win=5;

	//�Ծ���A���о�ֵ�˲�
	if (idx < CC-win && idy < RR-win && idx>=win && idy>=win)
	{
	  tem=0;
	  for (  m =-win; m <= win ; m++)
	   		for ( n = -win; n <= win; n++)
				{
					tem = tem + A[idy+m][idx+n];
				}
	//idx idy���Կ��������꣬�����ڸ�if��Χ������������ɨ�裬���￴��ÿ������ȡһ����
			C[idy][idx] =  tem/(2*win+1)/(2*win+1);

	}
}
__global__ void addKernel2(float **C, float **A, int RR, int CC)     //������
{
	//threadIdx(.x/.y/.z����ά����)���߳�����block�и���ά���ϵ��̺߳�
	//blockDim.x�������x���ϵ��߳�������blockDim.y�������y���ϵ��߳�������blockDim.z�������z���ϵ��߳�����
	//blockIdx(.x/.y/.z����ά����)��������grid�и���ά���ϵĿ��
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	float temp;
	int m;
	int n;
	int win;
	win = 5;
	float value[9];

	//�Ծ���A������ֵ�˲�
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
		//idx idy���Կ��������꣬�����ڸ�if��Χ������������ɨ�裬���￴��ÿ������ȡһ����
		C[idy][idx] = value[4];
	}
}



void main()
{	
	//��¼��ʼʱ��
	clock_t start, finish;
	float costtime;
	start = clock();	
				
    //�����������ֵ
	int Row, Col, a, b;
	Row= 1024;
	Col= 1024;
	a = 16;
	b = 16;
	//�������������CPU�ڴ�
	float **A = (float **)malloc(sizeof(float*) * Row);
	float **C = (float **)malloc(sizeof(float*) * Row);
	float *dataA = (float *)malloc(sizeof(float) * Row * Col);
	float *dataC = (float *)malloc(sizeof(float) * Row * Col);


	//�������
	float **d_A;
	float **d_C;
	float *d_dataA;
	float *d_dataC;

	float *temp,*temp1;
	FILE *ff;
	FILE *fid;

	
	//����GPU�ڴ�
	cudaMalloc((void**)&d_A, sizeof(float **) * Row);
	cudaMalloc((void**)&d_C, sizeof(float **) * Row);
	cudaMalloc((void**)&d_dataA, sizeof(float) *Row*Col);
	cudaMalloc((void**)&d_dataC, sizeof(float) *Row*Col);

	//����CPU�ڴ�
	temp = (float *)calloc(Row * Col, sizeof(float));
	temp1 = (float *)calloc(Col, sizeof(float));

	//��ʼ������
    for (int i=0;i<Row * Col;i++)
		 temp[i]=0;

	//���ı�����
	fid = fopen("C:\\Users\\LENOVO\\Desktop\\Cuda������2\\Meanfiltering\\11\\inputimage4", "rb");
	fread(temp, sizeof(float), Row * Col, fid);

	//���ı���ֵ����dataA
	for (int i = 0; i<Row * Col; i++)
	{
		dataA[i]= temp[i];
	}
    fclose(fid);


	//������ָ��Aָ���豸����λ�ã�Ŀ�������豸����ָ���ܹ�ָ���豸����һ��ָ��
	//A��dataA ���������豸�ϣ��������ߵĶ�Ӧ��ϵ
	for (int i = 0; i < Row; i++)
	 {
		A[i] = d_dataA + Col * i;
		C[i] = d_dataC + Col * i;
	}
	
	//��CPU�������ϴ���GPU��
	cudaMemcpy(d_A, A, sizeof(float*) * Row, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, C, sizeof(float*) * Row, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dataA, dataA, sizeof(float) * Row * Col, cudaMemcpyHostToDevice);

	//ȷ��grid��block��thread�Ĵ�С
	dim3 threadPerBlock(a,b); 
	dim3 blockNumber( (Col + threadPerBlock.x - 1)/ threadPerBlock.x, (Row + threadPerBlock.y - 1) / threadPerBlock.y );
	printf("Block(%d,%d)   Grid(%d,%d).\n", threadPerBlock.x, threadPerBlock.y, blockNumber.x, blockNumber.y);

	//��d_C��d_A��Row��Col��ֵ���������������
	//addKernel << <blockNumber, threadPerBlock >> > (d_C, d_A, Row, Col);
	addKernel2 << <blockNumber, threadPerBlock >> > (d_C, d_A, Row, Col);
	
	//������������-һ������ָ��
	cudaMemcpy(dataC, d_dataC, sizeof(float) * Row * Col, cudaMemcpyDeviceToHost);


	//������õ�����д���ı���ʽ���
	ff = fopen("C:\\Users\\LENOVO\\Desktop\\Cuda������2\\Meanfiltering\\11\\midd2.dat", "wb");
	for (int i = 0; i < Row; i++)
	{
		for (int j = 0;j < Col;j++)
		{
			temp1[j] = dataC[i * Col +j];
		}
		fwrite(temp1, sizeof(float), Col, ff);

	}
	fclose(ff);


	//��ý���ʱ��,����ʾ��ʱ���
	finish = clock();         
	costtime = (float)(finish - start) / CLOCKS_PER_SEC; 
	printf("1024*1024ͼ���ֵ�˲�GPU����Time=(%3.6f s)\n",costtime);

	//�ͷ�CPU��GPU�ڴ�
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
