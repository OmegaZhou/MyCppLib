#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utils.cuh"
#define STACK_SIZE 32
namespace ZLib
{
	// �㼯
	struct PointsSet
	{
		double* x;
		double* y;
		double* z;
	};
	// ���������ε���������
	struct TrianglesSet
	{
		int* v[3];
	};


	// type=0 ����Ҷ�ӽڵ㣬type=1�����Ҷ�ڵ�
	struct BVHNodes
	{
		int* fa;
		bool* l_types;
		int* lefts;
		bool* r_types;
		int* rights;
	};

	// (x0,y0,z0)�����Χ����Сֵ
	// (x1,y1,z1)�����Χ�����ֵ
	struct AABBs
	{
		double* x0;
		double* y0;
		double* z0;

		double* x1;
		double* y1;
		double* z1;
	};


	struct  LeafNodes
	{
		int* fa;
		int* data_ids;
		unsigned int* morton_codes;
	};

	// must be [0,1]
	__device__ __host__ unsigned int getMortonCode(double x, double y, double z);
	__device__ __host__ bool checkIntersection(int a, int b, const ZLib::PointsSet& points, const ZLib::TrianglesSet& triangles);
	class CudaBVH
	{
	public:
		CudaBVH(const PointsSet& points, const TrianglesSet& triangles, int point_size, int triangle_size, double min_s, double max_s);
		__device__ __host__ int getTriangleNum()const;
		__device__ void find(int root, int triangle_id, int offset);
		void free();
	public:
		void allocDeviceMemory(const PointsSet& points, const TrianglesSet& triangles);
		LeafNodes leaf_nodes;
		BVHNodes internal_nodes;
		PointsSet points;
		TrianglesSet triangles;
		// ��Ҷ�ڵ�ʹ��ǰn-1���ڵ㣬Ҷ�ڵ�ʹ�ú�n���ڵ�
		AABBs aabbs;
		int triangle_num;
		int point_num;
	};

	__global__ void findIntersections(CudaBVH bvh);
	__global__ void findone(int n,CudaBVH bvh);
}


