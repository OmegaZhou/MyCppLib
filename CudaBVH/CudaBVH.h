#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utils.cuh"
#define STACK_SIZE 32
namespace ZLib
{
	// 点集
	struct PointsSet
	{
		double* x;
		double* y;
		double* z;
	};
	// 构成三角形的三个顶点
	struct TrianglesSet
	{
		int* v[3];
	};


	// type=0 代表叶子节点，type=1代表非叶节点
	struct BVHNodes
	{
		int* fa;
		bool* l_types;
		int* lefts;
		bool* r_types;
		int* rights;
	};

	// (x0,y0,z0)代表包围盒最小值
	// (x1,y1,z1)代表包围盒最大值
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
		// 非叶节点使用前n-1个节点，叶节点使用后n个节点
		AABBs aabbs;
		int triangle_num;
		int point_num;
	};

	__global__ void findIntersections(CudaBVH bvh);
	__global__ void findone(int n,CudaBVH bvh);
}


