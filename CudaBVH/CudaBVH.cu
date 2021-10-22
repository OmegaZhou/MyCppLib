#include "CudaBVH.h"
#include "thrust\sort.h"
#include "thrust\device_vector.h"
#include "thrust/execution_policy.h"
#define MAX_BLOCK_SIZE 32

__device__ __host__ static unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}
__device__ __host__ static double my_min(double a, double b)
{
    return a < b ? a : b;
}
__device__ __host__ static double my_max(double a, double b)
{
    return a > b ? a : b;
}

__device__ __host__ unsigned int ZLib::getMortonCode(double x, double y, double z)
{
    x = my_min(my_max(x * 1024.0f, 0.0f), 1023.0f);
    y = my_min(my_max(y * 1024.0f, 0.0f), 1023.0f);
    z = my_min(my_max(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}


__global__ void ZLib::findIntersections(CudaBVH bvh)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int n = bvh.getTriangleNum();
    if (offset < n) {
        int id = bvh.leaf_nodes.data_ids[offset];
        bvh.find(0, id, offset);
    }
}

__global__ void ZLib::findone(int n, CudaBVH bvh)
{
    int id = bvh.leaf_nodes.data_ids[n];
    bvh.find(0, id, n);
}

static void sortLeafNodes(ZLib::LeafNodes nodes, int n)
{
    thrust::device_ptr<unsigned int> codes(nodes.morton_codes);
    thrust::device_ptr<int> ids(nodes.data_ids);
    thrust::sort_by_key(codes, codes + n, ids, thrust::less<unsigned int>());
}
// 给叶节点赋值，以及给叶节点所使用的aabb赋值
__global__ void initLeafNodes(ZLib::LeafNodes leaf_nodes, ZLib::PointsSet points, ZLib::TrianglesSet triangles, ZLib::AABBs aabbs, int n, double min_s, double max_s)
{
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < n) {
        size_t aabb_offset = n - 1 + offset;
        leaf_nodes.data_ids[offset] = offset;
        double x[5];
        double y[5];
        double z[5];
        x[3] = y[3] = z[3] = INFINITY;
        x[4] = y[4] = z[4] = -INFINITY;
        double center_p[3];
        for (int i = 0; i < 3; ++i) {
            int id = triangles.v[i][offset];

            x[i] = points.x[id];
            x[3] = my_min(x[i], x[3]);
            x[4] = my_max(x[i], x[4]);

            y[i] = points.y[id];
            y[3] = my_min(y[i], y[3]);
            y[4] = my_max(y[i], y[4]);

            z[i] = points.z[id];
            z[3] = my_min(z[i], z[3]);
            z[4] = my_max(z[i], z[4]);
        }

        aabbs.x0[aabb_offset] = x[3];
        aabbs.x1[aabb_offset] = x[4];
        aabbs.y0[aabb_offset] = y[3];
        aabbs.y1[aabb_offset] = y[4];
        aabbs.z0[aabb_offset] = z[3];
        aabbs.z1[aabb_offset] = z[4];

        center_p[0] = (x[3] + x[4]) / 2.0f;
        center_p[1] = (y[3] + y[4]) / 2.0f;
        center_p[2] = (z[3] + z[4]) / 2.0f;
        for (int i = 0; i < 3; ++i) {
            center_p[i] -= min_s;
            center_p[i] /= (max_s - min_s);
        }
        leaf_nodes.morton_codes[offset] = ZLib::getMortonCode(center_p[0], center_p[1], center_p[2]);
    }
}

__device__ static int getSimilarity(const unsigned int* morton_codes,int n, int i, int j)
{
    if (i < 0 || i >= n || j < 0 || j >= n) {
        return -1;
    }
    unsigned int a = morton_codes[i];
    unsigned int b = morton_codes[j];
    int k = 0;
    if (a == b) {
        k = __clz(i ^ j);
    }
    return __clz(a ^ b) + k;
}
__device__ static int sign(int a)
{
    return a > 0 ? 1 : -1;
}

__global__ static void buildTree(ZLib::BVHNodes internal_nodes, ZLib::LeafNodes leaf_nodes, int triangle_num)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < triangle_num - 1) {
        unsigned int* morton_codes = leaf_nodes.morton_codes;
        int d = sign(getSimilarity(morton_codes, triangle_num, i, i + 1) - getSimilarity(morton_codes, triangle_num, i, i - 1));
        int sigma_min = getSimilarity(morton_codes, triangle_num, i, i - d);
        int l_max = 2;
        while (getSimilarity(morton_codes, triangle_num, i, i + l_max * d) > sigma_min) {
            l_max *= 2;
        }
        int l = 0;
        for (int t = l_max / 2; t >= 1; t /= 2) {
            if (getSimilarity(morton_codes, triangle_num, i, i + (l + t) * d) > sigma_min) {
                l += t;
            }
        }
        int j = i + l * d;
        int sigma_node = getSimilarity(morton_codes, triangle_num, i, j);
        /*int div = 2;
        int t = (l + div - 1) / div;
        int s = 0;
        while (t >= 1) {
            div *= 2;
            t = (l + div - 1) / div;
            if (getSimilarity(morton_codes, n, i, i + (s + t) * d) > sigma_node) {
                s += t;
            }
        }*/
        int left = my_min(i, j);
        int judge_p = left;
        int right = my_max(i, j);
        int mid = 0;

        while (left <= right) {
            mid = (left + right) / 2;
            if (getSimilarity(morton_codes, triangle_num, judge_p, mid) > sigma_node) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        int gamma = right;
        internal_nodes.lefts[i] = gamma;
        internal_nodes.rights[i] = gamma + 1;
        if (my_min(i, j) == gamma) {
            internal_nodes.l_types[i] = 0;
            leaf_nodes.fa[gamma] = i;
        } else {
            internal_nodes.l_types[i] = 1;
            internal_nodes.fa[gamma] = i;
        }
        if (my_max(i, j) == gamma + 1) {
            internal_nodes.r_types[i] = 0;
            leaf_nodes.fa[gamma + 1] = i;
        } else {
            internal_nodes.r_types[i] = 1;
            internal_nodes.fa[gamma + 1] = i;
            
        }
    }
}

__device__ static void mergeAABB(int result, int a, int b,const ZLib::AABBs& aabbs)
{
    aabbs.x0[result] = my_min(aabbs.x0[a], aabbs.x0[b]);
    aabbs.x1[result] = my_max(aabbs.x1[a], aabbs.x1[b]);
    aabbs.y0[result] = my_min(aabbs.y0[a], aabbs.y0[b]);
    aabbs.y1[result] = my_max(aabbs.y1[a], aabbs.y1[b]);
    aabbs.z0[result] = my_min(aabbs.z0[a], aabbs.z0[b]);
    aabbs.z1[result] = my_max(aabbs.z1[a], aabbs.z1[b]);
}

__device__ void printfAABB(ZLib::AABBs aabbs, int a)
{
    printf("AABB %d x0:%lf, x1:%lf, y0:%lf, y1:%lf z0:%lf, z1:%lf\n",a,aabbs.x0[a], aabbs.x1[a], aabbs.y0[a], aabbs.y1[a], aabbs.z0[a], aabbs.z1[a]);
}
__device__ void printfPoint(ZLib::PointsSet p, int a)
{
    printf("x:%lf y:%lf z:%lf \n", p.x[a],p.y[a],p.z[a]);
}
__device__ void printfTri(ZLib::PointsSet p,ZLib::TrianglesSet t, int a)
{
    printf("Triangle %d\n", a);
    for (int i = 0; i < 3; ++i) {
        printfPoint(p, t.v[i][a]);
    }
}
__global__ static void mergeAABBs(int* visit, ZLib::BVHNodes internal_nodes, ZLib::LeafNodes leaf_nodes, ZLib::AABBs aabbs, int n)
{
    
    int leaf_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_id < n) {
        int fa = leaf_nodes.fa[leaf_id];
        while (fa != -1) {
            int add_result = atomicAdd(visit + fa, 1);
            if (add_result == 1){
                int a = internal_nodes.lefts[fa];
                int b = internal_nodes.rights[fa];
                if (internal_nodes.l_types[fa] == 0) {
                    a = leaf_nodes.data_ids[a];
                    a += n - 1;

                }
                if (internal_nodes.r_types[fa] == 0) {
                    b = leaf_nodes.data_ids[b];
                    b += n - 1;

                }
                mergeAABB(fa, a, b, aabbs);
 
                fa = internal_nodes.fa[fa];

            } else {
                break;
            }
        }
    }
}

__global__ void printLeaf(ZLib::LeafNodes leaf_nodes, ZLib::PointsSet points, ZLib::TrianglesSet triangles, ZLib::AABBs aabbs, int n)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < n) {
        printf("data_id=%d, sorted id=%d\n", leaf_nodes.data_ids[offset], offset);
    }
}


ZLib::CudaBVH::CudaBVH(const PointsSet& _points, const TrianglesSet& _triangles, int _point_num, int _triangle_num, double min_s, double max_s):triangle_num(_triangle_num), point_num(_point_num)
{
    cudaEvent_t s, e;
    float t;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);
    allocDeviceMemory(_points, _triangles);
    cudaEventRecord(e);
    cudaEventSynchronize(s);
    cudaEventSynchronize(e);
    cudaEventElapsedTime(&t, s, e);
    printf("\tAllocate device memory: %lfms\n", t);

    cudaEventRecord(s);
    size_t block_num = (triangle_num + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
    initLeafNodes << <block_num, MAX_BLOCK_SIZE >> > (leaf_nodes, points, triangles, aabbs, triangle_num, min_s, max_s);
    HANDLE_ERROR(cudaPeekAtLastError());
    cudaEventRecord(e);
    cudaEventSynchronize(s);
    cudaEventSynchronize(e);
    cudaEventElapsedTime(&t, s, e);
    printf("\tInit leaf node: %lfms\n", t);

    cudaEventRecord(s);
    sortLeafNodes(leaf_nodes, triangle_num);
    cudaEventRecord(e);
    cudaEventSynchronize(s);
    cudaEventSynchronize(e);
    cudaEventElapsedTime(&t, s, e);
    printf("\tSort leaf node by morton code: %lfms\n", t);

    cudaEventRecord(s);
    buildTree << <block_num, MAX_BLOCK_SIZE >> > (internal_nodes, leaf_nodes, triangle_num);
    HANDLE_ERROR(cudaPeekAtLastError());
    cudaEventRecord(e);
    cudaEventSynchronize(s);
    cudaEventSynchronize(e);
    cudaEventElapsedTime(&t, s, e);
    printf("\tBuild BVH tree: %lfms\n", t);

    cudaEventRecord(s);
    int* visits = nullptr;
    HANDLE_ERROR(cudaMalloc(&visits, triangle_num * sizeof(*visits)));
    HANDLE_ERROR(cudaMemset(visits, 0, triangle_num * sizeof(*visits)));
    mergeAABBs << <block_num, MAX_BLOCK_SIZE >> > (visits, internal_nodes, leaf_nodes, aabbs, triangle_num);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaFree(visits));
    cudaEventRecord(e);
    cudaEventSynchronize(s);
    cudaEventSynchronize(e);
    cudaEventElapsedTime(&t, s, e);
    printf("\tMerge AABBs: %lfms\n", t);
}

__device__ __host__ int ZLib::CudaBVH::getTriangleNum() const
{
    return triangle_num;
}


__device__ static bool checkInBox(int tri_id, int aabb_id, const ZLib::AABBs& aabb)
{
    double x00 = aabb.x0[tri_id];
    double y00 = aabb.y0[tri_id];
    double z00 = aabb.z0[tri_id];
    double x11 = aabb.x1[tri_id];
    double y11 = aabb.y1[tri_id];
    double z11 = aabb.z1[tri_id];

    double x0 = aabb.x0[aabb_id];
    double y0 = aabb.y0[aabb_id];
    double z0 = aabb.z0[aabb_id];
    double x1 = aabb.x1[aabb_id];
    double y1 = aabb.y1[aabb_id];
    double z1 = aabb.z1[aabb_id];

    double r_x0 = my_max(x00, x0);
    double r_x1 = my_min(x11, x1);
    double r_y0 = my_max(y00, y0);
    double r_y1 = my_min(y11, y1);
    double r_z0 = my_max(z00, z0);
    double r_z1 = my_min(z11, z1);
    return r_x0 < r_x1&& r_y0 < r_y1&& r_z0 < r_z1;
}

__device__ void my_swap(int& a, int& b)
{
    a = a ^ b;
    b = a ^ b;
    a = a ^ b;
}

__device__ void ZLib::CudaBVH::find(int root, int triangle_id, int offset)
{
    int n = triangle_num - 1;
    if (internal_nodes.l_types[root] == 0) {
        int id1 = internal_nodes.lefts[root];
        int id2 = leaf_nodes.data_ids[id1];
        if (offset < id1 && checkIntersection(triangle_id, id2, points, triangles)) {
            int l = triangle_id < id2 ? triangle_id : id2;
            int r = triangle_id < id2 ? id2 : triangle_id;
            printf("contact found at (%d, %d)\n", l, r);
        }
    } else {
        int id2 = internal_nodes.lefts[root];
        if (offset < id2 && checkInBox(triangle_id + n, id2, aabbs)) {
            find(id2, triangle_id, offset);
        }
    }

    if (internal_nodes.r_types[root] == 0) {
        int id1 = internal_nodes.rights[root];
        int id2 = leaf_nodes.data_ids[id1];
        if (offset < id1 && checkIntersection(triangle_id, id2, points, triangles)) {
            int l = triangle_id < id2 ? triangle_id : id2;
            int r = triangle_id < id2 ? id2 : triangle_id;
            printf("contact found at (%d, %d)\n", l, r);
        }
    } else {
        int id2 = internal_nodes.rights[root];
        if (checkInBox(triangle_id + n, id2, aabbs)) {
            find(id2, triangle_id, offset);
        }
    }
}

void ZLib::CudaBVH::free()
{
    HANDLE_ERROR(cudaFree(internal_nodes.lefts));
    HANDLE_ERROR(cudaFree(internal_nodes.l_types));
    HANDLE_ERROR(cudaFree(internal_nodes.rights));
    HANDLE_ERROR(cudaFree(internal_nodes.r_types));
    HANDLE_ERROR(cudaFree(internal_nodes.fa));

    HANDLE_ERROR(cudaFree(leaf_nodes.data_ids));
    HANDLE_ERROR(cudaFree(leaf_nodes.morton_codes));
    HANDLE_ERROR(cudaFree(leaf_nodes.fa));

    HANDLE_ERROR(cudaFree(points.x));
    HANDLE_ERROR(cudaFree(points.y));
    HANDLE_ERROR(cudaFree(points.z));

    for (int i = 0; i < 3; ++i) {
        HANDLE_ERROR(cudaFree(triangles.v[i]));
    }
    
    HANDLE_ERROR(cudaFree(aabbs.x0));
    HANDLE_ERROR(cudaFree(aabbs.x1));
    HANDLE_ERROR(cudaFree(aabbs.y0));
    HANDLE_ERROR(cudaFree(aabbs.y1));
    HANDLE_ERROR(cudaFree(aabbs.z0));
    HANDLE_ERROR(cudaFree(aabbs.z1));

}

void ZLib::CudaBVH::allocDeviceMemory(const PointsSet& _points, const TrianglesSet& _triangles)
{
    // 分配points内存，并复制内存
    HANDLE_ERROR(cudaMalloc(&points.x, point_num * sizeof(*(points.x))));
    HANDLE_ERROR(cudaMalloc(&points.y, point_num * sizeof(*(points.y))));
    HANDLE_ERROR(cudaMalloc(&points.z, point_num * sizeof(*(points.z))));
    HANDLE_ERROR(cudaMemcpy(points.x, _points.x, point_num * sizeof(*(points.x)), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(points.y, _points.y, point_num * sizeof(*(points.y)), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(points.z, _points.z, point_num * sizeof(*(points.z)), cudaMemcpyHostToDevice));

    // 分配三角形内存，并复制
    for (int i = 0; i < 3; ++i) {
        HANDLE_ERROR(cudaMalloc(&triangles.v[i], triangle_num * sizeof(*(triangles.v[i]))));
        HANDLE_ERROR(cudaMemcpy(triangles.v[i], _triangles.v[i], triangle_num * sizeof(*(triangles.v[i])), cudaMemcpyHostToDevice));
    }
    

    // 分配叶子节点内存，数量与三角形数一致
    HANDLE_ERROR(cudaMalloc(&leaf_nodes.data_ids, triangle_num * sizeof(*(leaf_nodes.data_ids))));
    HANDLE_ERROR(cudaMalloc(&leaf_nodes.morton_codes, triangle_num * sizeof(*(leaf_nodes.morton_codes))));
    HANDLE_ERROR(cudaMalloc(&leaf_nodes.fa, triangle_num * sizeof(*(leaf_nodes.fa))));

    // 分配非叶节点，数量为三角形数-1
    size_t n = triangle_num - 1;
    HANDLE_ERROR(cudaMalloc(&internal_nodes.lefts, n * sizeof(*(internal_nodes.lefts))));
    HANDLE_ERROR(cudaMalloc(&internal_nodes.l_types, n * sizeof(*(internal_nodes.l_types))));
    HANDLE_ERROR(cudaMalloc(&internal_nodes.rights, n * sizeof(*(internal_nodes.rights))));
    HANDLE_ERROR(cudaMalloc(&internal_nodes.r_types, n * sizeof(*(internal_nodes.r_types))));
    HANDLE_ERROR(cudaMalloc(&internal_nodes.fa, n * sizeof(*(internal_nodes.fa))));
    // 设根节点的父节点为-1
    HANDLE_ERROR(cudaMemset(internal_nodes.fa, -1, sizeof(*(internal_nodes.fa))));

    // 分配AABB内存，数量为2*triangle_num-1
    size_t m = 2 * triangle_num - 1;
    HANDLE_ERROR(cudaMalloc(&aabbs.x0, m * sizeof(*(aabbs.x0))));
    HANDLE_ERROR(cudaMalloc(&aabbs.x1, m * sizeof(*(aabbs.x1))));
    HANDLE_ERROR(cudaMalloc(&aabbs.y0, m * sizeof(*(aabbs.y0))));
    HANDLE_ERROR(cudaMalloc(&aabbs.y1, m * sizeof(*(aabbs.y1))));
    HANDLE_ERROR(cudaMalloc(&aabbs.z0, m * sizeof(*(aabbs.z0))));
    HANDLE_ERROR(cudaMalloc(&aabbs.z1, m * sizeof(*(aabbs.z1))));

    
}
