#include <cuda_runtime.h>
#include <cublas_v2.h>

float* dev_u, * dev_v, * dev_u_prev, * dev_v_prev, * dev_p, * dev_p_prev, * dev_D_x, * dev_D_y, * dev_D_xx, * dev_D_yy;

float* dev_dUdx, * dev_dUdy, * dev_udUdx, * dev_vdUdy, * dev_UnablaU;

__constant__ float dev_dx, dev_dy, dev_dt, dev_mu;

void allocate_2d_array_on_GPU(float* u, float* v, float* u_prev, float* v_prev, float* p, float* p_prev, float*& D_x, float*& D_y, float*& D_xx, float*& D_yy, int Nx, int Ny, float dx, float dy, float dt, float mu)
{
	// Выделение памяти на device
	cudaMalloc((void**)&dev_u, Nx * Ny * sizeof(float));
	cudaMalloc((void**)&dev_v, Nx * Ny * sizeof(float));
	cudaMalloc((void**)&dev_u_prev, Nx * Ny * sizeof(float));
	cudaMalloc((void**)&dev_v_prev, Nx * Ny * sizeof(float));
	cudaMalloc((void**)&dev_p, Nx * Ny * sizeof(float));
	cudaMalloc((void**)&dev_p_prev, Nx * Ny * sizeof(float));

	cudaMalloc((void**)&dev_D_x, Nx * Ny * sizeof(float));
	cudaMalloc((void**)&dev_D_y, Nx * Ny * sizeof(float));
	cudaMalloc((void**)&dev_D_xx, Nx * Ny * sizeof(float));
	cudaMalloc((void**)&dev_D_yy, Nx * Ny * sizeof(float));

	// Копирование массивов из ОЗУ в память device
	cudaMemcpy(dev_u, &u[0], Nx * Ny * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v, &v[0], Nx * Ny * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_u_prev, dev_u, Nx * Ny * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_v_prev, dev_v, Nx * Ny * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_p, &p[0], Nx * Ny * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_p_prev, dev_p, Nx * Ny * sizeof(float), cudaMemcpyDeviceToDevice);
	
	cudaMemcpy(dev_D_x, &D_x[0], Nx * Ny * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_D_y, &D_y[0], Nx * Ny * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_D_xx, &D_xx[0], Nx * Ny * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_D_yy, &D_yy[0], Nx * Ny * sizeof(float), cudaMemcpyHostToDevice);
}

void free_2d_array_on_GPU()
{
    cudaFree(dev_u);
    cudaFree(dev_v);
    cudaFree(dev_u_prev);
    cudaFree(dev_v_prev);
    cudaFree(dev_p);
    cudaFree(dev_p_prev);
    cudaFree(dev_D_x);
    cudaFree(dev_D_y);
    cudaFree(dev_D_xx);
    cudaFree(dev_D_yy);
}

// CUDA-ядро для поэлементного умножения матриц A и B, результат сохраняется в C
__global__ void elementwise_multiply(const float* A, const float* B, float* C, int Nx, int Ny)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < Nx && j < Ny)
	{
		int index = i * Ny + j;
		C[index] = A[index] * B[index]; // Поэлементное умножение
	}
}

// Функция для вызова CUDA-ядра
void launch_elementwise_multiply(const float* A, const float* B, float* C, int Nx, int Ny)
{
	dim3 blockSize(16, 16); // Размер блока (16x16 потоков)
	dim3 gridSize((Nx + blockSize.x - 1) / blockSize.x, (Ny + blockSize.y - 1) / blockSize.y);

	elementwise_multiply << <gridSize, blockSize >> > (A, B, C, Nx, Ny);

	cudaDeviceSynchronize(); // Синхронизация для завершения вычислений
}

void calc_diffuse(cublasHandle_t handle, float* dev_U, float* dev_U_prev, int Nx, int Ny, float dt, float mu)
{
	float alpha = 1.0f, beta = 1.0f;
	float dt_mu = dt * mu;

	float* dev_d2Udx2, * dev_d2Udy2, * dev_laplacianU;

	cudaMalloc((void**)&dev_d2Udx2, Nx * Ny * sizeof(float));
	cudaMalloc((void**)&dev_d2Udy2, Nx * Ny * sizeof(float));
	cudaMalloc((void**)&dev_laplacianU, Nx * Ny * sizeof(float));
	cudaMemset(dev_d2Udx2, 0.0f, Nx * Ny * sizeof(float));
	cudaMemset(dev_d2Udy2, 0.0f, Nx * Ny * sizeof(float));
	cudaMemset(dev_laplacianU, 0.0f, Nx * Ny * sizeof(float));

	// calc d2U/dx2 = Dxx*U
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Nx, Ny, Nx, &alpha, dev_D_xx, Nx, dev_U_prev, Nx, &beta, dev_d2Udx2, Nx);
	// calc d2U/dy2 = Dyy*U
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Nx, Ny, Nx, &alpha, dev_D_yy, Nx, dev_U_prev, Nx, &beta, dev_d2Udy2, Nx);
	// calc laplacianU = d2U/dx2 + d2U/dy2
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, Nx, Ny, &alpha, dev_d2Udx2, Nx, &beta, dev_d2Udy2, Nx, dev_laplacianU, Nx);
	// calc U = U_prev + (dt*mu)*laplacianU
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, Nx, Ny, &dt_mu, dev_laplacianU, Nx, &beta, dev_U_prev, Nx, dev_U, Nx);

	cudaFree(dev_d2Udx2);
	cudaFree(dev_d2Udy2);
	cudaFree(dev_laplacianU);
}

void calc_advect(cublasHandle_t handle, float* dev_U, int Nx, int Ny, float dt)
{
	cudaMalloc((void**)&dev_dUdx, Nx * Ny * sizeof(float));
	cudaMalloc((void**)&dev_dUdy, Nx * Ny * sizeof(float));
	cudaMalloc((void**)&dev_udUdx, Nx * Ny * sizeof(float));
	cudaMalloc((void**)&dev_vdUdy, Nx * Ny * sizeof(float));
	cudaMalloc((void**)&dev_UnablaU, Nx * Ny * sizeof(float));
	cudaMemset(dev_dUdx, 0.0f, Nx * Ny * sizeof(float));
	cudaMemset(dev_dUdy, 0.0f, Nx * Ny * sizeof(float));
	cudaMemset(dev_udUdx, 0.0f, Nx * Ny * sizeof(float));
	cudaMemset(dev_vdUdy, 0.0f, Nx * Ny * sizeof(float));
	cudaMemset(dev_UnablaU, 0.0f, Nx * Ny * sizeof(float));

	float alpha = 1.0f, beta = 1.0f, gamma = -dt;

	//X:
	// calc dU/dx = Dx*U
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Nx, Ny, Nx, &alpha, dev_D_x, Nx, dev_u, Nx, &beta, dev_dUdx, Nx);
	// calc dU/dy = Dy*U
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Nx, Ny, Nx, &alpha, dev_D_y, Nx, dev_u, Nx, &beta, dev_dUdy, Nx);
	// calc u*(dU/dx)
	launch_elementwise_multiply(dev_u, dev_dUdx, dev_udUdx, Nx, Ny);
	// calc v*(dU/dx)
	launch_elementwise_multiply(dev_v, dev_dUdy, dev_vdUdy, Nx, Ny);
	// calc UnablaU = dev_udUdx + dev_vdUdy
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, Nx, Ny, &alpha, dev_udUdx, Nx, &beta, dev_vdUdy, Nx, dev_UnablaU, Nx);
	// calc U = U_prev - dt * UnablaU
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, Nx, Ny, &gamma, dev_UnablaU, Nx, &beta, dev_u, Nx, dev_u, Nx);

	cudaFree(dev_dUdx);
	cudaFree(dev_dUdy);
	cudaFree(dev_udUdx);
	cudaFree(dev_vdUdy);
	cudaFree(dev_UnablaU);
}

void method_chorin_iteration(int Nx, int Ny, float dt, float mu)
{
	cublasHandle_t handle;
	cublasCreate(&handle);

	//X:
	calc_diffuse(handle, dev_u, dev_u_prev, Nx, Ny, dt, mu);
	calc_advect(handle, dev_u, Nx, Ny, dt);

	//Y:
	calc_diffuse(handle, dev_v, dev_v_prev, Nx, Ny, dt, mu);
	calc_advect(handle, dev_v, Nx, Ny, dt);

	cublasDestroy(handle);
}

void copy_GPU_to_host(float* u, float* v, int Nx, int Ny)
{
	cudaMemcpy(u, dev_u, Nx * Ny * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(v, dev_v, Nx * Ny * sizeof(float), cudaMemcpyDeviceToHost);
}