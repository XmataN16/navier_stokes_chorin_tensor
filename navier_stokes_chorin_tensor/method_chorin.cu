#include <cuda_runtime.h>
#include <cublas_v2.h>

__half* dev_u, * dev_v, * dev_u_prev, * dev_v_prev, * dev_p, * dev_p_prev, * dev_D_x, * dev_D_y, * dev_D_xx, * dev_D_yy;

__constant__ float dev_dx, dev_dy, dev_dt, dev_mu;

void allocate_2d_array_on_GPU(__half* u, __half* v, __half* u_prev, __half* v_prev, __half* p, __half* p_prev, __half*& D_x, __half*& D_y, __half*& D_xx, __half*& D_yy, int Nx, int Ny, __half dx, __half dy, __half dt, __half mu)
{
	//выделение памяти на device
	cudaMalloc((void**)&dev_u, Nx * Ny * sizeof(__half));
	cudaMalloc((void**)&dev_v, Nx * Ny * sizeof(__half));
	cudaMalloc((void**)&dev_u_prev, Nx * Ny * sizeof(__half));
	cudaMalloc((void**)&dev_v_prev, Nx * Ny * sizeof(__half));
	cudaMalloc((void**)&dev_p, Nx * Ny * sizeof(__half));
	cudaMalloc((void**)&dev_p_prev, Nx * Ny * sizeof(__half));

	cudaMalloc((void**)&dev_D_x, Nx * Ny * sizeof(__half));
	cudaMalloc((void**)&dev_D_y, Nx * Ny * sizeof(__half));
	cudaMalloc((void**)&dev_D_xx, Nx * Ny * sizeof(__half));
	cudaMalloc((void**)&dev_D_yy, Nx * Ny * sizeof(__half));

	//копирование массивов из ОЗУ в память device
	cudaMemcpy(dev_u, &u[0], Nx * Ny * sizeof(__half), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v, &v[0], Nx * Ny * sizeof(__half), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_u_prev, &u_prev[0], Nx * Ny * sizeof(__half), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v_prev, &v_prev[0], Nx * Ny * sizeof(__half), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_p, &p[0], Nx * Ny * sizeof(__half), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_p_prev, &p_prev[0], Nx * Ny * sizeof(__half), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_D_x, &D_x[0], Nx * Ny * sizeof(__half), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_D_y, &D_y[0], Nx * Ny * sizeof(__half), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_D_xx, &D_xx[0], Nx * Ny * sizeof(__half), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_D_yy, &D_yy[0], Nx * Ny * sizeof(__half), cudaMemcpyHostToDevice);
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

void calc_advect(__half* u, __half* v, __half* u_prev, __half* v_prev, __half*& D_x, __half*& D_y, int Nx, int Ny)
{
	cublasHandle_t handle;
	cublasCreate(&handle);

	float alpha = 1.0f, beta = 0.0f;

	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, Nx, Ny, Nx, &alpha, dev_D_x, CUDA_R_16F, Nx, dev_u_prev, CUDA_R_16F, Nx, &beta, dev_u, CUDA_R_16F, Nx, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);

}


void copy_GPU_to_host(__half* u, __half* v, int Nx, int Ny)
{
	cudaMemcpy(u, dev_u, Nx * Ny * sizeof(__half), cudaMemcpyDeviceToHost);
	cudaMemcpy(v, dev_v, Nx * Ny * sizeof(__half), cudaMemcpyDeviceToHost);
}