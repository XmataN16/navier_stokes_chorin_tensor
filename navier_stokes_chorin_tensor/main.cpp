#include <iostream>
#include <iomanip>
#include <cuda_fp16.h>
#include <host_memory_allocate.h>
#include <boundary_initial_func.h>
#include <derivative_matrix.h>


void allocate_2d_array_on_GPU(float* u, float* v, float* u_prev, float* v_prev, float* p, float* p_prev, float*& D_x, float*& D_y, float*& D_xx, float*& D_yy, int Nx, int Ny, float dx, float dy, float dt, float mu);
void free_2d_array_on_GPU();
void copy_GPU_to_host(float* u, float* v, int Nx, int Ny);
void method_chorin_iteration(int Nx, int Ny, float dt, float mu);

// Set global parameters
const float Lx = 0.1f;      // длина расчетной области вдоль оси x в м
const float Ly = 0.1f;      // длина расчетной области вдоль оси y в м
const float T = 0.001f;   // время расчета в сек.
const int Nx = 10;          // количество узлов по оси x
const int Ny = 10;          // количество узлов по оси y
const int Nt = 10;          // количество узлов по временной оси
const float rho = 1000.0f;  // плотность в кг/м^2
const float mu = 1.0f;      // [Pa*s] // динамическая вязкость
const float dx = Lx / (Nx - 1);
const float dy = Ly / (Ny - 1);
const float dt = T / (Nt - 1); // временной шаг в сек.
const float nu = mu / rho;     // кинематическая вязкость

// Функция для вывода матрицы в консоль
void print_matrix(const float* M, int Nx, int Ny)
{
    std::cout << std::fixed << std::setprecision(3);

    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            std::cout << std::setw(8) << M[(i * Nx) + j] << " ";
        }
        std::cout << std::endl;
    }
}


int main()
{
    float* u, * v, * u_prev, * v_prev, * p, * p_prev, * D_x, * D_y, * D_xx, * D_yy;

    // Allocate memory(HOST)
    allocate_2d_array(u, v, u_prev, v_prev, p, p_prev, D_x, D_y, D_xx, D_yy, Nx, Ny);

    // Set start and boundary conditions
    set_initial_conditions(u, v, p, Nx, Ny);
    set_boundary_conditions(u, v, Nx, Ny);
    set_pressure_boundary_conditions(p, Nx, Ny);

    // Init matrix diff operators
    createFirstDerivativeMatrix(D_x, Nx, dx);
    createFirstDerivativeMatrix(D_y, Ny, dy);
    createSecondDerivativeMatrix(D_xx, Nx, dx);
    createSecondDerivativeMatrix(D_yy, Ny, dy);

    // Allocate memory(DEVICE)
    allocate_2d_array_on_GPU(u, v, u_prev, v_prev, p, p_prev, D_x, D_y, D_xx, D_yy, Nx, Ny, dx, dy, dt, mu);

    method_chorin_iteration(Nx, Ny, dt, mu);
    //calc_advect(Nx, Ny, dt);

    copy_GPU_to_host(u, v, Nx, Ny);

    free_2d_array_on_GPU();

    print_matrix(u, Nx, Ny);

    return 0;
}