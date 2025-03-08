#include <iostream>
#include <iomanip>
#include <cuda_fp16.h>
#include <host_memory_allocate.h>
#include <boundary_initial_func.h>
#include <derivative_matrix.h>


void allocate_2d_array_on_GPU(__half* u, __half* v, __half* u_prev, __half* v_prev, __half* p, __half* p_prev, __half*& D_x, __half*& D_y, __half*& D_xx, __half*& D_yy, int Nx, int Ny, __half dx, __half dy, __half dt, __half mu);
void copy_GPU_to_host(__half* u, __half* v, int Nx, int Ny);
void calc_advect(__half* u, __half* v, __half* u_prev, __half* v_prev, __half*& D_x, __half*& D_y, int Nx, int Ny);

// ”становка глобальных параметров
const float Lx = 0.1f;      // длина расчетной области вдоль оси x в м
const float Ly = 0.1f;      // длина расчетной области вдоль оси y в м
const float T = 0.00001f;   // врем€ расчета в сек.
const int Nx = 10;          // количество узлов по оси x
const int Ny = 10;          // количество узлов по оси y
const int Nt = 10;          // количество узлов по временной оси
const float rho = 1000.0f;  // плотность в кг/м^2
const float mu = 1.0f;      // [Pa*s] // динамическа€ в€зкость
const __half dx = Lx / (Nx - 1);
const __half dy = Ly / (Ny - 1);
const __half dt = T / (Nt - 1); // временной шаг в сек.
const __half nu = mu / rho;     // кинематическа€ в€зкость

// ‘ункци€ дл€ вывода матрицы в консоль
void print_matrix(const __half* M, int Nx, int Ny)
{
    std::cout << std::fixed << std::setprecision(3);

    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            std::cout << std::setw(8) << __half2float(M[(i * Nx) + j]) << " ";
        }
        std::cout << std::endl;
    }
}


int main()
{
    __half* u, * v, * u_prev, * v_prev, * p, * p_prev, * D_x, * D_y, * D_xx, * D_yy;

    // ¬ыдел€ем пам€ть дл€ операторных матриц
    allocate_2d_array(u, v, u_prev, v_prev, p, p_prev, D_x, D_y, D_xx, D_yy, Nx, Ny);

    //начальные и граничные услови€
    set_initial_conditions(u, v, p, Nx, Ny);
    set_boundary_conditions(u, v, Nx, Ny);
    set_pressure_boundary_conditions(p, Nx, Ny);

    // »нициализируем разностные операторы
    createFirstDerivativeMatrix(D_x, Nx, dx);
    createFirstDerivativeMatrix(D_y, Ny, dy);
    createSecondDerivativeMatrix(D_xx, Nx, dx);
    createSecondDerivativeMatrix(D_yy, Ny, dy);

    //вызов функции дл€ выделени€ пам€ти под массивы на device
    allocate_2d_array_on_GPU(u, v, u_prev, v_prev, p, p_prev, D_x, D_y, D_xx, D_yy, Nx, Ny, dx, dy, dt, mu);

    calc_advect(u, v, u_prev, v_prev, D_x, D_y, Nx, Ny);

    return 0;
}