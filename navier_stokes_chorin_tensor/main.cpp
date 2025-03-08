#include <iostream>
#include <iomanip>
#include <cuda_fp16.h>
#include <host_memory_allocate.h>
#include <boundary_initial_func.h>
#include <derivative_matrix.h>


void allocate_2d_array_on_GPU(__half* u, __half* v, __half* u_prev, __half* v_prev, __half* p, __half* p_prev, __half*& D_x, __half*& D_y, __half*& D_xx, __half*& D_yy, int Nx, int Ny, __half dx, __half dy, __half dt, __half mu);
void copy_GPU_to_host(__half* u, __half* v, int Nx, int Ny);
void calc_advect(__half* u, __half* v, __half* u_prev, __half* v_prev, __half*& D_x, __half*& D_y, int Nx, int Ny);

// ��������� ���������� ����������
const float Lx = 0.1f;      // ����� ��������� ������� ����� ��� x � �
const float Ly = 0.1f;      // ����� ��������� ������� ����� ��� y � �
const float T = 0.00001f;   // ����� ������� � ���.
const int Nx = 10;          // ���������� ����� �� ��� x
const int Ny = 10;          // ���������� ����� �� ��� y
const int Nt = 10;          // ���������� ����� �� ��������� ���
const float rho = 1000.0f;  // ��������� � ��/�^2
const float mu = 1.0f;      // [Pa*s] // ������������ ��������
const __half dx = Lx / (Nx - 1);
const __half dy = Ly / (Ny - 1);
const __half dt = T / (Nt - 1); // ��������� ��� � ���.
const __half nu = mu / rho;     // �������������� ��������

// ������� ��� ������ ������� � �������
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

    // �������� ������ ��� ����������� ������
    allocate_2d_array(u, v, u_prev, v_prev, p, p_prev, D_x, D_y, D_xx, D_yy, Nx, Ny);

    //��������� � ��������� �������
    set_initial_conditions(u, v, p, Nx, Ny);
    set_boundary_conditions(u, v, Nx, Ny);
    set_pressure_boundary_conditions(p, Nx, Ny);

    // �������������� ���������� ���������
    createFirstDerivativeMatrix(D_x, Nx, dx);
    createFirstDerivativeMatrix(D_y, Ny, dy);
    createSecondDerivativeMatrix(D_xx, Nx, dx);
    createSecondDerivativeMatrix(D_yy, Ny, dy);

    //����� ������� ��� ��������� ������ ��� ������� �� device
    allocate_2d_array_on_GPU(u, v, u_prev, v_prev, p, p_prev, D_x, D_y, D_xx, D_yy, Nx, Ny, dx, dy, dt, mu);

    calc_advect(u, v, u_prev, v_prev, D_x, D_y, Nx, Ny);

    return 0;
}