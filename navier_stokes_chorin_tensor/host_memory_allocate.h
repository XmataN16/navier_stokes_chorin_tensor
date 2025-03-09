#pragma once

void allocate_2d_array(float*& u, float*& v, float*& u_prev, float*& v_prev, float*& p, float*& p_prev, float*& D_x, float*& D_y, float*& D_xx, float*& D_yy, int Nx, int Ny)
{
    u = new float[Nx * Ny];
    v = new float[Nx * Ny];
    u_prev = new float[Nx * Ny];
    v_prev = new float[Nx * Ny];
    p = new float[Nx * Ny];
    p_prev = new float[Nx * Ny];
    D_x = new float[Nx * Ny];
    D_y = new float[Nx * Ny];
    D_xx = new float[Nx * Ny];
    D_yy = new float[Nx * Ny];
}

void free_2d_array(float* u, float* v, float* u_prev, float* v_prev, float* p, float*& p_prev, float*& D_x, float*& D_y, float*& D_xx, float*& D_yy)
{
    delete[] u;
    delete[] v;
    delete[] u_prev;
    delete[] v_prev;
    delete[] p;
    delete[] p_prev;
    delete[] D_x;
    delete[] D_y;
    delete[] D_xx;
    delete[] D_yy;
}