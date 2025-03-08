#pragma once

void allocate_2d_array(__half*& u, __half*& v, __half*& u_prev, __half*& v_prev, __half*& p, __half*& p_prev, __half*& D_x, __half*& D_y, __half*& D_xx, __half*& D_yy, int Nx, int Ny)
{
    u = new __half[Nx * Ny];
    v = new __half[Nx * Ny];
    u_prev = new __half[Nx * Ny];
    v_prev = new __half[Nx * Ny];
    p = new __half[Nx * Ny];
    p_prev = new __half[Nx * Ny];
    D_x = new __half[Nx * Ny];
    D_y = new __half[Nx * Ny];
    D_xx = new __half[Nx * Ny];
    D_yy = new __half[Nx * Ny];
}

void free_2d_array(__half* u, __half* v, __half* u_prev, __half* v_prev, __half* p, __half*& p_prev, __half*& D_x, __half*& D_y, __half*& D_xx, __half*& D_yy)
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