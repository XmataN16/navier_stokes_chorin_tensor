#pragma once

void createFirstDerivativeMatrix(float* D_x, int Nx, float h)
{
    memset(D_x, 0, Nx * Nx * sizeof(float));

    float coeff = 1.0f / (2.0f * h);

    // Заполняем массив по заданному шаблону
    for (int i = 0; i < Nx; ++i) 
    {
        for (int j = 0; j < Nx; ++j) 
        {
            int index = i * Nx + j; // индекс в одномерном массиве

            if (i == j + 1 and j != 0 and j != Nx - 1) 
            {
                D_x[index] = coeff;  // ниже главной диагонали
            }
            else if (i == j - 1 and j != 0 and j != Nx - 1)
            {
                D_x[index] = -coeff;   // выше главной диагонали
            }
        }
    }
}

// Функция для создания разностной матрицы второго порядка (Лапласиан)
void createSecondDerivativeMatrix(float* D, int Nx, float h)
{
    memset(D, 0.0f, Nx * Nx * sizeof(float));

    float coeff = 1.0f / (h * h);
    float coeff2 = -2.0f / (h * h);

    // Заполняем массив по заданному шаблону
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Nx; ++j)
        {
            int index = i * Nx + j; // индекс в одномерном массиве

            if ((i == j + 1 or i == j - 1) and j != 0 and j != Nx - 1)
            {
                D[index] = coeff;  // выше или ниже главной диагонали
            }
            else if (i == j and j != 0 and j != Nx - 1)
            {
                D[index] = coeff2;   // на главной диагонали
            }
        }
    }
}

