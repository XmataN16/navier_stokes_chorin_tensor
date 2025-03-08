#pragma once

// Функция для создания разностной матрицы первого порядка
void createFirstDerivativeMatrix(__half* D, int size, __half h)
{
    memset(D, 0, size * size * sizeof(__half));
    for (int i = 0; i < size; i++)
    {
        D[i * size + (i - 1)] = __float2half(- 1.0f) / (__float2half(2.0f) * h);
        D[i * size + (i + 1)] = __float2half(1.0f) / (__float2half(2.0f) * h);
    }
}

// Функция для создания разностной матрицы второго порядка (Лапласиан)
void createSecondDerivativeMatrix(__half* D, int size, __half h)
{
    memset(D, 0, size * size * sizeof(__half));
    for (int i = 0; i < size; i++)
    {
        D[i * size + i] = __float2half(-2.0f) / (h * h);
        D[i * size + (i - 1)] = __float2half(1.0f) / (h * h);
        D[i * size + (i + 1)] = __float2half(1.0f) / (h * h);
    }
}
