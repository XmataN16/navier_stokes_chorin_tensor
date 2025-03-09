#pragma once

void createFirstDerivativeMatrix(float* D_x, int Nx, float h)
{
    memset(D_x, 0, Nx * Nx * sizeof(float));

    float coeff = 1.0f / (2.0f * h);

    // ��������� ������ �� ��������� �������
    for (int i = 0; i < Nx; ++i) 
    {
        for (int j = 0; j < Nx; ++j) 
        {
            int index = i * Nx + j; // ������ � ���������� �������

            if (i == j + 1 and j != 0 and j != Nx - 1) 
            {
                D_x[index] = coeff;  // ���� ������� ���������
            }
            else if (i == j - 1 and j != 0 and j != Nx - 1)
            {
                D_x[index] = -coeff;   // ���� ������� ���������
            }
        }
    }
}

// ������� ��� �������� ���������� ������� ������� ������� (���������)
void createSecondDerivativeMatrix(float* D, int Nx, float h)
{
    memset(D, 0.0f, Nx * Nx * sizeof(float));

    float coeff = 1.0f / (h * h);
    float coeff2 = -2.0f / (h * h);

    // ��������� ������ �� ��������� �������
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Nx; ++j)
        {
            int index = i * Nx + j; // ������ � ���������� �������

            if ((i == j + 1 or i == j - 1) and j != 0 and j != Nx - 1)
            {
                D[index] = coeff;  // ���� ��� ���� ������� ���������
            }
            else if (i == j and j != 0 and j != Nx - 1)
            {
                D[index] = coeff2;   // �� ������� ���������
            }
        }
    }
}

