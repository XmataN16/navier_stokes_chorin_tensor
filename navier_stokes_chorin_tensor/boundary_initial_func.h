#pragma once
void set_initial_conditions(float* u, float* v, float* p, int Nx, int Ny)
{
	for (int i = 0; i < Nx; i++)
	{
		for (int j = 0; j < Ny; j++)
		{
			int index = i * Ny + j;
			u[index] = 0.0f;
			v[index] = 0.0f;
			p[index] = 0.0f;
		}
	}
}

void set_boundary_conditions(float* u, float* v, int Nx, int Ny)
{
	// установка скоростей на границах
	for (int i = 0; i < Nx; i++)
	{
		u[i * Ny + 0] = 0.01f; // продольная скорость на верхней границе слева
		u[i * Ny + (Ny - 1)] = 0.01f; // продольная скорость на нижней границе слева
		v[i * Ny + 0] = 0.0f;
		v[i * Ny + (Ny - 1)] = 0.0f;
	}
	// нулевые скорости на верхней и нижней границах справа
	for (int j = 0; j < Ny; j++)
	{
		u[(Nx - 1) * Ny + j] = 0.0f;
		v[(Nx - 1) * Ny + j] = 0.0f;
	}

	// препятствие внутри трубы
	for (int i = 1; i < Nx - 1; i++)
	{
		for (int j = 1; j < Ny - 1; j++)
		{
			if ((i - 10) * (i - 10) + (j - 10) * (j - 10) <= 3 * 3)
			{
				u[i * Ny + j] = 0.0f;
				v[i * Ny + j] = 0.0f;
			}
		}
	}

	u[0 * Ny + 0] = 0.0f; // продольная скорость на верхнем левом углу
	u[0 * Ny + (Ny - 1)] = 0.0f; // продольная скорость на нижнем левом углу
	u[(Nx - 1) * Ny + 0] = 0.0f;
	u[(Nx - 1) * Ny + (Ny - 1)] = 0.0f;
}

void set_pressure_boundary_conditions(float* p, int Nx, int Ny)
{
	// установка давлений на границах
	for (int i = 0; i < Nx; i++)
	{
		int index_top = i * Ny;
		int index_bottom = i * Ny + Ny - 1;
		p[index_top] = p[index_top + 1];
		p[index_bottom] = p[index_bottom - 1];
	}
	for (int j = 0; j < Ny; j++)
	{
		int index_left = j;
		int index_right = (Nx - 1) * Ny + j;
		p[index_left] = p[index_left + Ny];
		p[index_right] = p[index_right - Ny];
	}
	// интерполяция давления на углах с отверстием
	p[0] = (p[Ny] + p[1]) / 2.0f;
	p[Ny - 1] = (p[Ny - 2] + p[2 * Ny - 1]) / 2.0f;
	p[(Nx - 1) * Ny] = (p[(Nx - 1) * Ny + 1] + p[(Nx - 2) * Ny]) / 2.0f;
	p[(Nx - 1) * Ny + Ny - 1] = (p[(Nx - 1) * Ny + Ny - 2] + p[(Nx - 2) * Ny + Ny - 1]) / 2.0f;
}