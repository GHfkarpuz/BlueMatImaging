#include <vector>
#include <iostream>

//catch
#include "catch.hpp"

//flexBox
#include "tools.h"
#include "flexBox.h"
#include "flexLinearOperator.h"
#include "flexGradientOperator.h"


using namespace std;

typedef double floatingType;

int main()
{
    int Nx = 100;
    int Ny = 100;

    std::vector<double> u(Nx * Ny);
    std::vector<floatingType> v(Nx * Ny);

    // random füllen
    for (int i = 0; i < Nx*Ny; ++i)
    {
        u[i] = rand() / (floatingType)RAND_MAX;
        v[i] = rand() / (floatingType)RAND_MAX;
    }

    std::vector<floatingType> grad_u(Nx * Ny, 0.0);
    std::vector<floatingType> gradT_v(Nx * Ny, 0.0);

    // dein Operator
    dxp2d(u, grad_u, EQUALS);                 // oder kombinierter grad!
    dxp2dTransposed(v, gradT_v, EQUALS);

    // Skalarprodukte
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += grad_u[i] * v[i];
        rhs += u[i] * gradT_v[i];
    }

    std::cout << "lhs: " << lhs << std::endl;
    std::cout << "rhs: " << rhs << std::endl;
    std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

    return 0;
}