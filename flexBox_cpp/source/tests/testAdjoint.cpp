#include <vector>
#include <iostream>

//catch
#include "catch.hpp"

//flexBox
#include "tools.h"
#include "flexBox.h"
#include "flexLinearOperator.h"
#include "flexGradientOperator.h"
#include "flexUpwindGradientOperator.h"
#include "flexUpwindDivergenceOperator.h"
#include "flexWeightedGradientOperator.h"
#include "flexWeightedDivergenceOperator.h"
#include "flexOpticalFlowOperator.h"
#include "flexMassPreservationOperator.h"
#include "flexProjectionOperator.h"
#include "flexProjectionOutputOperator.h"


using namespace std;

typedef double floatingType;


bool testGradientXforward(std::vector<floatingType>& x, std::vector<floatingType>& y, int Nx, int Ny)
{
    flexGradientOperator<floatingType> A = flexGradientOperator<floatingType>(std::vector<int>{Nx,Ny}, 0, gradientType::forward, false);

    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.doTimes(false, x, op, EQUALS);
    A.doTimes(true, y, opT, EQUALS);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for forward gradient operator in the direction of x passed."<<std::endl;;
        return true;
    }
    else
    {
        std::cout << "adjoint test for forward gradient operator in the direction of x did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}

bool testGradientYforward(std::vector<floatingType>& x, std::vector<floatingType>& y, int Nx, int Ny)
{
    flexGradientOperator<floatingType> A = flexGradientOperator<floatingType>(std::vector<int>{Nx,Ny}, 1, gradientType::forward, false);

    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.doTimes(false, x, op, EQUALS);
    A.doTimes(true, y, opT, EQUALS);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for forward gradient operator in the direction of y passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for forward gradient operator in the direction of y did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}

bool testGradientXcentral(std::vector<floatingType>& x, std::vector<floatingType>& y, int Nx, int Ny)
{
    flexGradientOperator<floatingType> A = flexGradientOperator<floatingType>(std::vector<int>{Nx,Ny}, 0, gradientType::central, false);

    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.doTimes(false, x, op, EQUALS);
    A.doTimes(true, y, opT, EQUALS);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for central gradient operator in the direction of x passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for central gradient operator in the direction of x did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}

bool testGradientYcentral(std::vector<floatingType>& x, std::vector<floatingType>& y, int Nx, int Ny)
{
    flexGradientOperator<floatingType> A = flexGradientOperator<floatingType>(std::vector<int>{Nx,Ny}, 1, gradientType::central, false);

    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.doTimes(false, x, op, EQUALS);
    A.doTimes(true, y, opT, EQUALS);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for central gradient operator in the direction of y passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for central gradient operator in the direction of y did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}


bool testUpwindGradientX(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& motionField, int Nx, int Ny)
{
    flexUpwindGradientOperator<floatingType> A = flexUpwindGradientOperator<floatingType>(std::vector<int>{Nx,Ny}, 0, motionField, false);

    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.doTimes(false, x, op, EQUALS);
    A.doTimes(true, y, opT, EQUALS);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for upwind gradient operator in the direction of x passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for upwind gradient operator in the direction of x did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}

bool testUpwindGradientY(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& motionField, int Nx, int Ny)
{
    flexUpwindGradientOperator<floatingType> A = flexUpwindGradientOperator<floatingType>(std::vector<int>{Nx,Ny}, 1, motionField, false);

    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.doTimes(false, x, op, EQUALS);
    A.doTimes(true, y, opT, EQUALS);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for upwind gradient operator in the direction of y passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for upwind gradient operator in the direction of y did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}

bool testUpwindDivergenceXforward(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& motionField, int Nx, int Ny)
{
    flexUpwindDivergenceOperator<floatingType> A = flexUpwindDivergenceOperator<floatingType>(motionField, std::vector<int>{Nx,Ny}, 0, gradientType::forward, false);
    
    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.times(false, x, op);
    A.times(true, y, opT);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for upwind divergence operator (forward gradient for weights) in the direction of x passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for upwind divergence operator (forward gradient for weights) in the direction of x did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}

bool testUpwindDivergenceYforward(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& motionField, int Nx, int Ny)
{
    flexUpwindDivergenceOperator<floatingType> A = flexUpwindDivergenceOperator<floatingType>(motionField, std::vector<int>{Nx,Ny}, 0, gradientType::forward, false);

    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.times(false, x, op);
    A.times(true, y, opT);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for upwind divergence operator (forward gradient for weights) in the direction of y passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for upwind divergence operator (forward gradient for weights) in the direction of y did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}
/*
bool testUpwindGradientXsimple(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& motionFieldBig, int Nx1, int Ny1)
{
    int Nx = 5;
    int Ny = 5;

    std::vector<double> motionField(Nx*Ny, 0.0);
    for(int i = 0; i<Nx; ++i)
    {
        for(int j = 0; j<Ny; ++j)
        {
            motionField[i + j*Nx]=motionFieldBig[i + j*Nx];
        }
    }

    for(int i = 0; i<Nx; ++i)
    {
        std::cout << "row" << i << ": ";

        for(int j = 0; j<Ny; ++j)
        {
            std::cout << motionField[i + j*Nx] << " ";
        }
        std::cout << std::endl;
    }


    flexUpwindGradientOperator<double> A(std::vector<int>{Nx,Ny},0,motionField,false);
    
    std::vector<double> e(Nx*Ny,0.0);
    std::vector<double> f(Nx*Ny,0.0);

    for(int i = 0; i<Nx; ++i)
    {
        for(int j = 0; j<Ny; ++j)
        {
            /*
            if(i==j)
            {
                e[i + j*Nx] = 1.0;
            }
            else
            {
                e[i + j*Nx] = 0.0;
            }
            e[i + j*Nx] = 1.0;
        }
    }
    
    std::vector<double> Ae(Nx*Ny,0.0);

    A.doTimes(false,e,Ae,EQUALS);

    for(int i = 0; i<Nx; ++i)
    {
        std::cout << "row" << i << ": ";

        for(int j = 0; j<Ny; ++j)
        {
            std::cout << Ae[j + i*Nx] << " ";
        }
        std::cout << std::endl;
    }


    A.doTimes(true,e,Ae,EQUALS);

    std::cout << "transposed Matrix" << ": " << std::endl;

    for(int i = 0; i<Nx; ++i)
    {
        std::cout << "row" << i << ": ";

        for(int j = 0; j<Ny; ++j)
        {
            std::cout << Ae[j + i*Nx] << " ";
        }
        std::cout << std::endl;
    }


    
    for(int k=0;k<Nx*Ny;k++)
    {
        std::vector<double> e(Nx*Ny,0.0);
        std::vector<double> f(Nx*Ny,0.0);
        e[k] = rand() / (floatingType)RAND_MAX;
        f[k] = -rand() / (floatingType)RAND_MAX;
        e[k] += f[k];

        std::vector<double> Ae(Nx,0.0);

        A.doTimes(false,e,Ae,EQUALS);

        std::cout << "column " << k << " : ";

        for(auto v : Ae)
            std::cout << v << " ";

        std::cout << std::endl;
    }*/
    /*
    std::cout << "Matrix AT:" << std::endl;

    for(int k=0;k<Nx;k++)
    {
        std::vector<double> e(Nx,0.0);
        std::vector<double> f(Nx,0.0);
        e[k] = rand() / (floatingType)RAND_MAX;
        f[k] = -rand() / (floatingType)RAND_MAX;
        e[k] += f[k];

        std::vector<double> ATe(Nx,0.0);

        A.doTimes(true,e,ATe,EQUALS);

        std::cout << "column " << k << " : ";

        for(auto v : ATe)
            std::cout << v << " ";

        std::cout << std::endl;
    }
    
    return true;
}*/

bool testUpwindGradientXsimple(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& motionFieldBig, int Nx1, int Ny1)
{
    int Nx = 5;
    int Ny = 1;

    std::vector<double> motionField(Nx*Ny);

    for(int j=0;j<Ny;++j)
    {
        for(int i=0;i<Nx;++i)
        {
            motionField[i + j*Nx] = motionFieldBig[i + j*Nx];
        }
    }

    flexUpwindGradientOperator<double> A( std::vector<int>{Nx,Ny}, 0, motionField, false);

    std::cout << "Forward matrix A" << std::endl;

    for(int col=0; col<Nx*Ny; ++col)
    {
        std::vector<double> e(Nx*Ny,0.0);
        e[col] = 1.0;

        std::vector<double> Ae(Nx*Ny,0.0);

        A.doTimes(false,e,Ae,EQUALS);

        std::cout << "column " << col << " : ";

        for(auto v : Ae)
        {
            std::cout << v << " ";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Transposed matrix A^T" << std::endl;

    for(int col=0; col<Nx*Ny; ++col)
    {
        std::vector<double> e(Nx*Ny,0.0);
        e[col] = 1.0;

        std::vector<double> ATe(Nx*Ny,0.0);

        A.doTimes(true,e,ATe,EQUALS);

        std::cout << "column " << col << " : ";

        for(auto v : ATe)
        {
            std::cout << v << " ";
        }

        std::cout << std::endl;
    }

    return true;
}


bool testUpwindDivergenceXcentral(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& motionField, int Nx, int Ny)
{
    flexUpwindDivergenceOperator<floatingType> A = flexUpwindDivergenceOperator<floatingType>(motionField, std::vector<int>{Nx,Ny}, 0, gradientType::central, false);
    
    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.times(false, x, op);
    A.times(true, y, opT);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for upwind divergence operator (central gradient for weights) in the direction of x passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for upwind divergence operator (central gradient for weights) in the direction of x did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}

bool testUpwindDivergenceYcentral(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& motionField, int Nx, int Ny)
{
    flexUpwindDivergenceOperator<floatingType> A = flexUpwindDivergenceOperator<floatingType>(motionField, std::vector<int>{Nx,Ny}, 0, gradientType::central, false);

    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.times(false, x, op);
    A.times(true, y, opT);

    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for upwind divergence operator (central gradient for weights) in the direction of y passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for upwind divergence operator (central gradient for weights) in the direction of y did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}










bool testWeightedGradientXforward(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& motionField, int Nx, int Ny)
{
    flexWeightedGradientOperator<floatingType> A = flexWeightedGradientOperator<floatingType>(motionField, std::vector<int>{Nx,Ny}, 0, gradientType::forward, false);

    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.times(false, x, op);
    A.times(true, y, opT);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for weighted gradient operator (forward gradient for weights) in the direction of x passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for weighted gradient operator (forward gradient for weights) in the direction of x did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}

bool testWeightedGradientYforward(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& motionField, int Nx, int Ny)
{
    flexWeightedGradientOperator<floatingType> A = flexWeightedGradientOperator<floatingType>(motionField, std::vector<int>{Nx,Ny}, 1, gradientType::forward, false);

    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.times(false, x, op);
    A.times(true, y, opT);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for weighted gradient operator (forward gradient for weights) in the direction of y passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for weighted gradient operator (forward gradient for weights) in the direction of y did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}

bool testWeightedGradientXcentral(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& motionField, int Nx, int Ny)
{
    flexWeightedGradientOperator<floatingType> A = flexWeightedGradientOperator<floatingType>(motionField, std::vector<int>{Nx,Ny}, 0, gradientType::central, false);

    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.times(false, x, op);
    A.times(true, y, opT);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for weighted gradient operator (central gradient for weights) in the direction of x passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for weighted gradient operator (central gradient for weights) in the direction of x did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}

bool testWeightedGradientYcentral(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& motionField, int Nx, int Ny)
{
    flexWeightedGradientOperator<floatingType> A = flexWeightedGradientOperator<floatingType>(motionField, std::vector<int>{Nx,Ny}, 1, gradientType::central, false);

    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.times(false, x, op);
    A.times(true, y, opT);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for weighted gradient operator (central gradient for weights) in the direction of y passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for weighted gradient operator (central gradient for weights) in the direction of y did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}



bool testWeightedDivergenceXforward(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& motionField, int Nx, int Ny)
{
    flexWeightedDivergenceOperator<floatingType> A = flexWeightedDivergenceOperator<floatingType>(motionField, std::vector<int>{Nx,Ny}, 0, gradientType::forward, false);

    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.times(false, x, op);
    A.times(true, y, opT);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for weighted divergence operator (forward gradient for weights) in the direction of x passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for weighted divergence operator (forward gradient for weights) in the direction of x did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}

bool testWeightedDivergenceYforward(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& motionField, int Nx, int Ny)
{
    flexWeightedDivergenceOperator<floatingType> A = flexWeightedDivergenceOperator<floatingType>(motionField, std::vector<int>{Nx,Ny}, 1, gradientType::forward, false);

    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.times(false, x, op);
    A.times(true, y, opT);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for weighted divergence operator (forward gradient for weights) in the direction of y passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for weighted divergence operator (forward gradient for weights) in the direction of y did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}

bool testWeightedDivergenceXcentral(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& motionField, int Nx, int Ny)
{
    flexWeightedDivergenceOperator<floatingType> A = flexWeightedDivergenceOperator<floatingType>(motionField, std::vector<int>{Nx,Ny}, 0, gradientType::central, false);
    
    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.times(false, x, op);
    A.times(true, y, opT);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for weighted divergence operator (central gradient for weights) in the direction of x passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for weighted divergence operator (central gradient for weights) in the direction of x did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}

bool testWeightedDivergenceYcentral(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& motionField, int Nx, int Ny)
{
    flexWeightedDivergenceOperator<floatingType> A = flexWeightedDivergenceOperator<floatingType>(motionField, std::vector<int>{Nx,Ny}, 1, gradientType::central, false);
    
    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.times(false, x, op);
    A.times(true, y, opT);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for weighted divergence operator (central gradient for weights) in the direction of y passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for weighted divergence operator (central gradient for weights) in the direction of y did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}





bool testOpticalFlowX(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& image, int Nx, int Ny)
{
    flexOpticalFlowOperator<floatingType> A = flexOpticalFlowOperator<floatingType>(image, std::vector<int>{Nx,Ny}, 0, false);

    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.times(false, x, op);
    A.times(true, y, opT);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for optical flow operator in the direction of x passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for optical flow operator in the direction of x did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}

bool testOpticalFlowY(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& image, int Nx, int Ny)
{
    flexOpticalFlowOperator<floatingType> A = flexOpticalFlowOperator<floatingType>(image, std::vector<int>{Nx,Ny}, 1, false);

    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.times(false, x, op);
    A.times(true, y, opT);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for optical flow operator in the direction of y passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for optical flow operator in the direction of y did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}

bool testMassPreservationX(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& image, int Nx, int Ny)
{
    flexMassPreservationOperator<floatingType> A = flexMassPreservationOperator<floatingType>(image, std::vector<int>{Nx,Ny}, 0, false);

    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.times(false, x, op);
    A.times(true, y, opT);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for mass preservation operator in the direction of x passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for mass preservation operator in the direction of x did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}

bool testMassPreservationY(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& image, int Nx, int Ny)
{
    flexMassPreservationOperator<floatingType> A = flexMassPreservationOperator<floatingType>(image, std::vector<int>{Nx,Ny}, 1, false);

    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny, 0.0);
	
    A.times(false, x, op);
    A.times(true, y, opT);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
        rhs += x[i] * opT[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for mass preservation operator in the direction of y passed."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test for mass preservation operator in the direction of y did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}

/*
bool testTimeDerivativeImageForward(std::vector<floatingType>& x, std::vector<floatingType>& y, std::vector<floatingType>& image, std::vector<floatingType>& prevImage, std::vector<floatingType>& nextImage, int Nx, int Ny)
{
    flexTimeDerivativeImageOperator<floatingType> A = flexTimeDerivativeImageOperator<floatingType>(image, prevImage, nextImage, std::vector<int>{Nx,Ny}, gradientType::forward, false);

    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny * 3, 0.0);
	
    std::vector<floatingType> threeImages(Nx * Ny * 3, 0.0);

    for(int i = 0; i<Nx; ++i)
    {
        for(int j = 0; j<Ny; ++j)
        {
            threeImages[i + Nx*j] = prevImage[i];
            threeImages[i + Nx*j + Nx*Ny] = image[i];
            threeImages[i + Nx*j + 2*Nx*Ny] = nextImage[i];
        }
    }

    A.times(false, threeImages, op);
    A.times(true, y, opT);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
    }

    for (int i = 0; i < Nx*Ny*3; ++i)
    {
        rhs += threeImages[i] * opT[i];
    }
        

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for time derivation operator with forward differences pass."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test time derivation operator with forward differences did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}
*/


bool testProjection(flexLinearOperator<floatingType>& projOp, std::vector<floatingType>& x, std::vector<floatingType>& y, int index, int Nx, int Ny)
{
    flexProjectionOperator<floatingType> A = flexProjectionOperator<floatingType>(&projOp, index, std::vector<int>{Nx,Ny,3}, false);

    std::vector<floatingType> op(Nx * Ny, 0.0);
    std::vector<floatingType> opT(Nx * Ny * 3, 0.0);
	

    A.times(false, x, op);
    A.times(true, y, opT);


    // scalar products
    floatingType lhs = 0.0;
    floatingType rhs = 0.0;
    for (int i = 0; i < Nx*Ny*3; ++i)
    {
        rhs += x[i] * opT[i];
    }

    for (int i = 0; i < Nx*Ny; ++i)
    {
        lhs += op[i] * y[i];
    }

    if(std::abs(lhs-rhs)<1e-12)
    {
        std::cout << "adjoint test for projection operator pass."<<std::endl;
        return true;
    }
    else
    {
        std::cout << "adjoint test  projection operator did not pass."<<std::endl;
        std::cout << "lhs: " << lhs << std::endl;
        std::cout << "rhs: " << rhs << std::endl;
        std::cout << "diff: " << std::abs(lhs - rhs) << std::endl;

        return false;
    }
}



TEST_CASE("Adjoint operator tests")
{
    int Nx = 100;
    int Ny = 100;

    std::vector<floatingType> x(Nx * Ny);
    std::vector<floatingType> xLong(Nx * Ny * 3);
    std::vector<floatingType> y(Nx * Ny);
    std::vector<floatingType> image(Nx * Ny);
    std::vector<floatingType> prevImage(Nx * Ny);
    std::vector<floatingType> nextImage(Nx * Ny);
    std::vector<floatingType> motionField(Nx * Ny);

    std::vector<floatingType> xNeg(Nx * Ny);
    std::vector<floatingType> yNeg(Nx * Ny);
    std::vector<floatingType> motionFieldNeg(Nx * Ny);

    flexMassPreservationOperator<floatingType> op = flexMassPreservationOperator<floatingType>(image, {Nx,Ny}, 0, false);

    for (int i = 0; i < Nx * Ny; ++i)
    {
        x[i] = rand() / (floatingType)RAND_MAX;
        y[i] = rand() / (floatingType)RAND_MAX;

        xNeg[i] = -rand() / (floatingType)RAND_MAX;
        yNeg[i] = -rand() / (floatingType)RAND_MAX;

        x[i] += xNeg[i];
        y[i] += yNeg[i];

        image[i] = rand() / (floatingType)RAND_MAX;
        prevImage[i] = rand() / (floatingType)RAND_MAX;
        nextImage[i] = rand() / (floatingType)RAND_MAX;
        
        motionField[i] = rand() / (floatingType)RAND_MAX;
        motionFieldNeg[i] = -rand() / (floatingType)RAND_MAX;

        motionField[i] += motionFieldNeg[i];
    }

    for (int i = 0; i < Nx * Ny * 3; ++i)
    {
        xLong[i] = rand() / (floatingType)RAND_MAX;
    }

    SECTION("Forward gradient x")
    {
        REQUIRE(testGradientXforward(x, y, Nx, Ny));
    }

    SECTION("Central gradient x")
    {
        REQUIRE(testGradientXcentral(x, y, Nx, Ny));
    }

    SECTION("Forward gradient y")
    {
        REQUIRE(testGradientYforward(x, y, Nx, Ny));
    }

    SECTION("Central gradient y")
    {
        REQUIRE(testGradientYcentral(x, y, Nx, Ny));
    }

    SECTION("Upwind gradient x")
    {
        REQUIRE(testUpwindGradientX(x, y, motionField, Nx, Ny));
    }

    SECTION("Upwind gradient y")
    {
        REQUIRE(testUpwindGradientY(x, y, motionField, Nx, Ny));
    }

    SECTION("Upwind divergence x (forward gradient for weights)")
    {
        REQUIRE(testUpwindDivergenceXforward(x, y, motionField, Nx, Ny));
    }

    SECTION("Upwind divergence y (forward gradient for weights)")
    {
        REQUIRE(testUpwindDivergenceYforward(x, y, motionField, Nx, Ny));
    }

    SECTION("Upwind divergence x (central gradient for weights)")
    {
        REQUIRE(testUpwindDivergenceXcentral(x, y, motionField, Nx, Ny));
    }

    SECTION("Upwind divergence y (central gradient for weights)")
    {
        REQUIRE(testUpwindDivergenceYcentral(x, y, motionField, Nx, Ny));
    }

    
    


    SECTION("Weighted gradient x (forward gradient for weights)")
    {
        REQUIRE(testWeightedGradientXforward(x, y, motionField, Nx, Ny));
    }

    SECTION("Weighted gradient y (forward gradient for weights)")
    {
        REQUIRE(testWeightedGradientYforward(x, y, motionField, Nx, Ny));
    }

    SECTION("Weighted gradient x (central gradient for weights)")
    {
        REQUIRE(testWeightedGradientXcentral(x, y, motionField, Nx, Ny));
    }

    SECTION("Weighted gradient y (central gradient for weights)")
    {
        REQUIRE(testWeightedGradientYcentral(x, y, motionField, Nx, Ny));
    }




    SECTION("Weighted divergence x (forward gradient for weights)")
    {
        REQUIRE(testWeightedDivergenceXforward(x, y, motionField, Nx, Ny));
    }

    SECTION("Weighted divergence y (forward gradient for weights)")
    {
        REQUIRE(testWeightedDivergenceYforward(x, y, motionField, Nx, Ny));
    }

    SECTION("Weighted divergence x (central gradient for weights)")
    {
        REQUIRE(testWeightedDivergenceXcentral(x, y, motionField, Nx, Ny));
    }

    SECTION("Weighted divergence y (central gradient for weights)")
    {
        REQUIRE(testWeightedDivergenceYcentral(x, y, motionField, Nx, Ny));
    }




    SECTION("Optical flow operator for x-direction")
    {
        REQUIRE(testOpticalFlowX(x, y, image, Nx, Ny));
    }

    SECTION("Optical flow operator for y-direction")
    {
        REQUIRE(testOpticalFlowX(x, y, image, Nx, Ny));
    }

    SECTION("Mass preservation operator for x-direction")
    {
        REQUIRE(testMassPreservationX(x, y, image, Nx, Ny));
    }

    SECTION("Mass preservation operator for y-direction")
    {
        REQUIRE(testMassPreservationY(x, y, image, Nx, Ny));
    }

    /*
    SECTION("time derivation operator with forward differences")
    {
        REQUIRE(testTimeDerivativeImageForward(xLong, y, image, prevImage, nextImage, Nx, Ny));
    }
    */

    
    SECTION("projection operator")
    {
        REQUIRE(testProjection(op, xLong, y, 1, Nx, Ny));
    }
    

    SECTION("Upwind gradient x simple case")
    {
        REQUIRE(testUpwindGradientXsimple(x, y, motionField, Nx, Ny));
    }
}
