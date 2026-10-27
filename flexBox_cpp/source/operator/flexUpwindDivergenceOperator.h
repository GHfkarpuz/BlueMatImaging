#ifndef flexUpwindDivergenceOperator_H
#define flexUpwindDivergenceOperator_H

#include <vector>
#include <algorithm>
#include "flexLinearOperator.h"
#include "flexUpwindGradientOperator.h"
#include "flexParallel.h"

/*
    Operator: K(u) = U_w (w u) where U_w = upwind gradient with the motion field w in one direction
    Adjoint:  K^T(y) = w * (U_w^T y)
*/
template<typename T>
class flexUpwindDivergenceOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
    typedef thrust::device_vector<T> Tdata;
#else
    typedef std::vector<T> Tdata;
#endif

private:
    std::vector<T> weightsHost;
    Tdata weights;
    std::vector<int> inputDimension;
    int direction;
    int N;
    gradientType type; //kept for the interface, the upwind direction is given by the sign of w
    flexUpwindGradientOperator<T>* upwindGradOp;
    Tdata tmp;

public:

    flexUpwindDivergenceOperator(const std::vector<T>& AWeights, const std::vector<int>& AInputDimension, int ADirection, gradientType aType, bool aIsMinus)
    : flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension), upwindDivergenceOp, aIsMinus),
      weightsHost(AWeights), inputDimension(AInputDimension), direction(ADirection), type(aType)
    {
        N = vectorProduct(inputDimension);
        flexAssign(weights, weightsHost);
        upwindGradOp = new flexUpwindGradientOperator<T>(inputDimension, direction, weightsHost, false);
        tmp.resize(N, (T)0);
    }

    ~flexUpwindDivergenceOperator()
    {
        delete upwindGradOp;
    }

    flexUpwindDivergenceOperator<T>* copy()
    {
        return new flexUpwindDivergenceOperator<T>(weightsHost, inputDimension, direction, this->type, this->isMinus);
    }

    void times(bool transposed, const Tdata& input, Tdata& output)
    {
        this->doTimes(transposed, input, output, EQUALS);
    }

    void timesPlus(bool transposed, const Tdata& input, Tdata& output)
    {
        this->doTimes(transposed, input, output, this->isMinus ? MINUS : PLUS);
    }

    void timesMinus(bool transposed, const Tdata& input, Tdata& output)
    {
        this->doTimes(transposed, input, output, this->isMinus ? PLUS : MINUS);
    }

    void doTimes(bool transposed, const Tdata& input, Tdata& output, const mySign s)
    {
        if (!transposed)
        {
            flexMul(tmp, weights, input, EQUALS);
            upwindGradOp->doTimes(false, tmp, output, s);
        }
        else
        {
            upwindGradOp->doTimes(true, input, tmp, EQUALS);
            flexMul(output, weights, tmp, s);
        }
    }

    //! as in the original implementation: max|w| times the row sums of the upwind gradient
    void absRowSum(bool transposed, Tdata& result)
    {
        std::vector<T> w = weightsHost;
        T vmax = (T)0;
        for (int k = 0; k < N; ++k)
            vmax = std::max(vmax, std::abs(w[k]));

        Tdata gradRows;
        upwindGradOp->absRowSum(transposed, gradRows);
        result.resize(N);
        flexScaleFunctor<T> f = { flexPtr(result), flexPtr(gradRows), vmax };
        flexForEach(N, f);
    }

    std::vector<T> getAbsRowSum(bool transposed)
    {
        Tdata result;
        this->absRowSum(transposed, result);
        return flexToHost(result);
    }

    T getMaxRowSumAbs(bool transposed)
    {
        std::vector<T> rowSum = getAbsRowSum(transposed);
        return rowSum.empty() ? T(0) : *std::max_element(rowSum.begin(), rowSum.end());
    }

    #ifdef __CUDACC__
    thrust::device_vector<T> getAbsRowSumCUDA(bool transposed)
    {
        thrust::device_vector<T> result;
        this->absRowSum(transposed, result);
        return result;
    }
    #endif
};

#endif
