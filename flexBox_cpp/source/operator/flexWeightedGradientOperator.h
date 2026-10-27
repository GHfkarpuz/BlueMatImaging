#ifndef flexWeightedGradientOperator_H
#define flexWeightedGradientOperator_H

#include <vector>
#include <algorithm>
#include "flexLinearOperator.h"
#include "flexGradientOperator.h"
#include "flexParallel.h"

/*
    Operator: K(u) = w * (D u) where D = gradient in one direction and w = fixed weight / motion field
    Adjoint:  K^T(y) = D^T (w * y)
*/
template<typename T>
class flexWeightedGradientOperator : public flexLinearOperator<T>
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
    gradientType type;
    flexGradientOperator<T>* gradOp;
    Tdata tmp;

public:

    flexWeightedGradientOperator(const std::vector<T>& AWeights, const std::vector<int>& AInputDimension, int ADirection, gradientType aType, bool aIsMinus)
    : flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension), weightedGradientOp, aIsMinus),
      weightsHost(AWeights), inputDimension(AInputDimension), direction(ADirection), type(aType)
    {
        N = vectorProduct(inputDimension);
        flexAssign(weights, weightsHost);
        gradOp = new flexGradientOperator<T>(inputDimension, direction, this->type, false);
        tmp.resize(N, (T)0);
    }

    ~flexWeightedGradientOperator()
    {
        delete gradOp;
    }

    flexWeightedGradientOperator<T>* copy()
    {
        return new flexWeightedGradientOperator<T>(weightsHost, inputDimension, direction, this->type, this->isMinus);
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
            gradOp->doTimes(false, input, tmp, EQUALS);
            flexMul(output, weights, tmp, s);
        }
        else
        {
            flexMul(tmp, weights, input, EQUALS);
            gradOp->doTimes(true, tmp, output, s);
        }
    }

    void absRowSum(bool transposed, Tdata& result)
    {
        Tdata gradRowSum;
        gradOp->absRowSum(transposed, gradRowSum);
        result.resize(N);
        flexAbsMulFunctor<T> f = { flexPtr(result), flexPtr(weights), flexPtr(gradRowSum) };
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
