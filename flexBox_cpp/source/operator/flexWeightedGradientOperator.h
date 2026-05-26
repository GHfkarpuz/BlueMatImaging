#ifndef flexWeightedGradientOperator_H
#define flexWeightedGradientOperator_H

#include <vector>
#include "flexLinearOperator.h"
#include "flexGradientOperator.h"

/*
    Operator: K(u) = w * (D u) where D = gradient in one  w = fixed weight / motion field
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

    const Tdata weights;

    std::vector<int> inputDimension;

    int direction;
    int N;

    gradientType type; //Possible values are forward, backward and central.


    flexGradientOperator<T>* gradOp;

    mutable Tdata tmp;

public:

    flexWeightedGradientOperator(const Tdata& AWeights,const std::vector<int>& AInputDimension,int ADirection, gradientType aType, bool aIsMinus):weights(AWeights),inputDimension(AInputDimension),direction(ADirection),type(aType),flexLinearOperator<T>(vectorProduct(AInputDimension),vectorProduct(AInputDimension),weightedGradientOp,aIsMinus)
    {
        N = vectorProduct(inputDimension);

        gradOp = new flexGradientOperator<T>(inputDimension,direction,this->type,false);

        tmp.resize(N, (T)0);
    }

    ~flexWeightedGradientOperator()
    {
        delete gradOp;
    }

    flexWeightedGradientOperator<T>* copy()
    {
        return new flexWeightedGradientOperator<T>(weights,inputDimension,direction,this->type,this->isMinus);
    }

    void updateWeights(const Tdata& newWeights)
    {
        weights = newWeights;
    }

    void times(bool transposed, const Tdata& input, Tdata& output)
    {
        this->doTimes(transposed, input, output, EQUALS);
    }

    void timesPlus(bool transposed, const Tdata& input, Tdata& output)
    {
        if(this->isMinus)
        {
            this->doTimes(transposed, input, output, MINUS);
        }
        else
        {
            this->doTimes(transposed, input, output, PLUS);
        }
    }

    void timesMinus(bool transposed, const Tdata& input, Tdata& output)
    {
        if(this->isMinus)
        {
            this->doTimes(transposed, input, output, PLUS);
        }
        else
        {
            this->doTimes(transposed, input, output, MINUS);
        }
    }

    std::vector<T> getAbsRowSum(bool transposed) override
    {
        std::vector<T> result(N, (T)0);

        std::vector<T> gradRowSum = gradOp->getAbsRowSum(transposed);

        #pragma omp parallel for
        for(int k=0; k<N; ++k)
        {
            result[k] = std::abs(weights[k]) * gradRowSum[k];
        }

        return result;
    }

    T getMaxRowSumAbs(bool transposed) override
    {
        std::vector<T> rowSum = getAbsRowSum(transposed);

        T maxVal = (T)0;

        #pragma omp parallel for reduction(max:maxVal)
        for(int k=0; k<N; ++k)
        {
            if(rowSum[k] > maxVal)
            {
                maxVal = rowSum[k];
            }
        }

        return maxVal;
    }

private:

    void doTimes(bool transposed,const Tdata& input,Tdata& output,const mySign s)
    {
        assert(input.size() == N);
        assert(output.size() == N);
        assert(tmp.size() == N);
        assert(weights.size() == N);
        if(!transposed)
        {
            gradOp->doTimes(false,input,tmp,EQUALS);

            #pragma omp parallel for
            for(int k=0; k<N; ++k)
            {
                T val = weights[k] * tmp[k];

                switch(s)
                {
                    case PLUS:
                        output[k] += val;
                        break;

                    case MINUS:
                        output[k] -= val;
                        break;

                    case EQUALS:
                        output[k] = val;
                        break;
                }
            }
        }
        else
        {
            #pragma omp parallel for
            for(int k=0; k<N; ++k)
            {
                tmp[k] = weights[k] * input[k];
            }

            gradOp->doTimes(true,tmp,output,s);
        }
    }
};

#endif