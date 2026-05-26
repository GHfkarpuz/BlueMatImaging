#ifndef flexWeightedDivergenceOperator_H
#define flexWeightedDivergenceOperator_H

#include <vector>
#include "flexLinearOperator.h"
#include "flexGradientOperator.h"

/*
    Operator: K(u) = D * (u w) where (D *) = divergence in one  w = fixed weight / motion field
*/

template<typename T>
class flexWeightedDivergenceOperator : public flexLinearOperator<T>
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
    Tdata gradWeights;

    gradientType type; //Possible values are forward, backward and central.


    flexGradientOperator<T>* gradOp;

    mutable Tdata tmp;

public:

    flexWeightedDivergenceOperator(const Tdata& AWeights,const std::vector<int>& AInputDimension,int ADirection, gradientType aType, bool aIsMinus):weights(AWeights),inputDimension(AInputDimension),direction(ADirection),type(aType),flexLinearOperator<T>(vectorProduct(AInputDimension),vectorProduct(AInputDimension),weightedDivergenceOp,aIsMinus)
    {
        N = vectorProduct(inputDimension);

        gradOp = new flexGradientOperator<T>(inputDimension,direction,this->type,false);

        
        gradWeights.resize(N, (T)0);
        gradOp->doTimes(false, this->weights, this->gradWeights, EQUALS);
        
        tmp.resize(N, (T)0);
    }

    ~flexWeightedDivergenceOperator()
    {
        delete gradOp;
    }

    flexWeightedDivergenceOperator<T>* copy()
    {
        return new flexWeightedDivergenceOperator<T>(weights,inputDimension,direction,this->type,this->isMinus);
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
            result[k] = std::abs(weights[k]) * gradRowSum[k] + std::abs(gradWeights[k]);
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
                switch(s)
                {
                    case PLUS:
                        output[k] += weights[k] * tmp[k];;
                        break;

                    case MINUS:
                        output[k] -= weights[k] * tmp[k];;
                        break;

                    case EQUALS:
                        output[k] = weights[k] * tmp[k];;
                        break;
                }
            }

            for(int k = 0; k<N; ++k)
            {
                switch(s)
                {
                    case PLUS:
                        output[k] += gradWeights[k] * input[k];
                        break;

                    case MINUS:
                        output[k] -= gradWeights[k] * input[k];
                        break;

                    case EQUALS:
                        output[k] += gradWeights[k] * input[k];
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

            gradOp->doTimes(true, tmp, output, s);

            for(int k = 0; k<N; ++k)
            {
                switch(s)
                {
                    case PLUS:
                        output[k] += gradWeights[k] * input[k];
                        break;

                    case MINUS:
                        output[k] -= gradWeights[k] * input[k];
                        break;

                    case EQUALS:
                        output[k] += gradWeights[k] * input[k];
                        break;
                }
            }
        }
    }
};

#endif