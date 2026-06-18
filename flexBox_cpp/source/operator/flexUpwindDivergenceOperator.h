#ifndef flexUpwindDivergenceOperator_H
#define flexUpwindDivergenceOperator_H

#include <vector>
#include "flexLinearOperator.h"
#include "flexGradientOperator.h"

/*
    Operator: K(u) = D * (u w) where (D *) = divergence in one  w = fixed weight / motion field
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

    const Tdata weights;

    std::vector<int> inputDimension;

    int direction;
    int N;
    Tdata gradWeights;
    bool isMinus;

    gradientType type; //Possible values are forward, backward and central.

    flexUpwindGradientOperator<T>* upwindGradOp;
    flexGradientOperator<T>* gradOp;

    mutable Tdata tmp;

public:

    flexUpwindDivergenceOperator(const Tdata& AWeights,const std::vector<int>& AInputDimension,int ADirection, gradientType aType, bool aIsMinus):weights(AWeights),inputDimension(AInputDimension),direction(ADirection),type(aType),isMinus(aIsMinus),flexLinearOperator<T>(vectorProduct(AInputDimension),vectorProduct(AInputDimension),upwindDivergenceOp,aIsMinus)
    {
        N = vectorProduct(inputDimension);

        upwindGradOp = new flexUpwindGradientOperator<T>(inputDimension,direction,this->weights,false);
        gradOp = new flexGradientOperator<T>(inputDimension,direction,this->type,false);

        
        gradWeights.resize(N, (T)0);
        gradOp->doTimes(false, this->weights, this->gradWeights, EQUALS);
        
        tmp.resize(N, (T)0);
    }

    ~flexUpwindDivergenceOperator()
    {
        delete gradOp;
        delete upwindGradOp;
    }

    flexUpwindDivergenceOperator<T>* copy()
    {
        return new flexUpwindDivergenceOperator<T>(weights,inputDimension,direction,this->type,this->isMinus);
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
        std::vector<T> result(N);

        T vmax = (T)0;

        for(int k=0;k<N;++k)
        {
            vmax = std::max(vmax,std::abs(weights[k]));
        }

        std::vector<T> gradRows =
            upwindGradOp->getAbsRowSum(transposed);

        for(int k=0;k<N;++k)
        {
            result[k] = vmax * gradRows[k];
        }

        return result;
    }

    T getMaxRowSumAbs(bool transposed) override
    {
        T vmax = (T)0;

        for(int k=0;k<N;++k)
        {
            vmax = std::max(vmax,std::abs(weights[k]));
        }

        return vmax *
            upwindGradOp->getMaxRowSumAbs(transposed);
    }

    #ifdef __CUDACC__
    struct AbsFunctor {
        __host__ __device__ T operator()(const T& x) const {
            return std::abs(x);
        }
    };

    struct ScaleFunctor {
        T factor;
        ScaleFunctor(T _factor) : factor(_factor) {}
        __host__ __device__ T operator()(const T& x) const {
            return factor * x;
        }
    };

    std::vector<T> getAbsRowSumCUDA(bool transposed) override
    {
        Tdata absWeights(N);
        thrust::transform(weights.begin(), weights.end(), absWeights.begin(), AbsFunctor());
        
        T vmax = *thrust::max_element(absWeights.begin(), absWeights.end());

        std::vector<T> hostGradRows = upwindGradOp->getAbsRowSumCUDA(transposed);
        thrust::device_vector<T> devGradRows = hostGradRows;
        thrust::device_vector<T> devResult(N);

        thrust::transform(devGradRows.begin(), devGradRows.end(), devResult.begin(), ScaleFunctor(vmax));

        std::vector<T> hostResult(N);
        thrust::copy(devResult.begin(), devResult.end(), hostResult.begin());

        return hostResult;
    }
    #endif

private:
    /*
    void doTimes(bool transposed,
                const Tdata& input,
                Tdata& output,
                const mySign s)
    {

        if(!transposed)
        {
            Tdata weightedInput(N);
            #pragma omp parallel for
            for(int k=0;k<N;++k)
            {
                weightedInput[k] = weights[k] * input[k];
            }

            upwindGradOp->doTimes(false, weightedInput, tmp, EQUALS);

            #pragma omp parallel for
            for(int k=0;k<N;++k)
            {
                switch(s)
                {
                    case PLUS:
                        output[k] += tmp[k];
                        break;

                    case MINUS:
                        output[k] -= tmp[k];
                        break;

                    case EQUALS:
                        output[k] = tmp[k];
                        break;
                }
            }
        }
        else
        {
            upwindGradOp->doTimes(true, input, tmp, EQUALS);

            #pragma omp parallel for
            for(int k=0;k<N;++k)
            {
                T value = weights[k] * tmp[k];

                switch(s)
                {
                    case PLUS:
                        output[k] += value;
                        break;

                    case MINUS:
                        output[k] -= value;
                        break;

                    case EQUALS:
                        output[k] = value;
                        break;
                }
            }
        }
    }*/

    void doTimes(bool transposed, const Tdata& input, Tdata& output, const mySign s)
    {
    #ifdef __CUDACC__
        if (!transposed)
        {
            Tdata weightedInput(N);
            thrust::transform(
                weights.begin(), weights.end(),
                input.begin(),
                weightedInput.begin(),
                thrust::multiplies<T>()
            );

            upwindGradOp->doTimes(false, weightedInput, tmp, EQUALS);

            if (s == PLUS) {
                thrust::transform(output.begin(), output.end(), tmp.begin(), output.begin(), thrust::plus<T>());
            } else if (s == MINUS) {
                thrust::transform(output.begin(), output.end(), tmp.begin(), output.begin(), thrust::minus<T>());
            } else if (s == EQUALS) {
                thrust::copy(tmp.begin(), tmp.end(), output.begin());
            }
        }
        else
        {
            upwindGradOp->doTimes(true, input, tmp, EQUALS);

            Tdata value(N);
            thrust::transform(
                weights.begin(), weights.end(),
                tmp.begin(),
                value.begin(),
                thrust::multiplies<T>()
            );

            if (s == PLUS) {
                thrust::transform(output.begin(), output.end(), value.begin(), output.begin(), thrust::plus<T>());
            } else if (s == MINUS) {
                thrust::transform(output.begin(), output.end(), value.begin(), output.begin(), thrust::minus<T>());
            } else if (s == EQUALS) {
                thrust::copy(value.begin(), value.end(), output.begin());
            }
        }
    #else
        if(!transposed)
        {
            Tdata weightedInput(N);
            #pragma omp parallel for
            for(int k=0;k<N;++k)
            {
                weightedInput[k] = weights[k] * input[k];
            }

            upwindGradOp->doTimes(false, weightedInput, tmp, EQUALS);

            #pragma omp parallel for
            for(int k=0;k<N;++k)
            {
                switch(s)
                {
                    case PLUS:
                        output[k] += tmp[k];
                        break;
                    case MINUS:
                        output[k] -= tmp[k];
                        break;
                    case EQUALS:
                        output[k] = tmp[k];
                        break;
                }
            }
        }
        else
        {
            upwindGradOp->doTimes(true, input, tmp, EQUALS);

            #pragma omp parallel for
            for(int k=0;k<N;++k)
            {
                T value = weights[k] * tmp[k];

                switch(s)
                {
                    case PLUS:
                        output[k] += value;
                        break;
                    case MINUS:
                        output[k] -= value;
                        break;
                    case EQUALS:
                        output[k] = value;
                        break;
                }
            }
        }
    #endif
    }
};

#endif