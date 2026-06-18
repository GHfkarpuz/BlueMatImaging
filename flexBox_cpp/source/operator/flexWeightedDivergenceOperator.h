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

    #ifdef __CUDACC__
    struct AbsProductOp {
        template <typename Tuple>
        __host__ __device__ void operator()(Tuple t) {
            thrust::get<0>(t) = std::abs(thrust::get<1>(t)) * thrust::get<2>(t);
        }
    };

    std::vector<T> getAbsRowSumCUDA(bool transposed) override
    {
        std::vector<T> hostGradRowSum = gradOp->getAbsRowSumCUDA(transposed);
        thrust::device_vector<T> devGradRowSum = hostGradRowSum;
        thrust::device_vector<T> devResult(N);

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(devResult.begin(), weights.begin(), devGradRowSum.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(devResult.end(), weights.end(), devGradRowSum.end())),
            AbsProductOp()
        );

        std::vector<T> hostResult(N);
        thrust::copy(devResult.begin(), devResult.end(), hostResult.begin());

        return hostResult;
    }
    #endif

private:
    /*
    void doTimes(bool transposed,const Tdata& input,Tdata& output,const mySign s)
    {
        if(!transposed)
        {
            Tdata product(N, (T)0.0);

            #pragma omp parallel for
            for(int k=0; k<N; ++k)
            {
                product[k] = weights[k] * input[k];
            }

            gradOp->doTimes(false,product,tmp,EQUALS);

            #pragma omp parallel for
            for(int k=0; k<N; ++k)
            {
                switch(s)
                {
                    case PLUS:
                        output[k] += tmp[k];;
                        break;

                    case MINUS:
                        output[k] -= tmp[k];;
                        break;

                    case EQUALS:
                        output[k] = tmp[k];;
                        break;
                }
            }
        }
        else
        {
            gradOp->doTimes(true, input, tmp, s);

            #pragma omp parallel for
            for(int k=0; k<N; ++k)
            {
                tmp[k] *= weights[k];
            }

            for(int k = 0; k<N; ++k)
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
    }*/

    void doTimes(bool transposed, const Tdata& input, Tdata& output, const mySign s)
    {
    #ifdef __CUDACC__
        if (!transposed)
        {
            Tdata product(N);
            thrust::transform(
                weights.begin(), weights.end(),
                input.begin(),
                product.begin(),
                thrust::multiplies<T>()
            );

            gradOp->doTimes(false, product, tmp, EQUALS);

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
            gradOp->doTimes(true, input, tmp, EQUALS);

            thrust::transform(
                tmp.begin(), tmp.end(),
                weights.begin(),
                tmp.begin(),
                thrust::multiplies<T>()
            );

            if (s == PLUS) {
                thrust::transform(output.begin(), output.end(), tmp.begin(), output.begin(), thrust::plus<T>());
            } else if (s == MINUS) {
                thrust::transform(output.begin(), output.end(), tmp.begin(), output.begin(), thrust::minus<T>());
            } else if (s == EQUALS) {
                thrust::copy(tmp.begin(), tmp.end(), output.begin());
            }
        }
    #else
        if(!transposed)
        {
            Tdata product(N, (T)0.0);

            #pragma omp parallel for
            for(int k=0; k<N; ++k)
            {
                product[k] = weights[k] * input[k];
            }

            gradOp->doTimes(false,product,tmp,EQUALS);

            #pragma omp parallel for
            for(int k=0; k<N; ++k)
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
            gradOp->doTimes(true, input, tmp, s);

            #pragma omp parallel for
            for(int k=0; k<N; ++k)
            {
                tmp[k] *= weights[k];
            }

            for(int k = 0; k<N; ++k)
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
    #endif
    }
};

#endif