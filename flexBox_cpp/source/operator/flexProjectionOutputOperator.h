#ifndef flexProjectionOutputOperator_H
#define flexProjectionOutputOperator_H

#include <vector>
#include <memory>
#include "flexLinearOperator.h"
#include "flexGradientOperator.h"

/*
    Operator to apply a linear operator only to the i-th block where i will stand for the timestep we are in for the joint image reconstruction and mostion estimation approach
*/

template<typename T>
class flexProjectionOutputOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
    typedef thrust::device_vector<T> Tdata;
#else
    typedef std::vector<T> Tdata;
#endif

private:
    std::vector<int> inputDimension;//including the number of frames in the last dimension for motion estimation methods especially the joint approach

    int direction;
    int N;
    int index;
    int NPerIndex;


public:

    flexProjectionOutputOperator(int index, const std::vector<int>& AInputDimension, bool aIsMinus): index(index), inputDimension(AInputDimension), flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension), projectionOutputOp, aIsMinus)
    {
        N = vectorProduct(inputDimension);
        NPerIndex = inputDimension[0] * inputDimension[1];
    }

    ~flexProjectionOutputOperator()
    {
    }

    flexProjectionOutputOperator<T>* copy()
    {
        return new flexProjectionOutputOperator<T>(index,inputDimension,this->isMinus);
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
        for(int i = 0; i<NPerIndex;++i)
        {
            result[index*NPerIndex + i] = (T)1;
        }
        return result;
        /*
        else
        {
            std::vector<T> result(N, (T)0);
            for(int k=0; k<NPerIndex; ++k)
            {
                result[k+NPerIndex*index] = (T)1;
            }

            return result;
        }
        */
    }

    T getMaxRowSumAbs(bool transposed) override
    {
		return (T)1;
    }

private:

    void doTimes(bool transposed,const Tdata& input,Tdata& output,const mySign s)
    {   
        std::fill(output.begin(), output.end(), 0.0);
        #pragma omp parallel for
        for(int k=0; k<NPerIndex; ++k)
        {
            switch(s)
            {
                case PLUS:
                    output[k+index*NPerIndex] += input[k];
                    break;

                case MINUS:
                    output[k+index*NPerIndex] -= input[k];
                    break;

                case EQUALS:
                    output[k+index*NPerIndex] = input[k];
                    break;
            }
        }
    }
};

#endif