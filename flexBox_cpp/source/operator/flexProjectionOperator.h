#ifndef flexProjectionOperator_H
#define flexProjectionOperator_H

#include <vector>
#include <memory>
#include "flexLinearOperator.h"
#include "flexGradientOperator.h"

/*
    Operator to apply a linear operator only to the i-th block where i will stand for the timestep we are in for the joint image reconstruction and mostion estimation approach
*/

template<typename T>
class flexProjectionOperator : public flexLinearOperator<T>
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
    flexLinearOperator<T>* op;

    mutable Tdata tmpInput;
    mutable Tdata tmp;
    mutable Tdata tmpLong;

public:

    flexProjectionOperator(flexLinearOperator<T>* op, int index, const std::vector<int>& AInputDimension, bool aIsMinus): flexLinearOperator<T>(op->getNumRows(), vectorProduct(AInputDimension), projectionOp, aIsMinus), inputDimension(AInputDimension), index(index), op(op)
    {
        N = vectorProduct(inputDimension);
        NPerIndex = inputDimension[0] * inputDimension[1];
        
        tmpInput.resize(op->getNumCols(), (T)0.0);
        
        tmp.resize(op->getNumRows(), (T)0.0);

        if(op->getNumCols() != NPerIndex)
        {
            throw std::runtime_error("The given inner operator's columns do not match NPerIndex.");
        }
    }

    ~flexProjectionOperator()
    {
        /*
        if (op != nullptr)
        {
            delete op;
            op = nullptr;
        }
        */
    }

    flexProjectionOperator<T>* copy()
    {
        return new flexProjectionOperator<T>(this->op->copy(), index,inputDimension,this->isMinus);
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
        if (!transposed)
        {
            return op->getAbsRowSum(false);
        }
        else
        {
            std::vector<T> result(this->N, (T)0);

            std::vector<T> absRowSumOperator = op->getAbsRowSum(true);

            for (int i = 0; i < NPerIndex; ++i)
            {
                result[NPerIndex * index + i] = absRowSumOperator[i]; 
            }

            return result;
        }
    }

    T getMaxRowSumAbs(bool transposed) override
    {
        return op->getMaxRowSumAbs(transposed);
    }

private:

    void doTimes(bool transposed,const Tdata& input,Tdata& output,const mySign s)
    {   
        if (!transposed)
        {
            for(int k = 0; k < NPerIndex; ++k)
            {
                tmpInput[k] = input[k + index * NPerIndex];
            }

            op->times(false, tmpInput, tmp);
            switch(s) {
                #pragma omp parallel for
                for(int k = 0; k < op->getNumRows(); ++k)
                {
                    switch(s) {
                        case PLUS:   output[k] += tmp[k]; break;
                        case MINUS:  output[k] -= tmp[k]; break;
                        case EQUALS: output[k]  = tmp[k]; break;
                    }
                }
            }
        }
        else
        {
            op->times(true, input, tmpInput);

            #pragma omp parallel for
            for(int k = 0; k < NPerIndex; ++k)
            {
                switch(s) {
                    case PLUS:   output[k + index * NPerIndex] += tmpInput[k]; break;
                    case MINUS:  output[k + index * NPerIndex] -= tmpInput[k]; break;
                    case EQUALS: output[k + index * NPerIndex]  = tmpInput[k]; break;
                }
            }
        }
    }
};

#endif