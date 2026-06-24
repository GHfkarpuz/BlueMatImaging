#ifndef flexConcatOperator_H
#define flexConcatOperator_H


#include "vector"
#include "tools.h"
#include "flexLinearOperator.h"

//! represents a concatenation operator
template<typename T>
class flexConcatOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
    flexLinearOperator<T>* A;
    flexLinearOperator<T>* B;
    mySign s;
	Tdata tmpVec;

public:

	//! initializes the concatenation operator
	/*!
		\param aA left hand side operator
		\param aB right hand side operator
		\param aS type of concatenation. Possible values are PLUS, SUBTRACT and COMPOSE.
		\param aMinus determines if operator is negated \sa isMinus
	*/
	flexConcatOperator(flexLinearOperator<T>* aA, flexLinearOperator<T>* aB, mySign aS, bool aMinus) : A(aA), B(aB), s(aS), tmpVec(aA->getNumRows()), flexLinearOperator<T>(aA->getNumRows(), aB->getNumCols(), concatOp, aMinus)
	{

	}

	flexConcatOperator<T>* copy()
	{
		auto cpOp = new flexConcatOperator<T>(this->A, this->B, this->s, this->isMinus);

		return cpOp;
	}

	//to implement
	void times(bool transposed, const Tdata &input, Tdata &output)
	{
	}

	void timesPlus(bool transposed, const Tdata &input, Tdata &output)
	{
        switch (this->s)
        {
            case PLUS:
            {
                if (this->isMinus)
                {
                    A->timesMinus(transposed, input, output);
                    B->timesMinus(transposed, input, output);
                }
                else
                {
                    A->timesPlus(transposed, input, output);
                    B->timesPlus(transposed, input, output);
                }
                break;
            }
            case MINUS:
            {
                if (this->isMinus)
                {
                    A->timesMinus(transposed, input, output);
                    B->timesPlus(transposed, input, output);
                }
                else
                {
                    A->timesPlus(transposed, input, output);
                    B->timesMinus(transposed, input, output);
                }
                break;
            }
            case COMPOSE:
            {
                if (transposed)
                {
                    //apply A first
					#ifdef __CUDACC__
						thrust::fill(this->tmpVec.begin(), this->tmpVec.end(), (T)0);
					#else
						std::fill(this->tmpVec.begin(), this->tmpVec.end(), (T)0);
					#endif

					A->timesPlus(true, input, this->tmpVec);

                    if (this->isMinus)
                    {
						B->timesMinus(true, this->tmpVec, output);
                    }
                    else
                    {
						B->timesPlus(true, this->tmpVec, output);
                    }
                }
                else
                {
                    //apply B first
					#ifdef __CUDACC__
						thrust::fill(this->tmpVec.begin(), this->tmpVec.end(), (T)0);
					#else
						std::fill(this->tmpVec.begin(), this->tmpVec.end(), (T)0);
					#endif

					B->timesPlus(false, input, this->tmpVec);

                    if (this->isMinus)
                    {
						A->timesMinus(false, this->tmpVec, output);
                    }
                    else
                    {
						A->timesPlus(false, this->tmpVec, output);
                    }
                }
                break;
            }
        }
	}

	void timesMinus(bool transposed, const Tdata &input, Tdata &output)
	{
        switch (this->s)
        {
            case PLUS:
            {
                if (this->isMinus)
                {
                    A->timesPlus(transposed, input, output);
                    B->timesPlus(transposed, input, output);
                }
                else
                {
                    A->timesMinus(transposed, input, output);
                    B->timesMinus(transposed, input, output);
                }
                break;
            }
            case MINUS:
            {
                if (this->isMinus)
                {
                    A->timesPlus(transposed, input, output);
                    B->timesMinus(transposed, input, output);
                }
                else
                {
                    A->timesMinus(transposed, input, output);
                    B->timesPlus(transposed, input, output);
                }
                break;
            }
            case COMPOSE:
            {
                if (transposed)
                {
                    //apply A first
					#ifdef __CUDACC__
						thrust::fill(this->tmpVec.begin(), this->tmpVec.end(), (T)0);
					#else
						std::fill(this->tmpVec.begin(), this->tmpVec.end(), (T)0);
					#endif

					A->timesPlus(true, input, tmpVec);
                    if (this->isMinus)
                    {
						B->timesPlus(true, this->tmpVec, output);
                    }
                    else
                    {
						B->timesMinus(true, this->tmpVec, output);
                    }
                }
                else
                {
                    //apply B first
					#ifdef __CUDACC__
						thrust::fill(this->tmpVec.begin(), this->tmpVec.end(), (T)0);
					#else
						std::fill(this->tmpVec.begin(), this->tmpVec.end(), (T)0);
					#endif

					B->timesPlus(false, input, this->tmpVec);
                    if (this->isMinus)
                    {
						A->timesPlus(false, this->tmpVec, output);
                    }
                    else
                    {
						A->timesMinus(false, this->tmpVec, output);
                    }
                }
                break;
            }
        }
	}

	
	T getMaxRowSumAbs(bool transposed)
    {
        #ifdef __CUDACC__
            thrust::device_vector<T> dev_res = getAbsRowSumCUDA(transposed);
            if (dev_res.empty()) return T(0);
            return *thrust::max_element(dev_res.begin(), dev_res.end());
        #else
            std::vector<T> host_res = getAbsRowSum(transposed);
            if (host_res.empty()) return T(0);
            return *std::max_element(host_res.begin(), host_res.end()); 
        #endif
    }

    std::vector<T> getAbsRowSum(bool transposed)
    {
        std::vector<T> result;
        auto rowSumA = A->getAbsRowSum(transposed);
        auto rowSumB = B->getAbsRowSum(transposed);

        switch (this->s)
        {
            case PLUS:
                result.resize(rowSumA.size());
                #pragma omp parallel for
                for (size_t k = 0; k < result.size(); ++k)
                {
                    result[k] = rowSumA[k] + rowSumB[k]; 
                }
                break;
            case MINUS: //right?
                result.resize(rowSumA.size());
                #pragma omp parallel for
                for (size_t k = 0; k < result.size(); ++k)
                {
                    result[k] = rowSumA[k] + rowSumB[k]; 
                }
                break;

            case COMPOSE:
                T maxA = *std::max_element(rowSumA.begin(), rowSumA.end());
                T maxB = *std::max_element(rowSumB.begin(), rowSumB.end());
                T maxProd = maxA * maxB;

                if(transposed)
                    result.resize(this->B->getNumCols(), maxProd);
                else
                    result.resize(this->A->getNumRows(), maxProd);
                break;
        }
        return result;
    }

    #ifdef __CUDACC__
    thrust::device_vector<T> getAbsRowSumCUDA(bool transposed)
    {
        thrust::device_vector<T> result;

        auto rowSumA = A->getAbsRowSumCUDA(transposed);
        auto rowSumB = B->getAbsRowSumCUDA(transposed);

        switch (this->s)
        {
            case PLUS:
            case MINUS:
                result.resize(rowSumA.size());
                thrust::transform(rowSumA.begin(), rowSumA.end(), rowSumB.begin(), result.begin(), thrust::plus<T>());
                break;

            case COMPOSE:
                T maxA = *thrust::max_element(rowSumA.begin(), rowSumA.end());
                T maxB = *thrust::max_element(rowSumB.begin(), rowSumB.end());
                T maxProd = maxA * maxB;

                if(transposed)
                    result.resize(this->B->getNumCols(), maxProd);
                else
                    result.resize(this->A->getNumRows(), maxProd);
                break;
        }
        return result;
    }
    #endif
};

#endif
