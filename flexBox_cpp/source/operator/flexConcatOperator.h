#ifndef flexConcatOperator_H
#define flexConcatOperator_H


#include "vector"
#include "tools.h"
#include "flexLinearOperator.h"
#ifdef __CUDACC__
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>
#endif

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
	flexConcatOperator(flexLinearOperator<T>* aA, flexLinearOperator<T>* aB, mySign aS, bool aMinus) : A(aA), B(aB), s(aS), tmpVec(aB->getNumRows()), flexLinearOperator<T>(aA->getNumRows(), aB->getNumCols(), concatOp, aMinus)
	{

	}

	//! the concatenation owns its two operators (they are created only for it), so they are deleted with it
	~flexConcatOperator()
	{
		delete A;
		delete B;
	}

	//! deep copy (the copy owns copies of both operators)
	flexConcatOperator<T>* copy()
	{
		return new flexConcatOperator<T>(this->A->copy(), this->B->copy(), this->s, this->isMinus);
	}

	//! applies the plain operator (like all FlexBox operators, times() ignores isMinus)
	void times(bool transposed, const Tdata &input, Tdata &output)
	{
		#ifdef __CUDACC__
			thrust::fill(output.begin(), output.end(), (T)0);
		#else
			std::fill(output.begin(), output.end(), (T)0);
		#endif

		const bool minus = this->isMinus;
		this->isMinus = false;
		this->timesPlus(transposed, input, output);
		this->isMinus = minus;
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
                #pragma omp parallel for schedule(static) if(flexUseThreads(result.size()))
                for (size_t k = 0; k < result.size(); ++k)
                {
                    result[k] = rowSumA[k] + rowSumB[k]; 
                }
                break;
            case MINUS: //right?
                result.resize(rowSumA.size());
                #pragma omp parallel for schedule(static) if(flexUseThreads(result.size()))
                for (size_t k = 0; k < result.size(); ++k)
                {
                    result[k] = rowSumA[k] + rowSumB[k]; 
                }
                break;

            case COMPOSE:
            {
                // row sums of A*B (column sums if transposed), per row instead of one global bound:
                //   rows:    sum_j |(AB)_ij| <= sum_k |A_ik| r_B(k) <= r_A(i) * max r_B,   exact r_A(i) * r_B(i) if A is diagonal
                //   columns: sum_i |(AB)_ij| <= sum_k c_A(k) |B_kj| <= c_B(j) * max c_A,   exact c_B(j) * c_A(j) if B is diagonal
                // (rowSumA / rowSumB are the column sums of A / B if transposed)
                const bool elementwise = transposed ? isDiagonalLike(this->B) : isDiagonalLike(this->A);
                const std::vector<T>& outer = transposed ? rowSumB : rowSumA;
                const std::vector<T>& inner = transposed ? rowSumA : rowSumB;
                result.resize(outer.size());
                if (elementwise && inner.size() == outer.size())
                {
                    #pragma omp parallel for schedule(static) if(flexUseThreads(result.size()))
                    for (int k = 0; k < (int)result.size(); ++k)
                        result[k] = outer[k] * inner[k];
                }
                else
                {
                    const T maxInner = inner.empty() ? T(0) : *std::max_element(inner.begin(), inner.end());
                    #pragma omp parallel for schedule(static) if(flexUseThreads(result.size()))
                    for (int k = 0; k < (int)result.size(); ++k)
                        result[k] = outer[k] * maxInner;
                }
                break;
            }
        }
        return result;
    }

    //! diagonal and identity operators: the row sums of A*B are then the products of the row sums
    static bool isDiagonalLike(flexLinearOperator<T>* op)
    {
        // operators that are diagonal matrices (the optical flow operators multiply pixelwise with an image derivative)
        if (op->type == diagonalOp || op->type == identityOp || op->type == gradientDiagonalOp || op->type == opticalFlowOp ||
            op->type == opticalFlowPhaseFieldOp || op->type == massPreservationPhaseFieldOp)
            return true;
        // sums and products of diagonal matrices are diagonal
        if (op->type == concatOp)
        {
            flexConcatOperator<T>* c = dynamic_cast<flexConcatOperator<T>*>(op);
            return c != nullptr && isDiagonalLike(c->A) && isDiagonalLike(c->B);
        }
        return false;
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
                result.resize(rowSumA.size());
                thrust::transform(rowSumA.begin(), rowSumA.end(), rowSumB.begin(), result.begin(), thrust::plus<T>());
                break;
            case MINUS:
                result.resize(rowSumA.size());
                thrust::transform(rowSumA.begin(), rowSumA.end(), rowSumB.begin(), result.begin(), thrust::plus<T>());
                break;

            case COMPOSE:
            {
                // same bounds as in getAbsRowSum (per row / column instead of one global value)
                const bool elementwise = transposed ? isDiagonalLike(this->B) : isDiagonalLike(this->A);
                const thrust::device_vector<T>& outer = transposed ? rowSumB : rowSumA;
                const thrust::device_vector<T>& inner = transposed ? rowSumA : rowSumB;
                result.resize(outer.size());
                if (elementwise && inner.size() == outer.size())
                {
                    thrust::transform(outer.begin(), outer.end(), inner.begin(), result.begin(), thrust::multiplies<T>());
                }
                else
                {
                    const T maxInner = inner.empty() ? T(0) : *thrust::max_element(inner.begin(), inner.end());
                    thrust::transform(outer.begin(), outer.end(), thrust::make_constant_iterator(maxInner), result.begin(), thrust::multiplies<T>());
                }
                break;
            }
        }
        return result;
    }
    #endif
};

#endif
