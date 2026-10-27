#ifndef flexDiagonalBasedOperator_H
#define flexDiagonalBasedOperator_H

#include <vector>
#include <algorithm>
#include "flexLinearOperator.h"
#include "flexParallel.h"

//! base class for operators that are a diagonal matrix diag(d) whose entries are computed by the derived class
/*!
	Derived classes fill the protected vector d in their constructor (on the CPU or the GPU).
	Since the operator is diagonal it is self-adjoint, the flag transposed is ignored.
*/
template<typename T>
class flexDiagonalBasedOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

protected:
	Tdata d;

public:
	flexDiagonalBasedOperator(int aN, linOp aType, bool aIsMinus) : flexLinearOperator<T>(aN, aN, aType, aIsMinus), d(aN, T(0))
	{
	}

	//! returns the diagonal entries (host copy)
	std::vector<T> getDiagonal()
	{
		return flexToHost(d);
	}

	void doTimes(const Tdata &input, Tdata &output, const mySign s)
	{
		flexMul(output, this->d, input, s);
	}

	void times(bool transposed, const Tdata &input, Tdata &output)
	{
		this->doTimes(input, output, EQUALS);
	}

	void timesPlus(bool transposed, const Tdata &input, Tdata &output)
	{
		this->doTimes(input, output, this->isMinus ? MINUS : PLUS);
	}

	void timesMinus(bool transposed, const Tdata &input, Tdata &output)
	{
		this->doTimes(input, output, this->isMinus ? PLUS : MINUS);
	}

	std::vector<T> getAbsRowSum(bool transposed)
	{
		std::vector<T> result = flexToHost(d);
		for (size_t k = 0; k < result.size(); ++k)
			result[k] = std::abs(result[k]);
		return result;
	}

	T getMaxRowSumAbs(bool transposed)
	{
		std::vector<T> rowSum = this->getAbsRowSum(transposed);
		return rowSum.empty() ? T(0) : *std::max_element(rowSum.begin(), rowSum.end());
	}

	#ifdef __CUDACC__
	thrust::device_vector<T> getAbsRowSumCUDA(bool transposed)
	{
		thrust::device_vector<T> ones(d.size(), T(1));
		thrust::device_vector<T> result(d.size());
		flexAbsMulFunctor<T> f = { flexPtr(result), flexPtr(d), flexPtr(ones) };
		flexForEach((int)result.size(), f);
		return result;
	}
	#endif
};

#endif
