#ifndef flexMassPreservationOperator_H
#define flexMassPreservationOperator_H

#include <vector>
#include <algorithm>
#include "flexLinearOperator.h"
#include "flexGradientOperator.h"
#include "flexParallel.h"

//! row sums of the mass preservation operator as in the original implementation: |d_i u| + |u| * rowSum(D)
template <typename T>
struct flexMassPreservationRowSumFunctor
{
	T* out; const T* gU; const T* u; const T* gradRowSum;
	FLEX_HD void operator()(const int k) const { out[k] = flexAbsHD(gU[k]) + flexAbsHD(u[k]) * gradRowSum[k]; }
};

//! represents the linear operator v_i -> d_i(u v_i) of the mass preservation condition u_t + div(uv) = 0 (one direction i)
/*!
	K(v) = D (u v) with central differences D in direction i, K^T(y) = u * (D^T y)
*/
template <typename T>
class flexMassPreservationOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	std::vector<T> imageHost;
	Tdata image;
	std::vector<int> inputDimension;
	Tdata gradU; //derivative of u
	Tdata tmp;
	flexGradientOperator<T>* A;
	int direction;
	int N;

public:
	/*!
		\param aImage the image u
		\param AInputDimension vector of dimensions
		\param ADirection direction of the derivative
		\param aIsMinus determines if operator is negated \sa isMinus
	*/
	flexMassPreservationOperator(const std::vector<T>& aImage, std::vector<int> AInputDimension, int ADirection, bool aIsMinus)
	: flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension), massPreservationOp, aIsMinus),
	  imageHost(aImage), inputDimension(AInputDimension), direction(ADirection)
	{
		this->N = vectorProduct(inputDimension);
		flexAssign(image, imageHost);
		tmp.resize(N);
		gradU.resize(N);

		A = new flexGradientOperator<T>(AInputDimension, direction, central, false);
		A->doTimes(false, this->image, this->gradU, EQUALS);
	}

	~flexMassPreservationOperator()
	{
		delete A;
	}

	flexMassPreservationOperator<T>* copy()
	{
		return new flexMassPreservationOperator<T>(imageHost, inputDimension, direction, this->isMinus);
	}

	std::vector<T> give_gradU()
	{
		return flexToHost(gradU);
	}

	void times(bool transposed, const Tdata &input, Tdata &output)
	{
		this->doTimes(transposed, input, output, EQUALS);
	}

	void timesPlus(bool transposed, const Tdata &input, Tdata &output)
	{
		this->doTimes(transposed, input, output, this->isMinus ? MINUS : PLUS);
	}

	void timesMinus(bool transposed, const Tdata &input, Tdata &output)
	{
		this->doTimes(transposed, input, output, this->isMinus ? PLUS : MINUS);
	}

	void doTimes(bool transposed, const Tdata &input, Tdata &output, const mySign s)
	{
		if (!transposed)
		{
			flexMul(tmp, input, image, EQUALS);
			A->doTimes(false, tmp, output, s);
		}
		else
		{
			A->doTimes(true, input, tmp, EQUALS);
			flexMul(output, tmp, image, s);
		}
	}

	void absRowSum(bool transposed, Tdata& result)
	{
		Tdata gradRowSum;
		A->absRowSum(transposed, gradRowSum);
		result.resize(N);
		flexMassPreservationRowSumFunctor<T> f = { flexPtr(result), flexPtr(gradU), flexPtr(image), flexPtr(gradRowSum) };
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
		std::vector<T> rowSum = this->getAbsRowSum(transposed);
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
