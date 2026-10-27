#ifndef flexUpwindGradientOperator_H
#define flexUpwindGradientOperator_H

#include <vector>
#include <stdexcept>
#include "flexLinearOperator.h"
#include "flexParallel.h"

//! stencil of the upwind gradient v * du along one axis (used on CPU and GPU)
/*!
	(Ku)_i = v_i * (u_{i+1} - u_i)   if v_i < 0 and i is not the last element along the axis
	(Ku)_i = v_i * (u_i - u_{i-1})   if v_i > 0 and i is not the first element along the axis
	(Ku)_i = 0                       otherwise
	The transposed version is the exact adjoint of this matrix.
*/
template <typename T>
struct flexUpwindGradientFunctor
{
	T* out;
	const T* in;
	const T* v;
	flexAxis ax;
	bool transposed;
	mySign s;

	FLEX_HD bool forwardRow(const int i, const int c) const { return c < ax.n - 1 && v[i] < T(0); }
	FLEX_HD bool backwardRow(const int i, const int c) const { return c > 0 && v[i] > T(0); }

	FLEX_HD void operator()(const int i) const { this->apply(i, ax.coord(i)); }

	FLEX_HD void apply(const int i, const int c) const
	{
		const int st = ax.stride;
		T val = T(0);

		if (!transposed)
		{
			if (forwardRow(i, c)) val = v[i] * (in[i + st] - in[i]);
			else if (backwardRow(i, c)) val = v[i] * (in[i] - in[i - st]);
		}
		else
		{
			if (forwardRow(i, c)) val -= v[i] * in[i];
			else if (backwardRow(i, c)) val += v[i] * in[i];

			if (c > 0 && forwardRow(i - st, c - 1)) val += v[i - st] * in[i - st];
			if (c < ax.n - 1 && backwardRow(i + st, c + 1)) val -= v[i + st] * in[i + st];
		}

		flexApplySign(out[i], val, s);
	}
};

//! exact absolute row (or column, if transposed) sums of the upwind gradient matrix
template <typename T>
struct flexUpwindGradientRowSumFunctor
{
	T* out;
	const T* v;
	flexAxis ax;
	bool transposed;

	FLEX_HD bool activeRow(const int i, const int c) const
	{
		return (c < ax.n - 1 && v[i] < T(0)) || (c > 0 && v[i] > T(0));
	}

	FLEX_HD void operator()(const int i) const
	{
		const int c = ax.coord(i);
		const int st = ax.stride;
		T val = T(0);

		if (!transposed)
		{
			if (activeRow(i, c)) val = T(2) * flexAbsHD(v[i]);
		}
		else
		{
			if (activeRow(i, c)) val += flexAbsHD(v[i]);
			if (c > 0 && v[i - st] < T(0)) val += flexAbsHD(v[i - st]);			//forward row i-1 touches i
			if (c < ax.n - 1 && v[i + st] > T(0)) val += flexAbsHD(v[i + st]);	//backward row i+1 touches i
		}
		out[i] = val;
	}
};

//! represents an upwind gradient operator
template<typename T>
class flexUpwindGradientOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	std::vector<int> inputDimension;
	int gradDirection;
	std::vector<T> motionFieldHost;
	Tdata motionField;
	flexAxis axis;

public:

	//! initializes the upwind gradient operator
	/*!
		\param AInputDimension vector of dimensions
		\param aGradDirection direction of gradient. 0 for first dimension and so on.
		\param aMotionField motion field in the direction aGradDirection
		\param aMinus determines if operator is negated \sa isMinus
	*/
	flexUpwindGradientOperator(std::vector<int> AInputDimension, int aGradDirection, const std::vector<T>& aMotionField, bool aMinus) :
		flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension), upwindGradientOp, aMinus),
		inputDimension(AInputDimension),
		gradDirection(aGradDirection),
		motionFieldHost(aMotionField),
		axis(flexMakeAxis(AInputDimension, aGradDirection))
	{
		if (aGradDirection < 0 || aGradDirection >= (int)AInputDimension.size())
			throw std::runtime_error("flexUpwindGradientOperator: gradDirection out of range");
		if ((int)aMotionField.size() != vectorProduct(AInputDimension))
			throw std::runtime_error("flexUpwindGradientOperator: motion field has wrong size");

		flexAssign(this->motionField, this->motionFieldHost);
	};

	flexUpwindGradientOperator<T>* copy()
	{
		return new flexUpwindGradientOperator<T>(this->inputDimension, this->gradDirection, this->motionFieldHost, this->isMinus);
	}

	void doTimes(bool transposed, const Tdata &input, Tdata &output, mySign s)
	{
		if (this->isMinus)
			s = flexFlipSign(s);

		flexUpwindGradientFunctor<T> f = { flexPtr(output), flexPtr(input), flexPtr(this->motionField), this->axis, transposed, s };
		flexForEachAxis(this->axis, (int)output.size(), f);
	}

	void timesPlus(bool transposed, const Tdata &input, Tdata &output)
	{
		this->doTimes(transposed, input, output, PLUS);
	}

	void timesMinus(bool transposed, const Tdata &input, Tdata &output)
	{
		this->doTimes(transposed, input, output, MINUS);
	}

	void times(bool transposed, const Tdata &input, Tdata &output)
	{
		this->doTimes(transposed, input, output, EQUALS);
	}

	void absRowSum(bool transposed, Tdata& result)
	{
		result.resize(this->getNumRows());
		flexUpwindGradientRowSumFunctor<T> f = { flexPtr(result), flexPtr(this->motionField), this->axis, transposed };
		flexForEach((int)result.size(), f);
	}

	T getMaxRowSumAbs(bool transposed)
	{
		std::vector<T> rowSum = this->getAbsRowSum(transposed);
		return rowSum.empty() ? T(0) : *std::max_element(rowSum.begin(), rowSum.end());
	}

	std::vector<T> getAbsRowSum(bool transposed)
	{
		Tdata result;
		this->absRowSum(transposed, result);
		return flexToHost(result);
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
