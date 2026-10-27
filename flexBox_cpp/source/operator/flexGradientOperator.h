#ifndef flexGradientOperator_H
#define flexGradientOperator_H

#include <vector>
#include <stdexcept>
#include "flexLinearOperator.h"
#include "flexParallel.h"

//! stencil of the gradient along one axis, evaluated for one output element (used on CPU and GPU)
/*!
	forward:  (Du)_i = u_{i+1} - u_i for i < n-1, (Du)_{n-1} = 0
	central:  (Du)_i = (u_{i+1} - u_{i-1}) / 2 inside, one sided differences at the two borders
	backward: (Du)_i = u_i - u_{i-1} for i > 0, (Du)_0 = 0  (a constant has derivative 0, like forward)
	The transposed versions are the exact adjoints of these matrices.
*/
template <typename T>
struct flexGradientFunctor
{
	T* out;
	const T* in;
	flexAxis ax;
	gradientType type;	//!< forward, backward or central
	bool transposed;
	T factor;		//!< +1 or -1
	mySign s;

	FLEX_HD void operator()(const int i) const { this->apply(i, ax.coord(i)); }

	FLEX_HD void apply(const int i, const int c) const
	{
		const int n = ax.n;
		const int st = ax.stride;
		T val = T(0);

		if (type == central)
		{
			if (n >= 2)
			{
				if (!transposed)
				{
					if (c == 0) val = in[i + st] - in[i];
					else if (c == n - 1) val = in[i] - in[i - st];
					else val = T(0.5) * (in[i + st] - in[i - st]);
				}
				else
				{
					//column c of the central difference matrix
					if (c == 0) val -= in[i];
					if (c == 1) val += in[i - st];
					if (c == n - 1) val += in[i];
					if (c == n - 2) val -= in[i + st];
					if (c - 1 >= 1 && c - 1 <= n - 2) val += T(0.5) * in[i - st];
					if (c + 1 >= 1 && c + 1 <= n - 2) val -= T(0.5) * in[i + st];
				}
			}
		}
		else if (type == backward)
		{
			if (!transposed)
			{
				if (c > 0) val = in[i] - in[i - st];
			}
			else
			{
				//column c: row c has +1 at c (c > 0), row c+1 has -1 at c
				if (c > 0) val += in[i];
				if (c < n - 1) val -= in[i + st];
			}
		}
		else
		{
			if (!transposed)
			{
				if (c < n - 1) val = in[i + st] - in[i];
			}
			else
			{
				if (c > 0) val += in[i - st];
				if (c < n - 1) val -= in[i];
			}
		}

		flexApplySign(out[i], factor * val, s);
	}
};

//! absolute row (or column, if transposed) sums of the gradient matrix
template <typename T>
struct flexGradientRowSumFunctor
{
	T* out;
	flexAxis ax;
	gradientType type;
	bool transposed;

	FLEX_HD void operator()(const int i) const
	{
		const int c = ax.coord(i);
		const int n = ax.n;

		if (type != central)
		{
			out[i] = T(2); //upper bound as in the original FlexBox
			return;
		}

		T val = T(0);
		if (n >= 2)
		{
			if (!transposed)
			{
				val = (c == 0 || c == n - 1) ? T(2) : T(1);
			}
			else
			{
				if (c == 0) val += T(1);
				if (c == 1) val += T(1);
				if (c == n - 1) val += T(1);
				if (c == n - 2) val += T(1);
				if (c - 1 >= 1 && c - 1 <= n - 2) val += T(0.5);
				if (c + 1 >= 1 && c + 1 <= n - 2) val += T(0.5);
			}
		}
		out[i] = val;
	}
};

//! represents a gradient operator (forward, backward or central differences along one axis)
template<typename T>
class flexGradientOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	std::vector<int> inputDimension;
	int gradDirection;
	gradientType type;
	flexAxis axis;

public:

	//! initializes the gradient operator
	/*!
		\param AInputDimension vector of dimensions
		\param aGradDirection direction of gradient. 0 for first dimension and so on.
		\param aType type of gradient. Possible values are forward, backward and central.
		\param aMinus determines if operator is negated \sa isMinus
	*/
	flexGradientOperator(std::vector<int> AInputDimension, int aGradDirection, gradientType aType, bool aMinus) :
		flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension), gradientOp, aMinus),
		inputDimension(AInputDimension),
		gradDirection(aGradDirection),
		type(aType),
		axis(flexMakeAxis(AInputDimension, aGradDirection))
	{
		if (aGradDirection < 0 || aGradDirection >= (int)AInputDimension.size())
			throw std::runtime_error("flexGradientOperator: gradDirection out of range");
	};

	flexGradientOperator<T>* copy()
	{
		return new flexGradientOperator<T>(this->inputDimension, this->gradDirection, this->type, this->isMinus);
	}

	//! applies the operator. isMinus negates PLUS and MINUS, EQUALS always writes the plain operator (as in FlexBox)
	void doTimes(bool transposed, const Tdata &input, Tdata &output, mySign s)
	{
		if (this->isMinus)
			s = flexFlipSign(s);

		T factor = T(1);
		gradientType stencil = this->type;

		flexGradientFunctor<T> f = { flexPtr(output), flexPtr(input), this->axis, stencil, transposed, factor, s };
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

	T getMaxRowSumAbs(bool transposed)
	{
		return static_cast<T>(2);
	}

	void absRowSum(bool transposed, Tdata& result)
	{
		result.resize(this->getNumRows());

		gradientType stencil = this->type;
		bool tr = transposed;

		flexGradientRowSumFunctor<T> f = { flexPtr(result), this->axis, stencil, tr };
		flexForEach((int)result.size(), f);
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
