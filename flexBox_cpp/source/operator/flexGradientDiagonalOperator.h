#ifndef flexGradientDiagonalOperator_H
#define flexGradientDiagonalOperator_H

#include <vector>
#include "flexDiagonalBasedOperator.h"
#include "flexGradientOperator.h"

//! represents the diagonal operator diag(grad(w)): gets a weight w and applies s -> (D w) * s
template<typename T>
class flexGradientDiagonalOperator : public flexDiagonalBasedOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	std::vector<T> weights;
	std::vector<int> inputDimension;
	int gradDirection;
	gradientType type;

public:
	/*!
		\param aWeights the weight w whose derivative is placed on the diagonal
		\param aInputDimension vector of dimensions
		\param aGradDirection direction of gradient. 0 for first dimension and so on.
		\param aType type of gradient. Possible values are forward, backward and central.
		\param aMinus determines if operator is negated \sa isMinus
	*/
	flexGradientDiagonalOperator(const std::vector<T>& aWeights, std::vector<int> aInputDimension, int aGradDirection, gradientType aType, bool aMinus)
	: flexDiagonalBasedOperator<T>(static_cast<int>(aWeights.size()), gradientDiagonalOp, aMinus), weights(aWeights), inputDimension(aInputDimension), gradDirection(aGradDirection), type(aType)
	{
		Tdata w;
		flexAssign(w, aWeights);

		flexGradientOperator<T> A(inputDimension, gradDirection, type, false);
		A.doTimes(false, w, this->d, EQUALS);
	}

	flexGradientDiagonalOperator<T>* copy()
	{
		return new flexGradientDiagonalOperator<T>(this->weights, this->inputDimension, this->gradDirection, this->type, this->isMinus);
	}
};

#endif
