#ifndef flexOpticalFlowOperator_H
#define flexOpticalFlowOperator_H

#include <vector>
#include "flexDiagonalBasedOperator.h"
#include "flexGradientOperator.h"

//! represents the operator v -> d_i(u) * v for the optical flow condition u_t + grad(u)*v = 0 (one direction i)
/*!
	diag(d_i u) with central differences of the image u in direction i. The operator is self-adjoint.
*/
template <typename T>
class flexOpticalFlowOperator : public flexDiagonalBasedOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	std::vector<T> image;
	std::vector<int> inputDimension;
	int direction;

public:
	/*!
		\param aImage the image to build the gradient of
		\param AInputDimension vector of dimensions
		\param ADirection direction of the derivative
		\param aIsMinus determines if operator is negated \sa isMinus
	*/
	flexOpticalFlowOperator(const std::vector<T>& aImage, std::vector<int> AInputDimension, int ADirection, bool aIsMinus)
	: flexDiagonalBasedOperator<T>(vectorProduct(AInputDimension), opticalFlowOp, aIsMinus), image(aImage), inputDimension(AInputDimension), direction(ADirection)
	{
		this->updateImage(aImage);
	}

	flexOpticalFlowOperator<T>* copy()
	{
		return new flexOpticalFlowOperator<T>(this->image, this->inputDimension, this->direction, this->isMinus);
	}

	//! returns the derivative of the image (host copy)
	std::vector<T> give_grad()
	{
		return this->getDiagonal();
	}

	void updateImage(const std::vector<T> &imageNew)
	{
		this->image = imageNew;

		Tdata u;
		flexAssign(u, imageNew);

		flexGradientOperator<T> A(inputDimension, direction, central, false);
		A.doTimes(false, u, this->d, EQUALS);
	}
};

#endif
