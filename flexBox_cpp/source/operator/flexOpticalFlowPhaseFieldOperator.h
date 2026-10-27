#ifndef flexOpticalFlowPhaseFieldOperator_H
#define flexOpticalFlowPhaseFieldOperator_H

#include <vector>
#include <stdexcept>
#include "flexDiagonalBasedOperator.h"
#include "flexGradientOperator.h"

//! represents s -> (u_t + grad(u)*v) * s, the optical flow residual as a diagonal operator acting on the phase field s
/*!
	d = d_x u * v_0 + d_y u * v_1 + d_t u with central differences in space and forward differences in time.
	Only implemented for 2D + time.
*/
template <typename T>
class flexOpticalFlowPhaseFieldOperator : public flexDiagonalBasedOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	std::vector<T> image;
	std::vector<T> motionField_0;
	std::vector<T> motionField_1;
	std::vector<int> inputDimension;

public:
	/*!
		\param aImage image sequence u
		\param aMotionField_0 motion field in first direction
		\param aMotionField_1 motion field in second direction
		\param aInputDimension vector of dimensions (x, y, t)
		\param aIsMinus determines if operator is negated \sa isMinus
	*/
	flexOpticalFlowPhaseFieldOperator(const std::vector<T>& aImage, const std::vector<T>& aMotionField_0, const std::vector<T>& aMotionField_1, std::vector<int> aInputDimension, bool aIsMinus)
	: flexDiagonalBasedOperator<T>(vectorProduct(aInputDimension), opticalFlowPhaseFieldOp, aIsMinus), image(aImage), motionField_0(aMotionField_0), motionField_1(aMotionField_1), inputDimension(aInputDimension)
	{
		if (aInputDimension.size() != 3)
			throw std::runtime_error("opticalFlowPhaseFieldOp needs inputDimension (x, y, t)");

		const int N = vectorProduct(aInputDimension);
		Tdata u, v0, v1, tmp(N);
		flexAssign(u, aImage);
		flexAssign(v0, aMotionField_0);
		flexAssign(v1, aMotionField_1);

		flexGradientOperator<T> A_x(aInputDimension, 0, central, false);
		flexGradientOperator<T> A_y(aInputDimension, 1, central, false);
		flexGradientOperator<T> A_T(aInputDimension, 2, forward, false);

		A_x.doTimes(false, u, tmp, EQUALS);
		flexMul(this->d, tmp, v0, EQUALS);
		A_y.doTimes(false, u, tmp, EQUALS);
		flexMul(this->d, tmp, v1, PLUS);
		A_T.doTimes(false, u, this->d, PLUS); //d is now u_t + grad_x(u)*v_x + grad_y(u)*v_y
	}

	flexOpticalFlowPhaseFieldOperator<T>* copy()
	{
		return new flexOpticalFlowPhaseFieldOperator<T>(image, motionField_0, motionField_1, inputDimension, this->isMinus);
	}
};

#endif
