#ifndef flexOpticalFlowOperator_H
#define flexOpticalFlowOperator_H

#include <vector>
#include "flexLinearOperator.h"
#include "flexGradientOperator.h"

//! represents the grad(u)/cdot operator for the optical flow condition (u_t + grad(u)*v)=0 where v is the searched motion 
template <typename T>
class flexOpticalFlowOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	Tdata image;
	std::vector<int> inputDimension;
	int direction;
	std::vector<Tdata> gradImage; //will be our linear operation grad u, so the gradient of the image
public:
	//STILL TODO!!!!!!!!!!!!
	//! initializes the concatenation operator for non-CUDA versions
	/*!
		\param image the image to build the gradient of
		\param AInputDimension vector of dimensions
		\param aDirection gives the direction of the motion field that gets calculated. It starts with 0
		\param aMinus determines if operator is negated \sa isMinus
	*/
	flexOpticalFlowOperator(std::vector<T> image, std::vector<int> AInputDimension, int aDirection, bool aIsMinus) : image(image), inputDimension(AInputDimension), direction(aDirection), flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension)*AInputDimension.size(), opticalFlowOp, aIsMinus)
	{
		//gradients for each direction are necessary
		std::vector<flexGradientOperator<T>> A(inputDimension.size());

		for(int i = 0; i < inputDimension.size(); i++)
		{
			A[i] = flexGradientOperator<T>(AInputDimension, i, central, aIsMinus);
		}

		//flexGradientOperator<T> A(AInputDimension, aDirection, central, aIsMinus);//the algorithm for the optical flow takes the central gradient for the space gradient
		
		this->gradImage.resize(inputDimension.size());
		
		for(int i = 0; i < inputDimension.size(); i++)
		{
			this->gradImage[i].resize(vectorProduct(AInputDimension));
			A[i].doTimes(false, image, this->gradImage[i], EQUALS);
		}

		//A.doTimes(false, image, gradImage, EQUALS); //now gradImage is grad(u). The multiplication with gradImage will be our linear operator in this class
		

		#ifdef __CUDACC__
			//??

		#else
			//??
		#endif
	}

	#ifdef __CUDACC__
		//! initializes the concatenation operator for CUDA versions
		/*!
			\param image the image to build the gradient of
			\param AInputDimension vector of dimensions
			\param aDirection gives the direction of the motion field that gets calculated. It starts with 0
			\param aIsMinus determines if operator is negated \sa isMinus
		*/
		flexOpticalFlowOperator(std::vector<T> image, std::vector<int> AInputDimension, int aDirection, bool aIsMinus) : image(image), inputDimension(AInputDimension), direction(aDirection), flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension)*AInputDimension.size(), opticalFlowOp, aIsMinus)
		{

		};
	#endif

	flexOpticalFlowOperator<T>* copy()
	{
		flexOpticalFlowOperator<T>* A = new flexOpticalFlowOperator<T>(image, inputDimension, direction, this->aIsMinus);

		return A;
	}

	std::vector<Tdata> give_grad()
	{
		return gradImage;
	}

	//apply linear operator to vector
	void times(bool transposed, const std::vector<Tdata> &input, Tdata &output)
	{
		this->doTimes(input,output);
	}


	std::vector<T> getAbsRowSum(bool transposed)
	{
		std::vector<T> result(this->getNumRows(), static_cast<T>(0));

		int dim = static_cast<int>(gradImage.size());
		int N   = static_cast<int>(result.size());

		#pragma omp parallel for
		for (int k = 0; k < N; ++k)
		{
			T sum = static_cast<T>(0);

			for (int i = 0; i < dim; ++i)
			{
				sum += std::abs(gradImage[i][k]);
			}

			result[k] = sum;
		}

		return result;
	}

	T getMaxRowSumAbs(bool transposed)
	{
		T maxVal = static_cast<T>(0);

		int dim = static_cast<int>(gradImage.size());
		int N   = static_cast<int>(this->getNumRows());

		#pragma omp parallel for reduction(max:maxVal)
		for (int k = 0; k < N; ++k)
		{
			T sum = static_cast<T>(0);

			for (int i = 0; i < dim; ++i)
			{
				sum += std::abs(gradImage[i][k]);
			}

			if (sum > maxVal)
				maxVal = sum;
		}

		return maxVal;
	}

	#ifdef __CUDACC__
	thrust::device_vector<T> getAbsRowSumCUDA(bool transposed)
	{
		Tdata tmp = this->gradImage;
		vectorAbs(tmp);
		return tmp;
	}
	#endif


private:
	//input is a vector of dimension inputDimension. The i-th vector is the velocity field to the i-th direction. typically 2 or 3 directions.
	//output holds sum_{i=1}^{inputDimension} (i-th spatial derivative of u)*(motion field for i-th direction)
	void doTimesCPU(const std::vector<Tdata> &input, Tdata &output)
	{
        int numElements = (int)output.size();//size is number of pixels or voxels respectively
		std::fill(output.begin(), output.end(), static_cast<T>(0));

		#pragma omp parallel for
		for (int j = 0; j < numElements; ++j)
		{
			T sum = static_cast<T>(0);

			for (int i = 0; i < gradImage.size(); ++i)
			{
				sum += input[i][j] * gradImage[i][j];
			}

			output[j] = sum;
		}

  	}

  	void doTimesCUDA(const std::vector<Tdata> &input, Tdata &output)
	{
	#ifdef __CUDACC__

		int numElements = static_cast<int>(output.size());
		int dim = static_cast<int>(gradImage.size());

		// output = 0
		thrust::fill(output.begin(), output.end(), static_cast<T>(0));

		// for each pixel
		#pragma omp parallel for
		for (int k = 0; k < numElements; ++k)
		{
			T sum = static_cast<T>(0);

			for (int i = 0; i < dim; ++i)
			{
				sum += input[i][k] * gradImage[i][k];
			}

			output[k] = sum;
		}

	#else
		this->doTimesCPU(input, output);
	#endif
	}
};

#endif