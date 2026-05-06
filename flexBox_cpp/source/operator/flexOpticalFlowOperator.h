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
	Tdata gradImage; //will be our linear operation grad u, so the gradient of the image
	int direction;
	int dim;
	int N;
public:
	//STILL TODO!!!!!!!!!!!!
	//! initializes the concatenation operator for non-CUDA versions
	/*!
		\param image the image to build the gradient of
		\param AInputDimension vector of dimensions
		\param ADirection
		\param aMinus determines if operator is negated \sa isMinus
	*/
	flexOpticalFlowOperator(std::vector<T> image, std::vector<int> AInputDimension, int ADirection, bool aIsMinus) : image(image), inputDimension(AInputDimension), direction(ADirection), flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension), opticalFlowOp, aIsMinus)
	{
		this->dim = inputDimension.size();
		this->N = vectorProduct(inputDimension);

		//gradients for each direction are necessary
		flexGradientOperator<T> A(AInputDimension, direction, central, aIsMinus);


		this->gradImage.resize(N);
		A.doTimes(false, image, this->gradImage, EQUALS);//setting gradImage to be the gradient of u for the given direction
		
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
			\param ADirection
			\param aIsMinus determines if operator is negated \sa isMinus
		*/
		flexOpticalFlowOperator(std::vector<T> image, std::vector<int> AInputDimension,  int ADirection, bool aIsMinus) : image(image), inputDimension(AInputDimension), direction(ADirection), flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension), opticalFlowOp, aIsMinus)
		{
			this->dim = inputDimension.size();
			this->N = vectorProduct(inputDimension);

			flexGradientOperator<T> A(AInputDimension, direction, central, aIsMinus);

			this->gradImage.resize(N);
			A.doTimes(false, image, this->gradImage, EQUALS);
		};
	#endif


	flexOpticalFlowOperator<T>* copy()
	{
		flexOpticalFlowOperator<T>* A = new flexOpticalFlowOperator<T>(image, inputDimension, direction, this->isMinus);//isMinus works here instead of aIsMinus, becuase the base class expects isMinus

		return A;
	}

	Tdata give_grad()
	{
		return gradImage;
	}

	//apply linear operator to vector
	void times(bool transposed, const Tdata &input, Tdata &output) override
	{
		this->doTimes(transposed, input, output);
	}

	void doTimes(bool transposed, const Tdata &input, Tdata &output)
    {
        std::fill(output.begin(), output.end(), static_cast<T>(0));
		

        for (int k = 0; k < N; ++k)
        {
            T sum = 0;

            output[k] = input[k]*gradImage[k];
        }
    }

    void timesPlus(bool transposed, const Tdata &input, Tdata &output) override
    {
        for (int k = 0; k < N; ++k)
        {
            T sum = 0;

            output[k] += input[k]*gradImage[k];
        }
    }

    void timesMinus(bool transposed, const Tdata &input, Tdata &output) override
    {
        for (int k = 0; k < N; ++k)
        {
            T sum = 0;

            output[k] -= input[k]*gradImage[k];
        }
    }




	std::vector<T> getAbsRowSum(bool transposed) override
	{
		std::vector<T> result(this->getNumRows(), static_cast<T>(0));

		int N = static_cast<int>(result.size());

		#pragma omp parallel for
		for (int k = 0; k < N; ++k)
		{
			result[k] = std::abs(gradImage[k]);
		}

		return result;
	}

	T getMaxRowSumAbs(bool transposed) override
	{
		T maxVal = static_cast<T>(0);

		int N = static_cast<int>(this->getNumRows());

		#pragma omp parallel for reduction(max:maxVal)
		for (int k = 0; k < N; ++k)
		{
			T val = std::abs(gradImage[k]);
			if (val > maxVal)
				maxVal = val;
		}

		return maxVal;
	}

	#ifdef __CUDACC__
	thrust::device_vector<T> getAbsRowSumCUDA(bool transposed) override
	{
		Tdata tmp = gradImage;
		vectorAbs(tmp);
		return tmp;
	}
	#endif


private:

	void doTimesCPU(const Tdata &input, Tdata &output)
	{
		int N = static_cast<int>(output.size());

		std::fill(output.begin(), output.end(), static_cast<T>(0));

		#pragma omp parallel for
		for (int k = 0; k < N; ++k)
		{
			output[k] = input[k] * gradImage[k];
		}
	}

  	void doTimesCUDA(const Tdata &input, Tdata &output)
	{
	#ifdef __CUDACC__

		int N = static_cast<int>(output.size());

		thrust::fill(output.begin(), output.end(), static_cast<T>(0));

		#pragma omp parallel for
		for (int k = 0; k < N; ++k)
		{
			output[k] = input[k] * gradImage[k];
		}

	#else
		this->doTimesCPU(input, output);
	#endif
	}
};

#endif
