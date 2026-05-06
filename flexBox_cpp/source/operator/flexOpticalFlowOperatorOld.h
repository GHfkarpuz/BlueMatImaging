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
	std::vector<Tdata> gradImage; //will be our linear operation grad u, so the gradient of the image
	int dim;
	int N;
public:
	//STILL TODO!!!!!!!!!!!!
	//! initializes the concatenation operator for non-CUDA versions
	/*!
		\param image the image to build the gradient of
		\param AInputDimension vector of dimensions
		\param aMinus determines if operator is negated \sa isMinus
	*/
	flexOpticalFlowOperator(std::vector<T> image, std::vector<int> AInputDimension, bool aIsMinus) : image(image), inputDimension(AInputDimension), flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension)*AInputDimension.size(), opticalFlowOp, aIsMinus)
	{
		this->dim = inputDimension.size();
		this->N = vectorProduct(inputDimension);

		//gradients for each direction are necessary
		std::vector<flexGradientOperator<T>> A;
		A.reserve(dim);

		for (int i = 0; i < dim; i++)
		{
			A.emplace_back(AInputDimension, i, central, aIsMinus);
		}

		//flexGradientOperator<T> A(AInputDimension, aDirection, central, aIsMinus);//the algorithm for the optical flow takes the central gradient for the space gradient
		
		this->gradImage.resize(dim);
		
		for(int i = 0; i < dim; i++)
		{
			this->gradImage[i].resize(N);
			A[i].doTimes(false, image, this->gradImage[i], EQUALS);//setting gradImage[i] to be the gradient of u for the i-th direction
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
			\param aIsMinus determines if operator is negated \sa isMinus
		*/
		flexOpticalFlowOperator(std::vector<T> image, std::vector<int> AInputDimension, bool aIsMinus) : image(image), inputDimension(AInputDimension), flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension)*AInputDimension.size(), opticalFlowOp, aIsMinus)
		{

		};
	#endif

	flexOpticalFlowOperator<T>* copy()
	{
		flexOpticalFlowOperator<T>* A = new flexOpticalFlowOperator<T>(image, inputDimension, this->isMinus);//isMinus works here instead of aIsMinus, becuase the base class expects isMinus

		return A;
	}

	std::vector<Tdata> give_grad()
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

            for (int d = 0; d < dim; ++d)
            {
                sum += input[d * N + k] * gradImage[d][k];
            }

            output[k] = sum;
        }
    }

    void timesPlus(bool transposed, const Tdata &input, Tdata &output) override
    {
        for (int k = 0; k < N; ++k)
        {
            T sum = 0;

            for (int d = 0; d < dim; ++d)
            {
                sum += input[d * N + k] * gradImage[d][k];
            }

            output[k] += sum;
        }
    }

    void timesMinus(bool transposed, const Tdata &input, Tdata &output) override
    {
        for (int k = 0; k < N; ++k)
        {
            T sum = 0;

            for (int d = 0; d < dim; ++d)
            {
                sum += input[d * N + k] * gradImage[d][k];
            }

            output[k] -= sum;
        }
    }




	std::vector<T> getAbsRowSum(bool transposed) override
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

	T getMaxRowSumAbs(bool transposed) override
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
	thrust::device_vector<T> getAbsRowSumCUDA(bool transposed) override
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








/*
template <typename T>
class flexOpticalFlowOperator : public flexLinearOperator<T>
{
#ifdef __CUDACC__
    typedef thrust::device_vector<T> Tdata;
#else
    typedef std::vector<T> Tdata;
#endif

private:
    std::vector<int> inputDimension;
    std::vector<Tdata> gradImage; // size = dim, each size = N
    int dim;
    int N;

public:
    flexOpticalFlowOperator(
        std::vector<T> image,
        std::vector<int> AInputDimension,
        bool aIsMinus
    )
    : inputDimension(AInputDimension),
      dim(AInputDimension.size()),
      N(vectorProduct(AInputDimension)),
      flexLinearOperator<T>(N, dim * N, opticalFlowOp, aIsMinus)
    {
        // compute gradients
        std::vector<flexGradientOperator<T>> gradOps(dim);

        for (int d = 0; d < dim; ++d)
            gradOps[d] = flexGradientOperator<T>(AInputDimension, d, central, false);

        gradImage.resize(dim);

        for (int d = 0; d < dim; ++d)
        {
            gradImage[d].resize(N);
            gradOps[d].doTimes(false, image, gradImage[d], EQUALS);
        }
    }

};
*/