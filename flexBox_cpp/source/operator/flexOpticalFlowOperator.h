#ifndef flexOpticalFlowOperator_H
#define flexOpticalFlowOperator_H

#include <vector>
#include "flexLinearOperator.h"
#include "flexGradientOperator.h"

//! represents the grad(u)/cdot operator for the optical flow condition (u_t + grad(u)*v)=0 where v is the searched motion
// in the times() methods transposed is always set to true, because the adjoint operator is self-adjoint in this case
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

		for (int k = 0; k < N; ++k)
		{
			this->image[k] *= 1.0 / 255.0;
		}

		//gradients for each direction are necessary
		flexGradientOperator<T> A(AInputDimension, direction, central, aIsMinus);


		this->gradImage.resize(N);
		A.doTimes(false, this->image, this->gradImage, EQUALS);//setting gradImage to be the gradient of u for the given direction

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

			for (int k = 0; k < N; ++k)
		{
			this->image[k] *= 1.0 / 255.0;
		}

		//gradients for each direction are necessary
		flexGradientOperator<T> A(AInputDimension, direction, central, aIsMinus);


		this->gradImage.resize(N);
		A.doTimes(false, this->image, this->gradImage, EQUALS);//setting gradImage to be the gradient of u for the given direction
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
	void times(bool transposed, const Tdata &input, Tdata &output)
	{
		this->doTimes(transposed, input,output,EQUALS);
	}

	void timesPlus(bool transposed, const Tdata &input, Tdata &output)
	{
        if (this->isMinus)
        {
            this->doTimes(transposed, input,output, MINUS);
        }
        else
        {
            this->doTimes(transposed, input,output, PLUS);
        }
	}

	void timesMinus(bool transposed, const Tdata &input, Tdata &output)
	{
        if (this->isMinus)
        {
            this->doTimes(transposed, input,output, PLUS);
        }
        else
        {
            this->doTimes(transposed, input,output, MINUS);
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

	#ifdef __CUDACC__
    struct flexOpticalFlowOperatorFunctor
	{
		mySign s;
		bool transposed;

		__host__ __device__
		flexOpticalFlowOperatorFunctor(const mySign _s, bool _t) : s(_s), transposed(_t){}

		template <typename Tuple>
		__host__ __device__
		void operator()(Tuple t)
		{
			switch (this->s)
			{
				case PLUS:
				{
					thrust::get<0>(t) += thrust::get<1>(t) * thrust::get<2>(t);
					break;
				}
				case MINUS:
				{
					thrust::get<0>(t) -= thrust::get<1>(t) * thrust::get<2>(t);
					break;
				}
				case EQUALS:
				{
					thrust::get<0>(t) = thrust::get<1>(t) * thrust::get<2>(t);
					break;
				}
			}
		}
	};
    #endif


private:

	void doTimesCPU(bool transposed, const Tdata &input, Tdata &output, const mySign s)
    {
		#pragma omp parallel for
		for (int k = 0; k < N; ++k)
		{
			switch (s)
			{
				case PLUS:
				{
					output[k] += input[k]*gradImage[k];
					break;
				}
				case MINUS:
				{
					output[k] -= input[k]*gradImage[k];
					break;
				}
				case EQUALS:
				{
					output[k] = input[k]*gradImage[k];
					break;
				}
			}
		}
    }



	void doTimes(bool transposed, const Tdata &input, Tdata &output,const mySign s)
	{
        #ifdef __CUDACC__
		{
			thrust::for_each(
				thrust::make_zip_iterator(thrust::make_tuple(output.begin(), input.begin(), this->gradImage.begin())),
				thrust::make_zip_iterator(thrust::make_tuple(output.end(),   input.end(),   this->gradImage.end())),
			flexOpticalFlowOperatorFunctor(s, transposed));
		}
        #else
            this->doTimesCPU(transposed,input,output,s);
        #endif
  }
};

#endif
