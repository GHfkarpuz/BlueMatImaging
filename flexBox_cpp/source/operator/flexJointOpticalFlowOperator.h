#ifndef flexJointOpticalFlowOperator_H
#define flexJointOpticalFlowOperator_H

#include <vector>
#include "flexLinearOperator.h"
#include "flexGradientOperator.h"

//! represents the (\partial_t + v*\partial) operator for the first problem in u of the joint motion estimation and image reconstruction using the optical flow condition (u_t + grad(u)*v)=0 where u is the searched motion
// in the times() methods transposed is always set to true, because the adjoint operator is self-adjoint in this case
template <typename T>
class flexJointOpticalFlowOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	Tdata image1;
	Tdata image2;
	Tdata velField;//the velocity field
	std::vector<int> inputDimension;
	//Tdata gradImage; 
	int direction;
	int dim;
	int N;
	T timeStep;
public:
	//STILL TODO!!!!!!!!!!!!
	//! initializes the concatenation operator for non-CUDA versions
	/*!
		\param image1 first image
		\param image1 second image
		\param velField velocity field
		\param AInputDimension vector of dimensions
		\param ADirection
		\param timeStep the time step between the two images
		\param aMinus determines if operator is negated \sa isMinus
	*/
	flexJointOpticalFlowOperator(std::vector<T> image1, std::vector<T> image2, Tdata velField, std::vector<int> AInputDimension, int ADirection, T timeStep, bool aIsMinus) : image1(image1), image2(image2), velField(velField), inputDimension(AInputDimension), direction(ADirection), timeStep(timeStep), flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension), jointOpticalFlowOp, aIsMinus)
	{
		this->dim = inputDimension.size();
		this->N = vectorProduct(inputDimension);

		#ifdef __CUDACC__
			//??

		#else
			//??
		#endif
		
	}

	#ifdef __CUDACC__
		//! initializes the concatenation operator for CUDA versions
		/*!
			\param image1 first image
			\param image2 second image
			\param velField velocity field
			\param AInputDimension vector of dimensions
			\param ADirection
			\param aIsMinus determines if operator is negated \sa isMinus
		*/
		flexJointOpticalFlowOperator(std::vector<T> image1, std::vector<T> image2, Tdata velField, std::vector<int> AInputDimension,  int ADirection, T timeStep, bool aIsMinus) : image1(image1), image2(image2), velField(velField), inputDimension(AInputDimension), direction(ADirection), timeStep(timeStep), flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension), jointOpticalFlowOp, aIsMinus)
		{
			this->dim = inputDimension.size();
			this->N = vectorProduct(inputDimension);

			flexGradientOperator<T> A(AInputDimension, direction, central, aIsMinus);

			this->gradImage.resize(N);
			A.doTimes(false, image, this->gradImage, EQUALS);
		};
	#endif


	flexJointOpticalFlowOperator<T>* copy()
	{
		flexJointOpticalFlowOperator<T>* A = new flexJointOpticalFlowOperator<T>(image, inputDimension, direction, timeStep, this->isMinus);//isMinus works here instead of aIsMinus, becuase the base class expects isMinus

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

		const T invDt = static_cast<T>(1) / this->timeStep;

		#pragma omp parallel for
		for (int k = 0; k < result.size(); ++k)
		{
			result[k] = invDt + std::abs(this->velField[k]);
		}

		return result;
	}

	T getMaxRowSumAbs(bool transposed) override
	{
		Tdata absRowSum(this->getNumRows) = getAbsRowSum(transposed);

		T maxVal = static_cast<T>(0);
		#pragma omp parallel for reduction(max:maxVal)
		for (int k = 0; k < this->getNumRows(); ++k)
		{
			if(absRowSum[k] > maxVal)
				maxVal = absRowSum[k];
		}

		return maxVal;
	}

	#ifdef __CUDACC__
	thrust::device_vector<T> getAbsRowSumCUDA(bool transposed) override
	{
		Tdata result(this->getNumRows());

		const T invDt = static_cast<T>(1) / this->timeStep;

		thrust::transform(
			this->velField.begin(),
			this->velField.end(),
			result.begin(),
			[invDt] __host__ __device__ (T v)
			{
				return invDt + abs(v);
			});

		return result;
	}
	#endif

	#ifdef __CUDACC__
	struct flexJointOpticalFlowOperatorFunctor
	{
		mySign s;
		float timeStep;
		bool transposed;

		__host__ __device__
		flexJointOpticalFlowOperatorFunctor(bool _transposed, mySign _s, float _timeStep): transposed(_transposed), s(_s), timeStep(_timeStep) {}

		template <typename Tuple>
		__host__ __device__
		void operator()(Tuple t)
		{
			if(!transposed)
			{
				// output, image2-image1 term, velocity, gradient
				const auto uT        = thrust::get<1>(t);
				const auto vel       = thrust::get<2>(t);
				const auto gradImage = thrust::get<3>(t);

				const auto value = uT + vel * gradImage;

				switch (s)
				{
					case PLUS:
						thrust::get<0>(t) += value;
						break;

					case MINUS:
						thrust::get<0>(t) -= value;
						break;

					case EQUALS:
						thrust::get<0>(t) = value;
						break;
				}
			}
			else
			{
				// output, input, transposed gradient
				const auto input      = thrust::get<1>(t);
				const auto gradImageT = thrust::get<2>(t);

				const auto value = -input / timeStep - gradImageT;

				switch (s)
				{
					case PLUS:
						thrust::get<0>(t) += value;
						break;

					case MINUS:
						thrust::get<0>(t) -= value;
						break;

					case EQUALS:
						thrust::get<0>(t) = value;
						break;
				}
			}
		}
	};
	#endif


private:

	void doTimesCPU(bool transposed, const Tdata &input, Tdata &output, const mySign s)
    {
		if(!transposed)
		{
			Tdata uT(N);//time derivative
			for (int i = 0; i < N; ++i)
			{
				uT[i] = (this->image2[i]-this->input[i])/this->timeStep;
			}

			Tdata gradImage;
			flexGradientOperator<T> A(AInputDimension, direction, central, aIsMinus);

			A.doTimes(false, input, gradImage, EQUALS);

			#pragma omp parallel for
			for (int k = 0; k < N; ++k)
			{
				switch (s)
				{
					case PLUS:
					{
						output[k] += uT[k] + this->velField[k]*gradImage[k];
						break;
					}
					case MINUS:
					{
						output[k] -= uT[k] + this->velField[k]*gradImage[k];
						break;
					}
					case EQUALS:
					{
						output[k] = uT[k] + this->velField[k]*gradImage[k];
						break;
					}
				}
			}
		}
		else
		{
			Tdata weightedInput(N);

			#pragma omp parallel for
			for (int k = 0; k < N; ++k)
			{
				weightedInput[k] = this->velField[k] * input[k];
			}

			Tdata gradImageT(N);

			flexGradientOperator<T> A(AInputDimension, direction, central, aIsMinus);

			// transposed gradient
			A.doTimes(true, weightedInput, gradImageT, EQUALS);

			#pragma omp parallel for
			for (int k = 0; k < N; ++k)
			{

				switch (s)
				{
					case PLUS:
					{
						output[k] += - (input[k] / this->timeStep) - gradImageT[k];;
						break;
					}

					case MINUS:
					{
						output[k] -= - (input[k] / this->timeStep) - gradImageT[k];;
						break;
					}

					case EQUALS:
					{
						output[k] = - (input[k] / this->timeStep) - gradImageT[k];;
						break;
					}
				}
			}
		}
    }



	void doTimes(bool transposed, const Tdata &input, Tdata &output, const mySign s)
	{
	#ifdef __CUDACC__
		flexGradientOperator<T> A(AInputDimension, direction, central, aIsMinus);

		if(!transposed)
		{
			// u_t = (image2 - input)/dt
			Tdata uT(N);

			thrust::transform(
				this->image2.begin(),
				this->image2.end(),
				input.begin(),
				uT.begin(),
				[this] __host__ __device__ (T a, T b)
				{
					return (a - b) / this->timeStep;
				});

			// grad(input)
			Tdata gradImage(N);
			A.doTimes(false, input, gradImage, EQUALS);

			thrust::for_each(
				thrust::make_zip_iterator(
					thrust::make_tuple(
						output.begin(),
						uT.begin(),
						this->velField.begin(),
						gradImage.begin())),

				thrust::make_zip_iterator(
					thrust::make_tuple(
						output.end(),
						uT.end(),
						this->velField.end(),
						gradImage.end())),

				flexJointOpticalFlowOperatorFunctor(
					false,
					s,
					this->timeStep));
		}
		else
		{
			// weightedInput = velField * input
			Tdata weightedInput(N);

			thrust::transform(
				this->velField.begin(),
				this->velField.end(),
				input.begin(),
				weightedInput.begin(),
				thrust::multiplies<T>());

			// gradTransposed(weightedInput)
			Tdata gradImageT(N);

			A.doTimes(true, weightedInput, gradImageT, EQUALS);

			thrust::for_each(
				thrust::make_zip_iterator(
					thrust::make_tuple(
						output.begin(),
						input.begin(),
						gradImageT.begin())),

				thrust::make_zip_iterator(
					thrust::make_tuple(
						output.end(),
						input.end(),
						gradImageT.end())),

				flexJointOpticalFlowOperatorFunctor(
					true,
					s,
					this->timeStep));
		}

	#else
		this->doTimesCPU(transposed, input, output, s);
	#endif
	}
};

#endif
