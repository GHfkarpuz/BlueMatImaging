#ifndef flexMassPreservationOperator_H
#define flexMassPreservationOperator_H

#include <vector>
#include "flexLinearOperator.h"
#include "flexGradientOperator.h"
//TODO fix a possibly big mathemtical mistake in the transposed operator
//! represents the linear operator for the mass preservation condition (u_t + div(uv)=0 where v is the searched motion
// We will implement it as u'*v_i + u*v_i' for each direction i seperately and using the product rule for derivation
template <typename T>
class flexMassPreservationOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	Tdata image;
	std::vector<int> inputDimension;
	Tdata gradU; //derivative of u
	Tdata gradV; //derivative of v
	flexGradientOperator<T>* A;
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
	flexMassPreservationOperator(std::vector<T> image, std::vector<int> AInputDimension, int ADirection, bool aIsMinus) : image(image), inputDimension(AInputDimension), direction(ADirection), flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension), massPreservationOp, aIsMinus)
	{
		this->dim = inputDimension.size();
		this->N = vectorProduct(inputDimension);
		this->gradV.resize(image.size(), T(0));

		//gradients for each direction are necessary
		A = new flexGradientOperator<T>(AInputDimension, direction, central, aIsMinus);


		this->gradU.resize(N);
		A->doTimes(false, image, this->gradU, EQUALS);//setting gradU to be the derivative of u for the given direction

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
		flexMassPreservationOperator(std::vector<T> image, std::vector<int> AInputDimension,  int ADirection, bool aIsMinus) : image(image), inputDimension(AInputDimension), direction(ADirection), flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension), MassPreservationOp, aIsMinus)
		{
			this->dim = inputDimension.size();
			this->N = vectorProduct(inputDimension);
			this->gradV.resize(image.size(), T(0));
			//gradients for each direction are necessary
			A = new flexGradientOperator<T>(AInputDimension, direction, central, aIsMinus);


			this->gradU.resize(N);
			A->doTimes(false, image, this->gradU, EQUALS);//setting gradU to be the derivative of u for the given direction
		};
	#endif


	flexMassPreservationOperator<T>* copy()
	{
		flexMassPreservationOperator<T>* A = new flexMassPreservationOperator<T>(image, inputDimension, direction, this->isMinus);//isMinus works here instead of aIsMinus, because the base class expects isMinus

		return A;
	}

	Tdata give_gradU()
	{
		return gradU;
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


	#ifdef __CUDACC__
	struct flexMassPreservationFunctor
	{
		mySign s;
		bool transposed;

		__host__ __device__
		flexMassPreservationFunctor(const mySign _s, bool _t)
			: s(_s), transposed(_t) {}

		template <typename Tuple>
		__host__ __device__
		void operator()(Tuple t)
		{
			T& out   = thrust::get<0>(t);
			T const& in   = thrust::get<1>(t);
			T const& gU   = thrust::get<2>(t);
			T const& gV   = thrust::get<3>(t);
			T const& img  = thrust::get<4>(t);

			if (!transposed)
			{

				switch (s)
				{
					case PLUS:
						out += in * gU + gV * img;
						break;
					case MINUS:
						out -= in * gU + gV * img;
						break;
					case EQUALS:
						out = in * gU + gV * img;
						break;
				}
			}
			else
			{
				switch (s)
				{
					case PLUS:
						out += -img * gV;
						break;
					case MINUS:
						out -= -img * gV;
						break;
					case EQUALS:
						out = -img * gV;
						break;
				}
			}
		}
	};
	#endif


	std::vector<T> getAbsRowSum(bool transposed) override
	{
		std::vector<T> result(N, static_cast<T>(0));

		std::vector<T> gradRowSum = A->getAbsRowSum(transposed);
		//std::vector<T> gradRowSum = this->gradV; 
		
		if(!transposed)
		{	
			#pragma omp parallel for
			for (int k = 0; k < N; ++k)
			{
				result[k] = std::abs(gradU[k]) + std::abs(image[k]) * gradRowSum[k];
			}
		}
		else
		{
			#pragma omp parallel for
			for (int k = 0; k < N; ++k)
			{
				result[k] = std::abs(image[k])*gradRowSum[k];
			}
		}

		return result;
	}

	T getMaxRowSumAbs(bool transposed) override
	{
		std::vector<T> rowSum = this->getAbsRowSum(transposed);

		T maxVal = static_cast<T>(0);

		#pragma omp parallel for reduction(max:maxVal)
		for (int k = 0; k < N; ++k)
		{
			if (rowSum[k] > maxVal)
			{
				maxVal = rowSum[k];
			}
		}

		return maxVal;
	}
	
	#ifdef __CUDACC__
	thrust::device_vector<T> getAbsRowSumCUDA(bool transposed) override
	{
		//Just for the moment to test the CPU Version
		getAbsRowSum(transposed);

		if(1==2){
		Tdata result = gradU;
		vectorAbs(result);

		Tdata gradRowSum = A->getAbsRowSumCUDA(transposed);

		Tdata absImage = image;
		vectorAbs(absImage);

		thrust::transform(
			absImage.begin(),
			absImage.end(),
			gradRowSum.begin(),
			gradRowSum.begin(),
			thrust::multiplies<T>()
		);

		thrust::transform(
			result.begin(),
			result.end(),
			gradRowSum.begin(),
			result.begin(),
			thrust::plus<T>()
		);

		return result;
		}
	}
	#endif


private:

	void doTimesCPU(bool transposed, const Tdata &input, Tdata &output, const mySign s)
	{
		A->doTimes(false, input, this->gradV, EQUALS);
		if(!transposed)
		{
			#pragma omp parallel for
			for (int k = 0; k < N; ++k)
			{
				switch (s)
				{
					case PLUS:
					{
						output[k] += input[k]*gradU[k]+gradV[k]*image[k];
						break;
					}
					case MINUS:
					{
						output[k] -= input[k]*gradU[k]+gradV[k]*image[k];
						break;
					}
					case EQUALS:
					{
						output[k] = input[k]*gradU[k]+gradV[k]*image[k];
						break;
					}
				}
			}
		}
		else
		{
			#pragma omp parallel for
			for (int k = 0; k<N; ++k)
			{
				switch (s)
					{
						case PLUS:
						{
							output[k] += -image[k]*gradV[k];
							break;
						}
						case MINUS:
						{
							output[k] -= -image[k]*gradV[k];
							break;
						}
						case EQUALS:
						{
							output[k] = -image[k]*gradV[k];
							break;
						}
					}
			}
		}
	}

  	void doTimes(bool transposed, const Tdata &input, Tdata &output, const mySign s)
	{
	#ifdef __CUDACC__
		A->doTimes(false, input, this->gradV, EQUALS);

		thrust::for_each(
			thrust::make_zip_iterator(
				thrust::make_tuple(
					output.begin(),
					input.begin(),
					gradU.begin(),
					gradV.begin(),
					image.begin()
				)
			),
			thrust::make_zip_iterator(
				thrust::make_tuple(
					output.end(),
					input.end(),
					gradU.end(),
					gradV.end(),
					image.end()
				)
			),
			flexMassPreservationFunctor(s, transposed)
		);

	#else

		doTimesCPU(transposed, input, output, s);

	#endif
	}
};

#endif
