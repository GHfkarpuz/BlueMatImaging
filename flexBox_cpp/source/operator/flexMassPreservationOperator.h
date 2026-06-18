#ifndef flexMassPreservationOperator_H
#define flexMassPreservationOperator_H

#include <vector>
#include "flexLinearOperator.h"
#include "flexGradientOperator.h"

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
	Tdata tmp;
	Tdata tmp2;
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
		tmp.resize(N);
		tmp2.resize(N);

		//gradients for each direction are necessary
		A = new flexGradientOperator<T>(AInputDimension, direction, central, aIsMinus);

		this->gradU.resize(N);
		A->doTimes(false, this->image, this->gradU, EQUALS);//setting gradU to be the derivative of u for the given direction

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
						out += gV + gU * in;
						break;
					case MINUS:
						out -= gV + gU * in;
						break;
					case EQUALS:
						out = gV + gU * in;
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
				result[k] = std::abs(image[k])*gradRowSum[k]+std::abs(gradU[k]);
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
	struct AbsRowSumOp {
		template <typename Tuple>
		__host__ __device__
		void operator()(Tuple t) {
			// tuple: (result, image, gradRowSum)
			thrust::get<0>(t) = std::abs(thrust::get<1>(t)) * thrust::get<2>(t);
		}
	};

	std::vector<T> getAbsRowSumCUDA(bool transposed) // entfernt das 'thrust::device_vector<T>' Missmatch
	{
		// 1. Hole Zeilensumme von A direkt auf der GPU
		// Da A->getAbsRowSumCUDA laut Basisklasse std::vector zurückgibt, 
		// laden wir sie für die Berechnung auf die GPU
		std::vector<T> hostGradRowSum = A->getAbsRowSumCUDA(transposed);
		thrust::device_vector<T> gradRowSum = hostGradRowSum; 

		thrust::device_vector<T> devResult(N);

		// 2. Berechne |image| * gradRowSum direkt auf der GPU via Zip-Iterator
		thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(devResult.begin(), image.begin(), gradRowSum.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(devResult.end(), image.end(), gradRowSum.end())),
			AbsRowSumOp()
		);

		// 3. Kopiere das Ergebnis zurück in einen Standard-Vektor für den CPU-Host
		std::vector<T> hostResult(N);
		thrust::copy(devResult.begin(), devResult.end(), hostResult.begin());

		return hostResult;
	}
	#endif


private:
/*
	void doTimesCPU(bool transposed, const Tdata &input, Tdata &output, const mySign s)
	{
		
		if(!transposed)
		{
			A->doTimes(false, input, this->gradV, EQUALS);
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

			//tmp = u * y
			#pragma omp parallel for
			for(int k=0;k<N;++k)
			{
				tmp[k] = image[k] * input[k];
			}

			//tmp2 = A^T (u y)
			A->doTimes(true, tmp, tmp2, EQUALS);

			#pragma omp parallel for
			for(int k=0;k<N;++k)
			{
				switch(s)
				{
					case PLUS:
					{
						output[k] += tmp2[k] + gradU[k] * input[k];
						break;
					}

					case MINUS:
					{
						output[k] -= tmp2[k] + gradU[k] * input[k];
						break;
					}

					case EQUALS:
					{
						output[k] = tmp2[k] + gradU[k] * input[k];
						break;
					}
				}
			}
		}
	}*/

	void doTimesCPU(bool transposed, const Tdata &input, Tdata &output, const mySign s)
	{
		
		if(!transposed)
		{
			#pragma omp parallel for
			for (int k = 0; k < N; ++k)
			{
				tmp[k]=input[k]*image[k];
			}
			A->doTimes(false, tmp, tmp2, EQUALS);
			#pragma omp parallel for
			for (int k = 0; k < N; ++k)
			{
				switch (s)
				{
					case PLUS:
					{
						output[k] += tmp2[k];
						break;
					}
					case MINUS:
					{
						output[k] -= tmp2[k];
						break;
					}
					case EQUALS:
					{
						output[k] = tmp2[k];
						break;
					}
				}
			}
		}
		else
		{
			//tmp2 = A^T (u y)
			A->doTimes(true, input, tmp, EQUALS);

			#pragma omp parallel for
			for (int k = 0; k < N; ++k)
			{
				tmp[k]*=image[k];
			}

			#pragma omp parallel for
			for(int k=0;k<N;++k)
			{
				switch(s)
				{
					case PLUS:
					{
						output[k] += tmp[k];
						break;
					}

					case MINUS:
					{
						output[k] -= tmp[k];
						break;
					}

					case EQUALS:
					{
						output[k] = tmp[k];
						break;
					}
				}
			}
		}
	}


  	void doTimes(bool transposed, const Tdata &input, Tdata &output, const mySign s)
	{
	#ifdef __CUDACC__
		if (!transposed)
		{
			// tmp = input * image
			thrust::transform(
				input.begin(), input.end(),
				image.begin(),
				tmp.begin(),
				thrust::multiplies<T>()
			);

			// tmp2 = A(tmp)
			A->doTimes(false, tmp, tmp2, EQUALS);

			// output +=/-/= tmp2
			if (s == PLUS) {
				thrust::transform(output.begin(), output.end(), tmp2.begin(), output.begin(), thrust::plus<T>());
			} else if (s == MINUS) {
				thrust::transform(output.begin(), output.end(), tmp2.begin(), output.begin(), thrust::minus<T>());
			} else if (s == EQUALS) {
				thrust::copy(tmp2.begin(), tmp2.end(), output.begin());
			}
		}
		else
		{
			// tmp = A^T(input)
			A->doTimes(true, input, tmp, EQUALS);

			// tmp2 = tmp * image
			thrust::transform(
				tmp.begin(), tmp.end(),
				image.begin(),
				tmp2.begin(),
				thrust::multiplies<T>()
			);

			// output +=/-/= tmp2
			if (s == PLUS) {
				thrust::transform(output.begin(), output.end(), tmp2.begin(), output.begin(), thrust::plus<T>());
			} else if (s == MINUS) {
				thrust::transform(output.begin(), output.end(), tmp2.begin(), output.begin(), thrust::minus<T>());
			} else if (s == EQUALS) {
				thrust::copy(tmp2.begin(), tmp2.end(), output.begin());
			}
		}
	#else
		doTimesCPU(transposed, input, output, s);
	#endif
	}
};

#endif