#ifndef flexUpwindGradientOperator_H
#define flexUpwindGradientOperator_H

#include <vector>
#include "flexLinearOperator.h"

//! represents a gradient operator
template<typename T>
class flexUpwindGradientOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	std::vector<int> inputDimension;
	int gradDirection;
	Tdata motionField;
	int numberDimensions;

public:

	//! initializes the gradient operator
	/*!
		\param AInputDimension vector of dimensions
		\param aGradDirection direction of gradient. 0 for first dimension and so on.
		\param aMotionField motion field in the direction aGradDirection
		\param aMinus determines if operator is negated \sa isMinus
	*/
	flexUpwindGradientOperator(std::vector<int> AInputDimension, int aGradDirection, Tdata aMotionField, bool aMinus) :
		inputDimension(AInputDimension),
		gradDirection(aGradDirection),
		motionField(aMotionField),
		numberDimensions(static_cast<int>(AInputDimension.size())), flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension), upwindGradientOp, aMinus)
	{};

	~flexUpwindGradientOperator()
	{
		/*
		//gradients for each direction are necessary
		flexUpwindGradientOperator<T> A = flexUpwindGradientOperator<T>(inputDimension, gradDirection, motionField, this->aMinus);

		int N = vectorProduct(inputDimension);
		this->gradU.resize(N);
		A.doTimes(false, this->image, this->gradU, EQUALS);//setting gradU to be the derivative of u for the given direction

		Tdata x(N), y(N);

		for(int i=0;i<N;i++){
			x[i] = 0.1 * i;
			y[i] = std::sin(i * 0.01);
		}


		Tdata Ax(N,0), ATy(N,0);

		// forward
		A.doTimes(false, x, Ax, EQUALS);

		// adjoint
		A.doTimes(true, y, ATy, EQUALS);

		// inner products
		T lhs = 0;
		T rhs = 0;

		for(int i=0;i<N;i++){
			lhs += Ax[i] * y[i];
			rhs += x[i] * ATy[i];
		}

		std::cout << "For the upwind gradient operator, we have " << std::endl;
		std::cout << "LHS: " << lhs << std::endl;
		std::cout << "RHS: " << rhs << std::endl;
		std::cout << "relative error: "
				<< std::abs(lhs-rhs)/(std::abs(lhs)+1e-12)
				<< std::endl;

		delete A;
		*/
	}

	flexUpwindGradientOperator<T>* copy()
	{
		std::vector<int> dimsCopy;
		dimsCopy.resize(this->inputDimension.size());

		std::copy(this->inputDimension.begin(), this->inputDimension.end(), dimsCopy.begin());

		return new flexUpwindGradientOperator<T>(dimsCopy, this->gradDirection, this->motionField, this->isMinus);
	}

	// 2D cases
	void dxUpwind2d(const Tdata &input, Tdata &output, mySign s)
	{
		int sizeY = this->inputDimension[1];
		int sizeX = this->inputDimension[0];
		std::fill(output.begin(), output.end(), 0.0);
		#pragma omp parallel for
		for (int j = 0; j < sizeY; ++j)
		{
			for (int i = 1; i < sizeX-1; ++i)
			{
				int tmpIndex = index2DtoLinear(i,j);
				int tmpIndex1 = index2DtoLinear(i+1,j);
				int tmpIndex2 = index2DtoLinear(i-1,j);

				if(motionField[tmpIndex]<0)
				{
					output[tmpIndex] = motionField[tmpIndex]*(input[tmpIndex1]-input[tmpIndex]);
				}
				else
				{
					output[tmpIndex] = motionField[tmpIndex]*(input[tmpIndex]-input[tmpIndex2]);
				}
			}
			
			int tmpIndex_edge1 = index2DtoLinear(0,j);

			if(motionField[tmpIndex_edge1]<0)
			{
				int tmpIndex_edge2 = index2DtoLinear(1,j);

				output[tmpIndex_edge1] = motionField[tmpIndex_edge1]*(input[tmpIndex_edge2]-input[tmpIndex_edge1]);
			}
			else
			{
				output[tmpIndex_edge1] = 0.0f;
			}


			int tmpIndex_edge3 = index2DtoLinear(sizeX-1,j);

			if(motionField[tmpIndex_edge3]>0)
			{
				int tmpIndex_edge4 = index2DtoLinear(sizeX - 2,j);

				output[tmpIndex_edge3] = motionField[tmpIndex_edge3]*(input[tmpIndex_edge3]-input[tmpIndex_edge4]);
			}
			else
			{
				output[tmpIndex_edge3] = 0.0f;
			}
			
		}
	}

	void dyUpwind2d(const Tdata &input, Tdata &output, mySign s)
	{
		int sizeY = this->inputDimension[1];
		int sizeX = this->inputDimension[0];
		std::fill(output.begin(), output.end(), 0.0);
		#pragma omp parallel for
		for (int i = 0; i < sizeX; ++i)
		{
			for (int j = 1; j < sizeY - 1; ++j)
			{
				int tmpIndex  = index2DtoLinear(i,j);
				int tmpIndex1 = index2DtoLinear(i,j+1);
				int tmpIndex2 = index2DtoLinear(i,j-1);

				if (motionField[tmpIndex] < 0)
				{
					output[tmpIndex] = motionField[tmpIndex] * (input[tmpIndex1] - input[tmpIndex]);
				}
				else
				{
					output[tmpIndex] = motionField[tmpIndex] * (input[tmpIndex] - input[tmpIndex2]);
				}
			}
			
			int tmpIndex_edge1 = index2DtoLinear(i,0);

			if (motionField[tmpIndex_edge1] < 0)
			{
				int tmpIndex_edge2 = index2DtoLinear(i,1);

				output[tmpIndex_edge1] = motionField[tmpIndex_edge1] * (input[tmpIndex_edge2] - input[tmpIndex_edge1]);
			}
			else
			{
				output[tmpIndex_edge1] = 0.0f;
			}
			
			int tmpIndex_edge3 = index2DtoLinear(i, sizeY-1);

			if (motionField[tmpIndex_edge3] > 0)
			{
				int tmpIndex_edge4 = index2DtoLinear(i,sizeY-2);

				output[tmpIndex_edge3] = motionField[tmpIndex_edge3] * (input[tmpIndex_edge3] - input[tmpIndex_edge4]);
			}
			else
			{
				output[tmpIndex_edge3] = 0.0f;
			}
		}
	}

	
	void dxUpwind2dT(const Tdata &input, Tdata &output, mySign s)
	{
		int sizeX = this->inputDimension[0];
		int sizeY = this->inputDimension[1];

		std::fill(output.begin(), output.end(), 0.0);

		for (int j = 0; j < sizeY; ++j)
		{
			for (int i = 2; i < sizeX-2; ++i)
			{
				int tmpIndex = index2DtoLinear(i, j);
				int tmpIndex1 = index2DtoLinear(i+1, j);
				int tmpIndex2 = index2DtoLinear(i-1, j);

				/*
				int motionFieldIsNegative = std::max(motionFieldField[tmpIndex], 0);
				int motionFieldIsNegative1 = std::max(motionFieldField[tmpIndex1], 0);
				int motionFieldIsNegative2 = std::max(motionFieldField[tmpIndex2], 0);

				output[tmpIndex] = -motionFieldIsNegative * motionField[tmpIndex]*input[tmpIndex]*/
				if (motionField[tmpIndex] < 0)
				{
					output[tmpIndex] -= motionField[tmpIndex]*input[tmpIndex];
				}
				else
				{
					output[tmpIndex] += motionField[tmpIndex]*input[tmpIndex];
				}


				if (motionField[tmpIndex1] > 0)
				{
					output[tmpIndex] -= motionField[tmpIndex1]*input[tmpIndex1];
				}


				if (motionField[tmpIndex2] < 0)
				{
					output[tmpIndex] += motionField[tmpIndex2]*input[tmpIndex2];
				}
			}

			int tmpIndex_edge1 = index2DtoLinear(0,j);
			int tmpIndex_edge2 = index2DtoLinear(1,j);
			int tmpIndex_edge3 = index2DtoLinear(2,j);

			if (motionField[tmpIndex_edge1] < 0)
			{
				output[tmpIndex_edge1] -= motionField[tmpIndex_edge1] * input[tmpIndex_edge1];
			}
			if (motionField[tmpIndex_edge2] > 0)
			{
				output[tmpIndex_edge1] -= motionField[tmpIndex_edge2] * input[tmpIndex_edge2];
			}


			if (motionField[tmpIndex_edge1] < 0)
			{
				output[tmpIndex_edge2] += motionField[tmpIndex_edge1] * input[tmpIndex_edge1];
			}
			if (motionField[tmpIndex_edge2] < 0)
			{
				output[tmpIndex_edge2] -= motionField[tmpIndex_edge2] * input[tmpIndex_edge2];
			}
			else
			{
				output[tmpIndex_edge2] += motionField[tmpIndex_edge2] * input[tmpIndex_edge2];
			}
			if (motionField[tmpIndex_edge3] > 0)
			{
				output[tmpIndex_edge2] -= motionField[tmpIndex_edge3] * input[tmpIndex_edge3];
			}

			int tmpIndex_edge4 = index2DtoLinear(sizeX-1, j);
			int tmpIndex_edge5 = index2DtoLinear(sizeX-2, j);
			int tmpIndex_edge6 = index2DtoLinear(sizeX-3, j);

			if (motionField[tmpIndex_edge6] < 0)
			{
				output[tmpIndex_edge5] += motionField[tmpIndex_edge6] * input[tmpIndex_edge6];
			}
			if (motionField[tmpIndex_edge5] < 0)
			{
				output[tmpIndex_edge5] -= motionField[tmpIndex_edge5] * input[tmpIndex_edge5];
			}
			else
			{
				output[tmpIndex_edge5] += motionField[tmpIndex_edge5] * input[tmpIndex_edge5];
			}
			if (motionField[tmpIndex_edge4] > 0)
			{
				output[tmpIndex_edge5] -= motionField[tmpIndex_edge4] * input[tmpIndex_edge4];
			}


			if (motionField[tmpIndex_edge5] < 0)
			{
				output[tmpIndex_edge4] += motionField[tmpIndex_edge5] * input[tmpIndex_edge5];
			}
			if (motionField[tmpIndex_edge4] > 0)
			{
				output[tmpIndex_edge4] += motionField[tmpIndex_edge4] * input[tmpIndex_edge4];
			}
		}
	}


	void dyUpwind2dT(const Tdata &input, Tdata &output, mySign s)
	{
		int sizeX = this->inputDimension[0];
		int sizeY = this->inputDimension[1];

		std::fill(output.begin(), output.end(), 0.0);

		for (int i = 0; i < sizeX; ++i)
		{
			for (int j = 2; j < sizeY-2; ++j)
			{
				int tmpIndex = index2DtoLinear(i, j);
				int tmpIndex1 = index2DtoLinear(i, j+1);
				int tmpIndex2 = index2DtoLinear(i, j-1);

				if (motionField[tmpIndex] < 0)
				{
					output[tmpIndex] -= motionField[tmpIndex]*input[tmpIndex];
				}
				else
				{
					output[tmpIndex] += motionField[tmpIndex]*input[tmpIndex];
				}


				if (motionField[tmpIndex1] > 0)
				{
					output[tmpIndex] -= motionField[tmpIndex1]*input[tmpIndex1];
				}


				if (motionField[tmpIndex2] < 0)
				{
					output[tmpIndex] += motionField[tmpIndex2]*input[tmpIndex2];
				}
			}
			

			
			int tmpIndex_edge1 = index2DtoLinear(i,0);
			int tmpIndex_edge2 = index2DtoLinear(i,1);
			int tmpIndex_edge3 = index2DtoLinear(i,2);

			if (motionField[tmpIndex_edge1] < 0)
			{
				output[tmpIndex_edge1] -= motionField[tmpIndex_edge1] * input[tmpIndex_edge1];
			}
			if (motionField[tmpIndex_edge2] > 0)
			{
				output[tmpIndex_edge1] -= motionField[tmpIndex_edge2] * input[tmpIndex_edge2];
			}


			if (motionField[tmpIndex_edge1] < 0)
			{
				output[tmpIndex_edge2] += motionField[tmpIndex_edge1] * input[tmpIndex_edge1];
			}
			if (motionField[tmpIndex_edge2] < 0)
			{
				output[tmpIndex_edge2] -= motionField[tmpIndex_edge2] * input[tmpIndex_edge2];
			}
			else
			{
				output[tmpIndex_edge2] += motionField[tmpIndex_edge2] * input[tmpIndex_edge2];
			}
			if (motionField[tmpIndex_edge3] > 0)
			{
				output[tmpIndex_edge2] -= motionField[tmpIndex_edge3] * input[tmpIndex_edge3];
			}

			int tmpIndex_edge4 = index2DtoLinear(i,sizeY-1);
			int tmpIndex_edge5 = index2DtoLinear(i,sizeY-2);
			int tmpIndex_edge6 = index2DtoLinear(i,sizeY-3);

			if (motionField[tmpIndex_edge6] < 0)
			{
				output[tmpIndex_edge5] += motionField[tmpIndex_edge6] * input[tmpIndex_edge6];
			}
			if (motionField[tmpIndex_edge5] < 0)
			{
				output[tmpIndex_edge5] -= motionField[tmpIndex_edge5] * input[tmpIndex_edge5];
			}
			else
			{
				output[tmpIndex_edge5] += motionField[tmpIndex_edge5] * input[tmpIndex_edge5];
			}
			if (motionField[tmpIndex_edge4] > 0)
			{
				output[tmpIndex_edge5] -= motionField[tmpIndex_edge4] * input[tmpIndex_edge4];
			}


			if (motionField[tmpIndex_edge5] < 0)
			{
				output[tmpIndex_edge4] += motionField[tmpIndex_edge5] * input[tmpIndex_edge5];
			}
			if (motionField[tmpIndex_edge4] > 0)
			{
				output[tmpIndex_edge4] += motionField[tmpIndex_edge4] * input[tmpIndex_edge4];
			}
		}
	}

	// 3D cases, TODO: extend an correct them using the 2D cases as a template
	void dxUpwind3d(const Tdata &input, Tdata &output, mySign s)
	{
		int sizeX = this->inputDimension[0];
		int sizeY = this->inputDimension[1];
		int sizeZ = this->inputDimension[2];
		std::fill(output.begin(), output.end(), 0.0);
		#pragma omp parallel for collapse(2)
		for (int k = 0; k < sizeZ; ++k)
		{
			for (int j = 0; j < sizeY; ++j)
			{
				for (int i = 1; i < sizeX - 1; ++i)
				{
					int idx  = index3DtoLinear(i,   j, k);
					int idxP = index3DtoLinear(i+1, j, k);
					int idxM = index3DtoLinear(i-1, j, k);

					if (motionField[idx] < 0)
					{
						output[idx] = motionField[idx] * (input[idxP] - input[idx]);
					}
					else
					{
						output[idx] = motionField[idx] * (input[idx] - input[idxM]);
					}
				}

				{
					int idx0 = index3DtoLinear(0, j, k);

					if (motionField[idx0] < 0)
					{
						int idx1 = index3DtoLinear(1, j, k);

						output[idx0] = motionField[idx0] * (input[idx1] - input[idx0]);
					}
					else
					{
						output[idx0] = 0.0f;
					}
				}

				{
					int idxN = index3DtoLinear(sizeX - 1, j, k);

					if (motionField[idxN] > 0)
					{
						int idxNm1 = index3DtoLinear(sizeX - 2, j, k);

						output[idxN] = motionField[idxN] * (input[idxN] - input[idxNm1]);
					}
					else
					{
						output[idxN] = 0.0f;
					}
				}
			}
		}
	}

	void dyUpwind3d(const Tdata &input, Tdata &output, mySign s)
	{
		int sizeX = this->inputDimension[0];
		int sizeY = this->inputDimension[1];
		int sizeZ = this->inputDimension[2];
		std::fill(output.begin(), output.end(), 0.0);
		#pragma omp parallel for collapse(2)
		for (int k = 0; k < sizeZ; ++k)
		{
			for (int i = 0; i < sizeX; ++i)
			{
				for (int j = 1; j < sizeY - 1; ++j)
				{
					int idx  = index3DtoLinear(i, j,   k);
					int idxP = index3DtoLinear(i, j+1, k);
					int idxM = index3DtoLinear(i, j-1, k);

					if (motionField[idx] < 0)
					{
						output[idx] = motionField[idx] * (input[idxP] - input[idx]);
					}
					else
					{
						output[idx] = motionField[idx] * (input[idx] - input[idxM]);
					}
				}

				{
					int idx0 = index3DtoLinear(i, 0, k);

					if (motionField[idx0] < 0)
					{
						int idx1 = index3DtoLinear(i, 1, k);

						output[idx0] = motionField[idx0] * (input[idx1] - input[idx0]);
					}
					else
					{
						output[idx0] = 0.0f;
					}
				}

				{
					int idxN = index3DtoLinear(i, sizeY-1, k);

					if (motionField[idxN] > 0)
					{
						int idxNm1 = index3DtoLinear(i, sizeY-2, k);

						output[idxN] = motionField[idxN] * (input[idxN] - input[idxNm1]);
					}
					else
					{
						output[idxN] = 0.0f;
					}
				}
			}
		}
	}

	void dzUpwind3d(const Tdata &input, Tdata &output, mySign s)
	{
		int sizeX = this->inputDimension[0];
		int sizeY = this->inputDimension[1];
		int sizeZ = this->inputDimension[2];
		std::fill(output.begin(), output.end(), 0.0);
		#pragma omp parallel for collapse(2)
		for (int j = 0; j < sizeY; ++j)
		{
			for (int i = 0; i < sizeX; ++i)
			{
				for (int k = 1; k < sizeZ - 1; ++k)
				{
					int idx  = index3DtoLinear(i, j, k);
					int idxP = index3DtoLinear(i, j, k+1);
					int idxM = index3DtoLinear(i, j, k-1);

					if (motionField[idx] < 0)
					{
						output[idx] = motionField[idx] * (input[idxP] - input[idx]);
					}
					else
					{
						output[idx] = motionField[idx] * (input[idx] - input[idxM]);
					}
				}

				{
					int idx0 = index3DtoLinear(i, j, 0);

					if (motionField[idx0] < 0)
					{
						int idx1 = index3DtoLinear(i, j, 1);

						output[idx0] = motionField[idx0] * (input[idx1] - input[idx0]);
					}
					else
					{
						output[idx0] = 0.0f;
					}
				}

				{
					int idxN = index3DtoLinear(i, j, sizeZ-1);

					if (motionField[idxN] > 0)
					{
						int idxNm1 = index3DtoLinear(i, j, sizeZ-2);

						output[idxN] = motionField[idxN] * (input[idxN] - input[idxNm1]);
					}
					else
					{
						output[idxN] = 0.0f;
					}
				}
			}
		}
	}


	void dxUpwind3dT(const Tdata &input, Tdata &output, mySign s)
	{
		int sizeX = this->inputDimension[0];
		int sizeY = this->inputDimension[1];
		int sizeZ = this->inputDimension[2];
		std::fill(output.begin(), output.end(), 0.0);
		#pragma omp parallel for collapse(2)
		for (int k = 0; k < sizeZ; ++k)
		{
			for (int j = 0; j < sizeY; ++j)
			{
				for (int i = 1; i < sizeX; ++i)
				{
					int idx = index3DtoLinear(i, j, k);

					float v = motionField[idx];

					if (v < 0)
					{
						output[idx]     += -v * input[idx];

						if (i > 0)
						{
							int idxM = index3DtoLinear(i - 1, j, k);
							output[idxM] += v * input[idx];
						}
					}
					else
					{
						output[idx] += v * input[idx];

						if (i < sizeX - 1)
						{
							int idxP = index3DtoLinear(i + 1, j, k);
							output[idxP] += -v * input[idx];
						}
					}
				}
			}
		}
	}



	void dyUpwind3dT(const Tdata &input, Tdata &output, mySign s)
	{
		int sizeX = this->inputDimension[0];
		int sizeY = this->inputDimension[1];
		int sizeZ = this->inputDimension[2];
		std::fill(output.begin(), output.end(), 0.0);
		#pragma omp parallel for collapse(2)
		for (int k = 0; k < sizeZ; ++k)
		{
			for (int i = 0; i < sizeX; ++i)
			{
				for (int j = 1; j < sizeY; ++j)
				{
					int idx = index3DtoLinear(i, j, k);
					float v = motionField[idx];

					if (v < 0)
					{
						output[idx] += -v * input[idx];

						if (j > 0)
						{
							int idxM = index3DtoLinear(i, j - 1, k);
							output[idxM] += v * input[idx];
						}
					}
					else
					{
						output[idx] += v * input[idx];

						if (j < sizeY - 1)
						{
							int idxP = index3DtoLinear(i, j + 1, k);
							output[idxP] += -v * input[idx];
						}
					}
				}
			}
		}
	}


	void dzUpwind3dT(const Tdata &input, Tdata &output, mySign s)
	{
		int sizeX = this->inputDimension[0];
		int sizeY = this->inputDimension[1];
		int sizeZ = this->inputDimension[2];
		std::fill(output.begin(), output.end(), 0.0);
		#pragma omp parallel for collapse(2)
		for (int j = 0; j < sizeY; ++j)
		{
			for (int i = 0; i < sizeX; ++i)
			{
				for (int k = 1; k < sizeZ; ++k)
				{
					int idx = index3DtoLinear(i, j, k);
					float v = motionField[idx];

					if (v < 0)
					{
						output[idx] += -v * input[idx];

						if (k > 0)
						{
							int idxM = index3DtoLinear(i, j, k - 1);
							output[idxM] += v * input[idx];
						}
					}
					else
					{
						output[idx] += v * input[idx];

						if (k < sizeZ - 1)
						{
							int idxP = index3DtoLinear(i, j, k + 1);
							output[idxP] += -v * input[idx];
						}
					}
				}
			}
		}
	}

	void doTimesCPU(bool transposed, const Tdata &input, Tdata &output, mySign s)
	{
		Tdata tmp(output.size());

		if (this->inputDimension.size() == 2)
		{
			if (this->gradDirection == 0)
			{
				if (transposed == false)
				{
					this->dxUpwind2d(input, tmp, EQUALS);
				}
				else
				{
					this->dxUpwind2dT(input, tmp, EQUALS);
				}
			}
			else if (this->gradDirection == 1)
			{
				if (transposed == false)
				{
					this->dyUpwind2d(input, tmp, EQUALS);
				}
				else
				{
					this->dyUpwind2dT(input, tmp, EQUALS);
				}
			}
		}
		else if (this->inputDimension.size() == 3)
		{
			if (this->gradDirection == 0)
			{
				if (transposed == false)
				{
					this->dxUpwind3d(input, tmp, EQUALS);
				}
				else
				{
					this->dxUpwind3dT(input, tmp, EQUALS);
				}
			}
			else if (this->gradDirection == 1)
			{
				if (transposed == false)
				{
					this->dyUpwind3d(input, tmp, EQUALS);
				}
				else
				{
					this->dyUpwind3dT(input, tmp, EQUALS);
				}
			}
			else if (this->gradDirection == 2)
			{
				if (transposed == false)
				{
					this->dzUpwind3d(input, tmp, EQUALS);
				}
				else
				{
					this->dzUpwind3dT(input, tmp, EQUALS);
				}
			}
		}
		else
		{
			printf("UpwindGradient not implemented for dim!={2,3}\n");
			//TODO: implement gradient for dim!={2,3} for CPU version
		}

		switch(s)
		{
			case EQUALS:
				output = tmp;
				break;

			case PLUS:
				for(size_t i=0;i<output.size();++i)
					output[i] += tmp[i];
				break;

			case MINUS:
				for(size_t i=0;i<output.size();++i)
					output[i] -= tmp[i];
				break;
		}
	}

	//TODO correct for sign cases
	#ifdef __CUDACC__
	void doTimesCUDA(bool transposed, const Tdata &input, Tdata &output, mySign s)
	{
		size_t sizeX = this->inputDimension[0];
		size_t sizeY = 1;
		size_t sizeZ = 1;

		if (this->inputDimension.size() > 1)
		{
			sizeY = this->inputDimension[1];
		}
		if (this->inputDimension.size() > 2)
		{
			sizeZ = this->inputDimension[2];
		}

		dim3 block = dim3(32,16,1);
		dim3 grid = dim3((sizeX + block.x - 1) / block.x, (sizeY + block.y - 1) / block.y, (sizeZ + block.z - 1) / block.z);

		T* ptrOutput = thrust::raw_pointer_cast(output.data());
		const T* ptrInput = thrust::raw_pointer_cast(input.data());

		if (this->inputDimension.size() == 2)
		{
			if (this->gradDirection == 0)
			{
				if (transposed == false)
				{
					dxUpwind2dCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s);
				}
				else
				{
					dxUpwind2dTCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s);
				}
			}
			else if (this->gradDirection == 1)
			{
				if (transposed == false)
				{
					dyUpwind2dCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s);
				}
				else
				{
					dyUpwind2dTCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s);
				}
			}
		}
		else if (this->inputDimension.size() == 3)
		{
			if (this->gradDirection == 0)
			{
				if (transposed == false)
				{
					dxUpwind3dCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], this->inputDimension[2], s);
				}
				else
				{
					dxUpwind3dTCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], this->inputDimension[2], s);
				}
			}
			else if (this->gradDirection == 1)
			{
				if (transposed == false)
				{
					dyUpwind3dCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], this->inputDimension[2], s);
				}
				else
				{
					dyUpwind3dTCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], this->inputDimension[2], s);
				}
			}
			else if (this->gradDirection == 2)
			{
				if (transposed == false)
				{
					dzUpwind3dCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], this->inputDimension[2], s);
				}
				else
				{
					dzUpwind3dTCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], this->inputDimension[2], s);
				}
			}
		}
		else
		{
			printf("UpwindGradient not implemented for dim!={2,3}\n");
			//TODO: implement gradient for dim!={2,3} for GPU version
		}
	}
	#endif

	void doTimes(bool transposed, const Tdata &input, Tdata &output, mySign s)
	{
		if (this->isMinus && s == PLUS)
		{
			s = MINUS;
		}
		else if (this->isMinus && s == MINUS)
		{
			s = PLUS;
		}

		/*
		// flip sign and transposed
		if (this->type == backward)
		{
			transposed = !transposed;

			if (s == PLUS)
				s = MINUS;
			else if (s == MINUS)
				s = PLUS;
		}
		*/

		#ifdef __CUDACC__
			this->doTimesCUDA(transposed, input, output, s);
		#else
			this->doTimesCPU(transposed, input, output, s);
		#endif
	}

	void timesPlus(bool transposed, const Tdata &input, Tdata &output)
	{
		this->doTimes(transposed, input, output, PLUS);
	}

	void timesMinus(bool transposed, const Tdata &input, Tdata &output)
	{
		this->doTimes(transposed, input, output, MINUS);
	}

	void times(bool transposed, const Tdata &input, Tdata &output)
	{
		this->doTimes(transposed, input, output, EQUALS);
	}

	T getMaxRowSumAbs(bool transposed)
	{
		T vmax = 0;
		Tdata absRowSum = getAbsRowSum(false);

		for(size_t i=0;i<motionField.size();++i)
		{
			vmax = std::max(vmax,absRowSum[i]);
		}

		return (T)vmax;
	}

	std::vector<T> getAbsRowSum(bool transposed)
	{
		std::vector<T> result(this->getNumRows());

		for(size_t j = 0; j<inputDimension[1]; ++j)
		{
			for(size_t i=1;i<inputDimension[0]-1;++i)
			{
				int tmpIndex = index2DtoLinear(i,j);
				int tmpIndex1 = index2DtoLinear(i+1,j);
				int tmpIndex2 = index2DtoLinear(i-1,j);
				result[tmpIndex] += (T)std::abs(motionField[tmpIndex])+std::abs(motionField[tmpIndex1])+std::abs(motionField[tmpIndex2]);	
			}
			int tmpIndex_edge1 = index2DtoLinear(0,j);
			int tmpIndex_edge2 = index2DtoLinear(1,j);

			result[tmpIndex_edge1] = (T)std::abs(motionField[tmpIndex_edge1])+std::abs(motionField[tmpIndex_edge2]);

			int tmpIndex_edge3 = index2DtoLinear(inputDimension[0]-1,j);
			int tmpIndex_edge4 = index2DtoLinear(inputDimension[0]-2,j);
			
			result[tmpIndex_edge3] = (T)std::abs(motionField[tmpIndex_edge3])+std::abs(motionField[tmpIndex_edge4]);
			//result[i] = (T)3*std::abs(motionField[i]);
		}
		return result;
	}

	int index3DtoLinear(int i, int j, int k)
	{
		return (i + j*this->inputDimension[0] + k*this->inputDimension[0] * this->inputDimension[1]);
	}

	int index2DtoLinear(int i, int j)
	{
		return (i + j*this->inputDimension[0]);
	}

	#ifdef __CUDACC__
	thrust::device_vector<T> getAbsRowSumCUDA(bool transposed)
	{
		thrust::device_vector<T> result(this->getNumRows());

		thrust::transform(
			motionField.begin(),
			motionField.end(),
			result.begin(),
			[] __host__ __device__ (T v)
			{
				return (T)2 * abs(v);
			}
		);

		return result;
	}
	#endif
};

#endif
