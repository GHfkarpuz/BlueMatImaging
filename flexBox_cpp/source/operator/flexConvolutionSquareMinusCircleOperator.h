#ifndef flexConvolutionSquareMinusCircleOperator_H
#define flexConvolutionSquareMinusCircleOperator_H

#include <vector>
#include <cmath>
#include "flexLinearOperator.h"

//! represents an upwind gradient operator
template<typename T>
class flexConvolutionSquareMinusCircleOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	std::vector<int> inputDimension;
    int radius;
    int squareSize;
    int circleSize;
    bool withTimeDimension;
    std::vector<int> circleIndexSet;

public:

	//! initializes the upwind gradient operator
	/*!
        \param aRadius radius of the circle in pixels
        \param aWithTimeDimension is true if the last dimension is the for the time
        \param AInputDimension vector of dimensions   
        \param aMinus determines if operator is negated \sa isMinus
	*/
	flexConvolutionSquareMinusCircleOperator(int aRadius, bool aWithTimeDimension, std::vector<int> AInputDimension, bool aMinus) :
		radius(aRadius),
        withTimeDimension(aWithTimeDimension),
        inputDimension(AInputDimension),
        flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension), ConvolutionSquareMinusCircleOp, aMinus)
	{
        if(this->withTimeDimension)
        {
            if(this->inputDimension.size()==3)
            {
                if(this->radius<2)
                {
                    printf("radius has to be at least 3");
                }
                this->squareSize = 4*this->radius*this->radius;
                this->circleIndexSet.resize(this->radius, (int) 0);
                for(int i = 0; i<radius;++i)
                {
                    while(sqrt((this->circleIndexSet[i]+1)*(this->circleIndexSet[i]+1)+i*i)<=this->radius)
                    {
                        this->circleIndexSet[i]+=1;
                    }
                }
            }
            else
            {
                printf("cases for spatial dimension >2 not implemented yet");
            }
        }
    };

	~flexConvolutionSquareMinusCircleOperator()
	{
	}

	flexConvolutionSquareMinusCircleOperator<T>* copy()
	{
		return new flexConvolutionSquareMinusCircleOperator<T>(this->radius, this->withTimeDimension, this->inputDimension, this->isMinus);
	}

	//new with sign cases here
	void doTimesCPU(bool transposed, const Tdata &input, Tdata &output, mySign s)
    {
        if(!transposed)
        {   
            if(withTimeDimension)
            {
                if(inputDimension.size()==2)
                {
                    int sizeT = this->inputDimension[2];
                    int sizeY = this->inputDimension[1];
                    int sizeX = this->inputDimension[0];

                    for(int t = 0; t < sizeT; ++t)
                    {
                        for(int j = 0; j < sizeY; ++j)
                        {
                            for(int i = 0; i < sizeX; ++i)
                            {
                                int tmpIndex = index3DtoLinear(i,j,t);
                                T valCircle = 0.0;
                                T valSquare = 0.0;

                                int numSquareElements = 0;
                                int numCircleElements = 0;
                            
                                for(int k = 0; k<this->radius; ++k)
                                {
                                    for(int l = 0; l<circleIndexSet[k]; ++l)
                                    {
                                        int tmpIndexToCheck1 = index3DtoLinear(i+l, j+k,t);
                                        int tmpIndexToCheck2 = index3DtoLinear(i-l, j+k,t);
                                        if((i+l)<sizeX && (j+k)<sizeY)
                                        {
                                            valCircle += input[tmpIndexToCheck1];
                                            numCircleElements += 1;
                                        }
                                        if((i-l)>=0 && (j+k)<sizeY)
                                        {
                                            valCircle += input[tmpIndexToCheck2];
                                            numCircleElements += 1;
                                        }
                                        int tmpIndexToCheck3 = index3DtoLinear(i+l, j-k,t);
                                        int tmpIndexToCheck4 = index3DtoLinear(i-l, j-k,t);
                                        if((i+l)<sizeX && (j+k)>=0)
                                        {
                                            valCircle += input[tmpIndexToCheck3];
                                            numCircleElements += 1;
                                        }
                                        if((i-l)>=0 && (j-k)>=0)
                                        {
                                            valCircle += input[tmpIndexToCheck4];
                                            numCircleElements += 1;
                                        }
                                    }
                                }
                                
                                valSquare = valCircle;
                                numSquareElements = numCircleElements;
                                
                                for(int k = 0; k<this->radius; ++k)
                                {
                                    for(int l = circleIndexSet[k]; l < radius; ++l)
                                    {
                                        int tmpIndexToCheck1 = index3DtoLinear(i+l, j+k,t);
                                        int tmpIndexToCheck2 = index3DtoLinear(i-l, j+k,t);
                                        if((i+l)<sizeX && (j+k)<sizeY)
                                        {
                                            valSquare += input[tmpIndexToCheck1];
                                            numSquareElements += 1;
                                        }
                                        if((i-l)>=0 && (j+k)<sizeY)
                                        {
                                            valSquare += input[tmpIndexToCheck2];
                                            numSquareElements += 1;
                                        }
                                        int tmpIndexToCheck3 = index3DtoLinear(i+l, j-k,t);
                                        int tmpIndexToCheck4 = index3DtoLinear(i-l, j-k,t);
                                        if((i+l)<sizeX && (j+k)>=0)
                                        {
                                            valSquare += input[tmpIndexToCheck3];
                                            numSquareElements += 1;
                                        }
                                        if((i-l)>=0 && (j-k)>=0)
                                        {
                                            valSquare += input[tmpIndexToCheck4];
                                            numSquareElements += 1;
                                        }
                                    }
                                }

                                T val = (valSquare/float(numSquareElements))-(valCircle/float(numCircleElements));
                                switch (s)
                                {
                                    case PLUS:   output[tmpIndex] += val; break;
                                    case MINUS:  output[tmpIndex] -= val; break;
                                    case EQUALS: output[tmpIndex]  = val; break;
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                printf("not implemented yet for spatial dimensions > 2 and no time dimension");
            }
        }
        else
        {
            if(withTimeDimension)
            {
                if(inputDimension.size()==2)
                {
                    int sizeT = this->inputDimension[2];
                    int sizeY = this->inputDimension[1];
                    int sizeX = this->inputDimension[0];

                    if (s == EQUALS)
                    {
                        std::fill(output.begin(), output.end(), static_cast<T>(0));
                    }

                    T signFactor = (s == MINUS) ? -1.0 : 1.0;
                    
                    for(int t = 0; t < sizeT; ++t)
                    {
                        for(int j = 0; j < sizeY; ++j)
                        {
                            for(int i = 0; i < sizeX; ++i)
                            {
                                int tmpIndex = index3DtoLinear(i, j,t);
                                T inputVal = input[tmpIndex];

                                if (inputVal == 0.0) continue;

                                int numCircleElements = 0;
                                for(int k = 0; k < this->radius; ++k)
                                {
                                    for(int l = 0; l < circleIndexSet[k]; ++l)
                                    {
                                        if((i+l)<sizeX && (j+k)<sizeY) numCircleElements++;
                                        if((i-l)>=0   && (j+k)<sizeY) numCircleElements++;
                                        if((i+l)<sizeX && (j-k)>=0)   numCircleElements++;
                                        if((i-l)>=0   && (j-k)>=0)   numCircleElements++;
                                    }
                                }

                                int numSquareElements = numCircleElements;
                                for(int k = 0; k < this->radius; ++k)
                                {
                                    for(int l = circleIndexSet[k]; l < radius; ++l)
                                    {
                                        if((i+l)<sizeX && (j+k)<sizeY) numSquareElements++;
                                        if((i-l)>=0   && (j+k)<sizeY) numSquareElements++;
                                        if((i+l)<sizeX && (j-k)>=0)   numSquareElements++;
                                        if((i-l)>=0   && (j-k)>=0)   numSquareElements++;
                                    }
                                }

                                T weightSquare = (inputVal / static_cast<float>(numSquareElements)) * signFactor;
                                T weightCircle = (inputVal / static_cast<float>(numCircleElements)) * signFactor;

                                for(int k = 0; k < this->radius; ++k)
                                {
                                    for(int l = 0; l < circleIndexSet[k]; ++l)
                                    {
                                        T totalWeight = weightSquare - weightCircle; // Weil: (Val/Sq) - (Val/Circ)

                                        if((i+l)<sizeX && (j+k)<sizeY) output[index3DtoLinear(i+l, j+k,t)] += totalWeight;
                                        if((i-l)>=0   && (j+k)<sizeY) output[index3DtoLinear(i-l, j+k,t)] += totalWeight;
                                        if((i+l)<sizeX && (j-k)>=0)   output[index3DtoLinear(i+l, j-k,t)] += totalWeight;
                                        if((i-l)>=0   && (j-k)>=0)   output[index3DtoLinear(i-l, j-k,t)] += totalWeight;
                                    }
                                }

                                for(int k = 0; k < this->radius; ++k)
                                {
                                    for(int l = circleIndexSet[k]; l < radius; ++l)
                                    {
                                        if((i+l)<sizeX && (j+k)<sizeY) output[index3DtoLinear(i+l, j+k,t)] += weightSquare;
                                        if((i-l)>=0   && (j+k)<sizeY) output[index3DtoLinear(i-l, j+k,t)] += weightSquare;
                                        if((i+l)<sizeX && (j-k)>=0)   output[index3DtoLinear(i+l, j-k,t)] += weightSquare;
                                        if((i-l)>=0   && (j-k)>=0)   output[index3DtoLinear(i-l, j-k,t)] += weightSquare;
                                    }
                                }

                            }
                        }
                    }
                }
            }
            else
            {
                printf("not implemented yet for spatial dimensions > 2 and no time dimension\n");
            }
        }
    }

	//TODO correct for sign cases
	#ifdef __CUDACC__
	void doTimesCUDA(bool transposed, const Tdata &input, Tdata &output, mySign s)
	{
		
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
        /*
		#ifdef __CUDACC__
			this->doTimesCUDA(transposed, input, output, s);
		#else
			this->doTimesCPU(transposed, input, output, s);
		#endif
        */
        this->doTimesCPU(transposed, input, output, s);
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
		Tdata absRowSum = getAbsRowSum(transposed);

		

		return (T)vmax;
	}

    std::vector<T> getAbsRowSum(bool transposed)
    {
        int numRows = transposed ? this->getNumCols() : this->getNumRows();
        std::vector<T> result(numRows, static_cast<T>(0));

        if (withTimeDimension && inputDimension.size() == 3)
        {
            int sizeT = this->inputDimension[2];
            int sizeY = this->inputDimension[1];
            int sizeX = this->inputDimension[0];

            for (int t = 0; t < sizeT; ++t)
            {
                for (int j = 0; j < sizeY; ++j)
                {
                    for (int i = 0; i < sizeX; ++i)
                    {
                        int tmpIndex = index3DtoLinear(i, j, t);

                        int numCircleElements = 0;
                        for (int k = 0; k < this->radius; ++k)
                        {
                            for (int l = 0; l < circleIndexSet[k]; ++l)
                            {
                                if ((i + l) < sizeX && (j + k) < sizeY) numCircleElements++;
                                if ((i - l) >= 0   && (j + k) < sizeY) numCircleElements++;
                                if ((i + l) < sizeX && (j - k) >= 0)   numCircleElements++;
                                if ((i - l) >= 0   && (j - k) >= 0)   numCircleElements++;
                            }
                        }

                        int numSquareElements = numCircleElements;
                        for (int k = 0; k < this->radius; ++k)
                        {
                            for (int l = circleIndexSet[k]; l < radius; ++l)
                            {
                                if ((i + l) < sizeX && (j + k) < sizeY) numSquareElements++;
                                if ((i - l) >= 0   && (j + k) < sizeY) numSquareElements++;
                                if ((i + l) < sizeX && (j - k) >= 0)   numSquareElements++;
                                if ((i - l) >= 0   && (j - k) >= 0)   numSquareElements++;
                            }
                        }

                        T absWeightCircleZone = std::abs(1.0f / static_cast<float>(numSquareElements) - 1.0f / static_cast<float>(numCircleElements));
                        T absWeightSquareZone = 1.0f / static_cast<float>(numSquareElements);

                        T rowSum = 0.0;

                        rowSum += numCircleElements * absWeightCircleZone;
                        
                        rowSum += (numSquareElements - numCircleElements) * absWeightSquareZone;

                        if (!transposed)
                        {
                            result[tmpIndex] = rowSum;
                        }
                        else
                        {
                            for (int k = 0; k < this->radius; ++k)
                            {
                                for (int l = 0; l < circleIndexSet[k]; ++l)
                                {
                                    if ((i + l) < sizeX && (j + k) < sizeY) result[index3DtoLinear(i + l, j + k, t)] += absWeightCircleZone;
                                    if ((i - l) >= 0   && (j + k) < sizeY) result[index3DtoLinear(i - l, j + k, t)] += absWeightCircleZone;
                                    if ((i + l) < sizeX && (j - k) >= 0)   result[index3DtoLinear(i + l, j - k, t)] += absWeightCircleZone;
                                    if ((i - l) >= 0   && (j - k) >= 0)   result[index3DtoLinear(i - l, j - k, t)] += absWeightCircleZone;
                                }
                                for (int l = circleIndexSet[k]; l < radius; ++l)
                                {
                                    if ((i + l) < sizeX && (j + k) < sizeY) result[index3DtoLinear(i + l, j + k, t)] += absWeightSquareZone;
                                    if ((i - l) >= 0   && (j + k) < sizeY) result[index3DtoLinear(i - l, j + k, t)] += absWeightSquareZone;
                                    if ((i + l) < sizeX && (j - k) >= 0)   result[index3DtoLinear(i + l, j - k, t)] += absWeightSquareZone;
                                    if ((i - l) >= 0   && (j - k) >= 0)   result[index3DtoLinear(i - l, j - k, t)] += absWeightSquareZone;
                                }
                            }
                        }
                    }
                }
            }
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

    /*
	#ifdef __CUDACC__
	thrust::device_vector<T> getAbsRowSumCUDA3D(bool transposed)
	{
	}
	#endif
    */

	/*
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
	*/
};

#endif
