#ifndef flexGradientOperator_H
#define flexGradientOperator_H

#include <vector>
#include "flexLinearOperator.h"

//TODO: extend the CUDA methods for the central gradient case with the right handling for the edges
#ifdef __CUDACC__
template<typename T>
__global__ void dxp2dCUDA(T* output, const T* input, const size_t sizeX, const size_t sizeY, mySign s)
{
	if(type == forward)
	{
		const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
		const size_t y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= sizeX || y >= sizeY)
			return;

		const size_t tmpIndex = x + y * sizeX;

		if (x < sizeX - 1)
		{
			switch (s)
			{
				case PLUS:
				{
					output[tmpIndex] += input[tmpIndex + 1] - input[tmpIndex];
					break;
				}
				case MINUS:
				{
					output[tmpIndex] -= input[tmpIndex + 1] - input[tmpIndex];
					break;
				}
				case EQUALS:
				{
					output[tmpIndex] = input[tmpIndex + 1] - input[tmpIndex];
					break;
				}
			}
		}
	}
	else if(type == central)
	{
		const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
		const size_t y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= sizeX || y >= sizeY)
			return;

		const size_t idx = x + y * sizeX;

		// interior
		if (x > 0 && x < sizeX - 1)
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(0.5) * (input[idx + 1] - input[idx - 1]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(0.5) * (input[idx + 1] - input[idx - 1]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(0.5) * (input[idx + 1] - input[idx - 1]);
					break;
			}
		}
		else if (x == 0)
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(0.5) * (input[idx + 1] - input[idx]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(0.5) * (input[idx + 1] - input[idx]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(0.5) * (input[idx + 1] - input[idx]);
					break;
			}
		}
		else
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(0.5) * (input[idx] - input[idx - 1]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(0.5) * (input[idx] - input[idx - 1]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(0.5) * (input[idx] - input[idx - 1]);
					break;
			}
		}
	}
}

template<typename T>
__global__ void dxp2dTransposedCUDA(T* output, const T* input, const size_t sizeX, const size_t sizeY, mySign s)
{
	if(type == forward)
	{
		const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
		const size_t y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= sizeX || y >= sizeY)
			return;

		const size_t tmpIndex = x + y * sizeX;

		if (x < sizeX - 1 && x > 0)
		{
			switch (s)
			{
				case PLUS:
				{
					output[tmpIndex] += -(input[tmpIndex] - input[tmpIndex - 1]);
					break;
				}
				case MINUS:
				{
					output[tmpIndex] -= -(input[tmpIndex] - input[tmpIndex - 1]);
					break;
				}
				case EQUALS:
				{
					output[tmpIndex] = -(input[tmpIndex] - input[tmpIndex - 1]);
					break;
				}
			}
		}
		else if (x == 0)
		{
			switch (s)
			{
			case PLUS:
			{
				output[tmpIndex] += -(input[tmpIndex]);
				break;
			}
			case MINUS:
			{
				output[tmpIndex] -= -(input[tmpIndex]);
				break;
			}
			case EQUALS:
			{
				output[tmpIndex] = -(input[tmpIndex]);
				break;
			}
			}
		}
		else
		{
			switch (s)
			{
			case PLUS:
			{
				output[tmpIndex] += (input[tmpIndex - 1]);
				break;
			}
			case MINUS:
			{
				output[tmpIndex] -= (input[tmpIndex - 1]);
				break;
			}
			case EQUALS:
			{
				output[tmpIndex] = (input[tmpIndex - 1]);
				break;
			}
			}
		}
	}
	// dxp2dTransposedCUDA
	else if(type == central)
	{
		const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
		const size_t y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= sizeX || y >= sizeY)
			return;

		const size_t idx = x + y * sizeX;

		if (x > 0 && x < sizeX - 1)
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(-0.5) * (input[idx + 1] - input[idx - 1]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(-0.5) * (input[idx + 1] - input[idx - 1]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(-0.5) * (input[idx + 1] - input[idx - 1]);
					break;
			}
		}
		else if (x == 0)
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(-0.5) * (input[idx + 1] + input[idx]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(-0.5) * (input[idx + 1] + input[idx]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(-0.5) * (input[idx + 1] + input[idx]);
					break;
			}
		}
		else
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(0.5) * (input[idx] + input[idx - 1]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(0.5) * (input[idx] + input[idx - 1]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(0.5) * (input[idx] + input[idx - 1]);
					break;
			}
		}
	}
}

template<typename T>
__global__ void dyp2dCUDA(T* output, const T* input, const size_t sizeX, const size_t sizeY, mySign s)
{
	if(type == forward)
		{
		const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
		const size_t y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= sizeX || y >= sizeY)
			return;

		const size_t tmpIndex = x + y * sizeX;

		if (y < sizeY - 1)
		{
			switch (s)
			{
			case PLUS:
			{
				output[tmpIndex] += input[tmpIndex + sizeX] - input[tmpIndex];
				break;
			}
			case MINUS:
			{
				output[tmpIndex] -= input[tmpIndex + sizeX] - input[tmpIndex];
				break;
			}
			case EQUALS:
			{
				output[tmpIndex] = input[tmpIndex + sizeX] - input[tmpIndex];
				break;
			}
			}
		}
	}
	// dyp2dCUDA
	else if(type == central)
	{
		const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
		const size_t y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= sizeX || y >= sizeY)
			return;

		const size_t idx = x + y * sizeX;

		if (y > 0 && y < sizeY - 1)
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(0.5) * (input[idx + sizeX] - input[idx - sizeX]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(0.5) * (input[idx + sizeX] - input[idx - sizeX]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(0.5) * (input[idx + sizeX] - input[idx - sizeX]);
					break;
			}
		}
		else if (y == 0)
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(0.5) * (input[idx + sizeX] - input[idx]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(0.5) * (input[idx + sizeX] - input[idx]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(0.5) * (input[idx + sizeX] - input[idx]);
					break;
			}
		}
		else
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(0.5) * (input[idx] - input[idx - sizeX]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(0.5) * (input[idx] - input[idx - sizeX]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(0.5) * (input[idx] - input[idx - sizeX]);
					break;
			}
		}
	}
}

template<typename T>
__global__ void dyp2dTransposedCUDA(T* output, const T* input, const size_t sizeX, const size_t sizeY, mySign s)
{
	if(type == forward)
	{
		const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
		const size_t y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= sizeX || y >= sizeY)
			return;

		const size_t tmpIndex = x + y * sizeX;

		if (y < sizeY - 1 && y > 0)
		{
			switch (s)
			{
			case PLUS:
			{
				output[tmpIndex] += -(input[tmpIndex] - input[tmpIndex - sizeX]);
				break;
			}
			case MINUS:
			{
				output[tmpIndex] -= -(input[tmpIndex] - input[tmpIndex - sizeX]);
				break;
			}
			case EQUALS:
			{
				output[tmpIndex] = -(input[tmpIndex] - input[tmpIndex - sizeX]);
				break;
			}
			}
		}
		else if (y == 0)
		{
			switch (s)
			{
			case PLUS:
			{
				output[tmpIndex] += -(input[tmpIndex]);
				break;
			}
			case MINUS:
			{
				output[tmpIndex] -= -(input[tmpIndex]);
				break;
			}
			case EQUALS:
			{
				output[tmpIndex] = -(input[tmpIndex]);
				break;
			}
			}
		}
		else
		{
			switch (s)
			{
			case PLUS:
			{
				output[tmpIndex] += (input[tmpIndex - sizeX]);
				break;
			}
			case MINUS:
			{
				output[tmpIndex] -= (input[tmpIndex - sizeX]);
				break;
			}
			case EQUALS:
			{
				output[tmpIndex] = (input[tmpIndex - sizeX]);
				break;
			}
			}
		}
	}
	// dyp2dTransposedCUDA
	else if(type == central)
	{
		const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
		const size_t y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= sizeX || y >= sizeY)
			return;

		const size_t idx = x + y * sizeX;

		if (y > 0 && y < sizeY - 1)
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(-0.5) * (input[idx + sizeX] - input[idx - sizeX]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(-0.5) * (input[idx + sizeX] - input[idx - sizeX]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(-0.5) * (input[idx + sizeX] - input[idx - sizeX]);
					break;
			}
		}
		else if (y == 0)
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(-0.5) * (input[idx + sizeX] + input[idx]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(-0.5) * (input[idx + sizeX] + input[idx]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(-0.5) * (input[idx + sizeX] + input[idx]);
					break;
			}
		}
		else
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(0.5) * (input[idx] + input[idx - sizeX]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(0.5) * (input[idx] + input[idx - sizeX]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(0.5) * (input[idx] + input[idx - sizeX]);
					break;
			}
		}
	}
}


__device__
int getGlobalIdx_3D_3D(){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

template<typename T>
__global__ void dxp3dCUDA(T* output, const T* input, const size_t sizeX, const size_t sizeY, const size_t sizeZ, mySign s)
{
	if(type == forward)
	{
		const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
		const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
		const size_t z = blockDim.z * blockIdx.z + threadIdx.z;

		const size_t tmpIndex = x + sizeX * y + sizeX * sizeY * z;

		if (x >= sizeX || y >= sizeY || z >= sizeZ)
			return;

		if (x < sizeX - 1)
		{
			switch (s)
			{
			case PLUS:
			{
				output[tmpIndex] += input[tmpIndex + 1] - input[tmpIndex];
				break;
			}
			case MINUS:
			{
				output[tmpIndex] -= input[tmpIndex + 1] - input[tmpIndex];
				break;
			}
			case EQUALS:
			{
				output[tmpIndex] = input[tmpIndex + 1] - input[tmpIndex];
				break;
			}
			}
		}
	}
	else if(type == central)
	{
		const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
		const size_t y = threadIdx.y + blockIdx.y * blockDim.y;
		const size_t z = threadIdx.z + blockIdx.z * blockDim.z;

		if (x >= sizeX || y >= sizeY || z >= sizeZ)
			return;

		const size_t idx = x + y * sizeX + z * sizeX * sizeY;

		if (x > 0 && x < sizeX - 1)
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(0.5) * (input[idx + 1] - input[idx - 1]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(0.5) * (input[idx + 1] - input[idx - 1]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(0.5) * (input[idx + 1] - input[idx - 1]);
					break;
			}
		}
		else if (x == 0)
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(0.5) * (input[idx + 1] - input[idx]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(0.5) * (input[idx + 1] - input[idx]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(0.5) * (input[idx + 1] - input[idx]);
					break;
			}
		}
		else
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(0.5) * (input[idx] - input[idx - 1]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(0.5) * (input[idx] - input[idx - 1]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(0.5) * (input[idx] - input[idx - 1]);
					break;
			}
		}
	}
}

template<typename T>
__global__ void dxp3dTransposedCUDA(T* output, const T* input, const size_t sizeX, const size_t sizeY, const size_t sizeZ, mySign s)
{
	if(type == forward)
	{
		const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
		const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
		const size_t z = blockDim.z * blockIdx.z + threadIdx.z;

		const size_t tmpIndex = x + sizeX * y + sizeX * sizeY * z;

		if (x >= sizeX || y >= sizeY || z >= sizeZ)
			return;

		if (x < sizeX - 1 && x > 0)
		{
			switch (s)
			{
				case PLUS:
				{
					output[tmpIndex] += -(input[tmpIndex] - input[tmpIndex - 1]);
					break;
				}
				case MINUS:
				{
					output[tmpIndex] -= -(input[tmpIndex] - input[tmpIndex - 1]);
					break;
				}
				case EQUALS:
				{
					output[tmpIndex] = -(input[tmpIndex] - input[tmpIndex - 1]);
					break;
				}
			}
		}
		else if (x == 0)
		{
			switch (s)
			{
				case PLUS:
				{
					output[tmpIndex] += -(input[tmpIndex]);
					break;
				}
				case MINUS:
				{
					output[tmpIndex] -= -(input[tmpIndex]);
					break;
				}
				case EQUALS:
				{
					output[tmpIndex] = -(input[tmpIndex]);
					break;
				}
			}
		}
		else
		{
			switch (s)
			{
				case PLUS:
				{
					output[tmpIndex] += (input[tmpIndex - 1]);
					break;
				}
				case MINUS:
				{
					output[tmpIndex] -= (input[tmpIndex - 1]);
					break;
				}
				case EQUALS:
				{
					output[tmpIndex] = (input[tmpIndex - 1]);
					break;
				}
			}
		}
	}
	else if(type == central)
	{
		const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
		const size_t y = threadIdx.y + blockIdx.y * blockDim.y;
		const size_t z = threadIdx.z + blockIdx.z * blockDim.z;

		if (x >= sizeX || y >= sizeY || z >= sizeZ)
			return;

		const size_t idx = x + y * sizeX + z * sizeX * sizeY;

		if (x > 0 && x < sizeX - 1)
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(-0.5) * (input[idx + 1] - input[idx - 1]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(-0.5) * (input[idx + 1] - input[idx - 1]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(-0.5) * (input[idx + 1] - input[idx - 1]);
					break;
			}
		}
		else if (x == 0)
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(-0.5) * (input[idx + 1] + input[idx]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(-0.5) * (input[idx + 1] + input[idx]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(-0.5) * (input[idx + 1] + input[idx]);
					break;
			}
		}
		else
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(0.5) * (input[idx] + input[idx - 1]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(0.5) * (input[idx] + input[idx - 1]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(0.5) * (input[idx] + input[idx - 1]);
					break;
			}
		}
	}
}

template<typename T>
__global__ void dyp3dCUDA(T* output, const T* input, const size_t sizeX, const size_t sizeY, const size_t sizeZ, mySign s)
{
	if(type == forward)
	{
		const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
		const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
		const size_t z = blockDim.z * blockIdx.z + threadIdx.z;

		const size_t tmpIndex = x + sizeX * y + sizeX * sizeY * z;

		if (x >= sizeX || y >= sizeY || z >= sizeZ)
			return;

		if (y < sizeY - 1)
		{
			switch (s)
			{
				case PLUS:
				{
					output[tmpIndex] += input[tmpIndex + sizeX] - input[tmpIndex];
					break;
				}
				case MINUS:
				{
					output[tmpIndex] -= input[tmpIndex + sizeX] - input[tmpIndex];
					break;
				}
				case EQUALS:
				{
					output[tmpIndex] = input[tmpIndex + sizeX] - input[tmpIndex];
					break;
				}
			}
		}
	}
	else if(type == central)
	{
		const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
		const size_t y = threadIdx.y + blockIdx.y * blockDim.y;
		const size_t z = threadIdx.z + blockIdx.z * blockDim.z;

		if (x >= sizeX || y >= sizeY || z >= sizeZ)
			return;

		const size_t idx = x + y * sizeX + z * sizeX * sizeY;

		if (y > 0 && y < sizeY - 1)
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(0.5) * (input[idx + sizeX] - input[idx - sizeX]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(0.5) * (input[idx + sizeX] - input[idx - sizeX]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(0.5) * (input[idx + sizeX] - input[idx - sizeX]);
					break;
			}
		}
		else if (y == 0)
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(0.5) * (input[idx + sizeX] - input[idx]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(0.5) * (input[idx + sizeX] - input[idx]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(0.5) * (input[idx + sizeX] - input[idx]);
					break;
			}
		}
		else
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(0.5) * (input[idx] - input[idx - sizeX]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(0.5) * (input[idx] - input[idx - sizeX]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(0.5) * (input[idx] - input[idx - sizeX]);
					break;
			}
		}
	}
}

template<typename T>
__global__ void dyp3dTransposedCUDA(T* output, const T* input, const size_t sizeX, const size_t sizeY, const size_t sizeZ, mySign s)
{
	if(type == forward)
	{
		const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
		const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
		const size_t z = blockDim.z * blockIdx.z + threadIdx.z;

		const size_t tmpIndex = x + sizeX * y + sizeX * sizeY * z;

		if (x >= sizeX || y >= sizeY || z >= sizeZ)
			return;

		if (y < sizeY - 1 && y > 0)
		{
			switch (s)
			{
				case PLUS:
				{
					output[tmpIndex] += -(input[tmpIndex] - input[tmpIndex - sizeX]);
					break;
				}
				case MINUS:
				{
					output[tmpIndex] -= -(input[tmpIndex] - input[tmpIndex - sizeX]);
					break;
				}
				case EQUALS:
				{
					output[tmpIndex] = -(input[tmpIndex] - input[tmpIndex - sizeX]);
					break;
				}
			}
		}
		else if (y == 0)
		{
			switch (s)
			{
				case PLUS:
				{
					output[tmpIndex] += -(input[tmpIndex]);
					break;
				}
				case MINUS:
				{
					output[tmpIndex] -= -(input[tmpIndex]);
					break;
				}
				case EQUALS:
				{
					output[tmpIndex] = -(input[tmpIndex]);
					break;
				}
			}
		}
		else
		{
			switch (s)
			{
				case PLUS:
				{
					output[tmpIndex] += (input[tmpIndex - sizeX]);
					break;
				}
				case MINUS:
				{
					output[tmpIndex] -= (input[tmpIndex - sizeX]);
					break;
				}
				case EQUALS:
				{
					output[tmpIndex] = (input[tmpIndex - sizeX]);
					break;
				}
			}
		}
	}
	else if(type == central)
	{
		const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
		const size_t y = threadIdx.y + blockIdx.y * blockDim.y;
		const size_t z = threadIdx.z + blockIdx.z * blockDim.z;

		if (x >= sizeX || y >= sizeY || z >= sizeZ)
			return;

		const size_t idx = x + y * sizeX + z * sizeX * sizeY;

		if (y > 0 && y < sizeY - 1)
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(-0.5) * (input[idx + sizeX] - input[idx - sizeX]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(-0.5) * (input[idx + sizeX] - input[idx - sizeX]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(-0.5) * (input[idx + sizeX] - input[idx - sizeX]);
					break;
			}
		}
		else if (y == 0)
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(-0.5) * (input[idx + sizeX] + input[idx]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(-0.5) * (input[idx + sizeX] + input[idx]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(-0.5) * (input[idx + sizeX] + input[idx]);
					break;
			}
		}
		else
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(0.5) * (input[idx] + input[idx - sizeX]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(0.5) * (input[idx] + input[idx - sizeX]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(0.5) * (input[idx] + input[idx - sizeX]);
					break;
			}
		}
	}
}

template<typename T>
__global__ void dzp3dCUDA(T* output, const T* input, const size_t sizeX, const size_t sizeY, const size_t sizeZ, mySign s)
{
	if(type == forward)
	{
		const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
		const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
		const size_t z = blockDim.z * blockIdx.z + threadIdx.z;

		const size_t tmpIndex = x + sizeX * y + sizeX * sizeY * z;

		if (x >= sizeX || y >= sizeY || z >= sizeZ)
			return;

		if (z < sizeZ - 1)
		{
			switch (s)
			{
				case PLUS:
				{
					output[tmpIndex] += input[tmpIndex + sizeX * sizeY] - input[tmpIndex];
					break;
				}
				case MINUS:
				{
					output[tmpIndex] -= input[tmpIndex + sizeX * sizeY] - input[tmpIndex];
					break;
				}
				case EQUALS:
				{
					output[tmpIndex] = input[tmpIndex + sizeX * sizeY] - input[tmpIndex];
					break;
				}
			}
		}
	}
	else if(type == central)
	{
		const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
		const size_t y = threadIdx.y + blockIdx.y * blockDim.y;
		const size_t z = threadIdx.z + blockIdx.z * blockDim.z;

		if (x >= sizeX || y >= sizeY || z >= sizeZ)
			return;

		const size_t idx = x + y * sizeX + z * sizeX * sizeY;

		if (z > 0 && z < sizeZ - 1)
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(0.5) * (input[idx + sizeX * sizeY] - input[idx - sizeX * sizeY]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(0.5) * (input[idx + sizeX * sizeY] - input[idx - sizeX * sizeY]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(0.5) * (input[idx + sizeX * sizeY] - input[idx - sizeX * sizeY]);
					break;
			}
		}
		else if (z == 0)
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(0.5) * (input[idx + sizeX * sizeY] - input[idx]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(0.5) * (input[idx + sizeX * sizeY] - input[idx]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(0.5) * (input[idx + sizeX * sizeY] - input[idx]);
					break;
			}
		}
		else
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(0.5) * (input[idx] - input[idx - sizeX * sizeY]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(0.5) * (input[idx] - input[idx - sizeX * sizeY]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(0.5) * (input[idx] - input[idx - sizeX * sizeY]);
					break;
			}
		}
	}
}

template<typename T>
__global__ void dzp3dTransposedCUDA(T* output, const T* input, const size_t sizeX, const size_t sizeY, const size_t sizeZ, mySign s)
{
	if(type == forward)
	{
		const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
		const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
		const size_t z = blockDim.z * blockIdx.z + threadIdx.z;

		const size_t tmpIndex = x + sizeX * y + sizeX * sizeY * z;

		if (x >= sizeX || y >= sizeY || z >= sizeZ)
			return;

		if (z < sizeZ - 1 && z > 0)
		{
			switch (s)
			{
				case PLUS:
				{
					output[tmpIndex] += -(input[tmpIndex] - input[tmpIndex - sizeX * sizeY]);
					break;
				}
				case MINUS:
				{
					output[tmpIndex] -= -(input[tmpIndex] - input[tmpIndex - sizeX * sizeY]);
					break;
				}
				case EQUALS:
				{
					output[tmpIndex] = -(input[tmpIndex] - input[tmpIndex - sizeX * sizeY]);
					break;
				}
			}
		}
		else if (z == 0)
		{
			switch (s)
			{
				case PLUS:
				{
					output[tmpIndex] += -(input[tmpIndex]);
					break;
				}
				case MINUS:
				{
					output[tmpIndex] -= -(input[tmpIndex]);
					break;
				}
				case EQUALS:
				{
					output[tmpIndex] = -(input[tmpIndex]);
					break;
				}
			}
		}
		else
		{
			switch (s)
			{
				case PLUS:
				{
					output[tmpIndex] += (input[tmpIndex - sizeX * sizeY]);
					break;
				}
				case MINUS:
				{
					output[tmpIndex] -= (input[tmpIndex - sizeX * sizeY]);
					break;
				}
				case EQUALS:
				{
					output[tmpIndex] = (input[tmpIndex - sizeX * sizeY]);
					break;
				}
			}
		}
	}
	else if(type == central)
	{
		const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
		const size_t y = threadIdx.y + blockIdx.y * blockDim.y;
		const size_t z = threadIdx.z + blockIdx.z * blockDim.z;

		if (x >= sizeX || y >= sizeY || z >= sizeZ)
			return;

		const size_t idx = x + y * sizeX + z * sizeX * sizeY;

		if (z > 0 && z < sizeZ - 1)
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(-0.5) * (input[idx + sizeX * sizeY] - input[idx - sizeX * sizeY]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(-0.5) * (input[idx + sizeX * sizeY] - input[idx - sizeX * sizeY]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(-0.5) * (input[idx + sizeX * sizeY] - input[idx - sizeX * sizeY]);
					break;
			}
		}
		else if (z == 0)
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(-0.5) * (input[idx + sizeX * sizeY] + input[idx]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(-0.5) * (input[idx + sizeX * sizeY] + input[idx]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(-0.5) * (input[idx + sizeX * sizeY] + input[idx]);
					break;
			}
		}
		else
		{
			switch (s)
			{
				case PLUS:
					output[idx] += static_cast<T>(0.5) * (input[idx] + input[idx - sizeX * sizeY]);
					break;
				case MINUS:
					output[idx] -= static_cast<T>(0.5) * (input[idx] + input[idx - sizeX * sizeY]);
					break;
				case EQUALS:
					output[idx]  = static_cast<T>(0.5) * (input[idx] + input[idx - sizeX * sizeY]);
					break;
			}
		}
	}
}

#endif

//! represents a gradient operator
template<typename T>
class flexGradientOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	std::vector<int> inputDimension;
	int gradDirection;
	gradientType type;
	int numberDimensions;

public:

	//! initializes the gradient operator
	/*!
		\param AInputDimension vector of dimensions
		\param aGradDirection direction of gradient. 0 for first dimension and so on.
		\param aType type of gradient. Possible values are forward, backward and central.
		\param aMinus determines if operator is negated \sa isMinus
	*/
	flexGradientOperator(std::vector<int> AInputDimension, int aGradDirection, gradientType aType, bool aMinus) :
		inputDimension(AInputDimension),
		gradDirection(aGradDirection),
		type(aType),
		numberDimensions(static_cast<int>(AInputDimension.size())), flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension), gradientOp, aMinus)
	{};

	flexGradientOperator<T>* copy()
	{
		std::vector<int> dimsCopy;
		dimsCopy.resize(this->inputDimension.size());

		std::copy(this->inputDimension.begin(), this->inputDimension.end(), dimsCopy.begin());

		return new flexGradientOperator<T>(dimsCopy, this->gradDirection, this->type, this->isMinus);
	}

	//
	//
	//  3d cases
	//
	//

	void updateValue(T* ptr, mySign s, T value)
	{
		switch (s)
		{
			case PLUS:
				{
					*ptr += value;
					break;
				}
			case MINUS:
				{
					*ptr -= value;
					break;
				}
			case EQUALS:
				{
					*ptr = value;
					break;
				}
		}
	}

	void dxp3d(const Tdata &input, Tdata &output, mySign s)
	{
		int sizeZ = this->inputDimension[2];
			int sizeY = this->inputDimension[1];
			int sizeX = this->inputDimension[0] - 1;
		if(this->type == forward)
		{
			#pragma omp parallel for
			for (int k = 0; k < sizeZ; ++k)
			{
				for (int j = 0; j < sizeY; ++j)
				{
					for (int i = 0; i < sizeX; ++i)
					{
						const int tmpIndex = this->index3DtoLinear(i, j, k);

						this->updateValue(&output[tmpIndex], s, input[tmpIndex + 1] - input[tmpIndex]);
					}
				}
			}
		}
		else if(this->type == central)
		{
			int sizeZ = this->inputDimension[2];
			int sizeY = this->inputDimension[1];
			int sizeX = this->inputDimension[0];

			#pragma omp parallel for
			for (int k = 0; k < sizeZ; ++k)
			{
				for (int j = 0; j < sizeY; ++j)
				{
					for (int i = 1; i < sizeX-1; ++i)
					{
						const int tmpIndex = this->index3DtoLinear(i, j, k);

						switch (s)
						{
							case PLUS:
							{
								output[tmpIndex] += 0.5*(input[tmpIndex + 1] - input[tmpIndex - 1]);
								break;
							}
							case MINUS:
							{
								output[tmpIndex] -= 0.5*(input[tmpIndex + 1] - input[tmpIndex - 1]);
								break;
							}
							case EQUALS:
							{
								output[tmpIndex] = 0.5*(input[tmpIndex + 1] - input[tmpIndex - 1]);
								break;
							}
						}
					}

					int tmpIndex_edge1 = this->index3DtoLinear(0, j, k);
					int tmpIndex_edge2 = this->index3DtoLinear(1, j, k);

					switch (s)
						{
							case PLUS:
							{
								output[tmpIndex_edge1] += 0.5*(input[tmpIndex_edge2]-input[tmpIndex_edge1]);
								break;
							}
							case MINUS:
							{
								output[tmpIndex_edge1] -= 0.5*(input[tmpIndex_edge2]-input[tmpIndex_edge1]);
								break;
							}
							case EQUALS:
							{
								output[tmpIndex_edge1] = 0.5*(input[tmpIndex_edge2]-input[tmpIndex_edge1]);
								break;
							}
						}

					int tmpIndex_edge3 = this->index3DtoLinear(sizeX-2, j, k);
					int tmpIndex_edge4 = this->index3DtoLinear(sizeX-1, j, k);

					switch (s)
						{
							case PLUS:
							{
								output[tmpIndex_edge4] += 0.5*(input[tmpIndex_edge4]-input[tmpIndex_edge3]);
								break;
							}
							case MINUS:
							{
								output[tmpIndex_edge4] -= 0.5*(input[tmpIndex_edge4]-input[tmpIndex_edge3]);
								break;
							}
							case EQUALS:
							{
								output[tmpIndex_edge4] = 0.5*(input[tmpIndex_edge4]-input[tmpIndex_edge3]);
								break;
							}
						}
					
				}
			}

			/*
			#pragma omp parallel for
			for (int k = 0; k < sizeZ; ++k)
			{
				for (int j = 0; j < sizeY; ++j)
				{
					for (int i = 1; i < sizeX; ++i)
					{
						const int tmpIndex = this->index3DtoLinear(i, j, k);

						this->updateValue(&output[tmpIndex], s, static_cast<T>(0.5) * (input[tmpIndex + 1] - input[tmpIndex-1]));
					}

					int tmpIndex_edge1 = this->index3DtoLinear(0, j, k);

					this->updateValue(&output[tmpIndex_edge1], s, static_cast<T>(0));

					int tmpIndex_edge2 = this->index3DtoLinear(sizeX, j, k);

					this->updateValue(&output[tmpIndex_edge2], s, static_cast<T>(0));
				}
			}
			*/
		}
	}

	void dyp3d(const Tdata &input, Tdata &output, mySign s)
	{
		int sizeZ = this->inputDimension[2];
		int sizeY = this->inputDimension[1] - 1;
		int sizeX = this->inputDimension[0];
		
		if(this->type == forward)
		{
			#pragma omp parallel for
			for (int k = 0; k < sizeZ; ++k)
			{
				for (int j = 0; j < sizeY; ++j)
				{
					for (int i = 0; i < sizeX; ++i)
					{
						const int tmpIndex1 = this->index3DtoLinear(i, j, k);
						const int tmpIndex2 = this->index3DtoLinear(i, j + 1, k);

						this->updateValue(&output[tmpIndex1], s, input[tmpIndex2] - input[tmpIndex1]);
					}
				}
			}
		}
		else if(this->type == central)
		{
			int sizeZ = this->inputDimension[2];
			int sizeY = this->inputDimension[1];
			int sizeX = this->inputDimension[0];

			#pragma omp parallel for
			for (int k = 0; k < sizeZ; ++k)
			{
				for (int i = 0; i < sizeX; ++i)
				{
					for (int j = 1; j < sizeY-1; ++j)
					{
						const int tmpIndex = this->index3DtoLinear(i, j, k);
						int tmpIndex1 = this->index3DtoLinear(i, j+1, k);
						int tmpIndex2 = this->index3DtoLinear(i, j-1, k);

						switch (s)
						{
							case PLUS:
							{
								output[tmpIndex] += 0.5*(input[tmpIndex1] - input[tmpIndex2]);
								break;
							}
							case MINUS:
							{
								output[tmpIndex] -= 0.5*(input[tmpIndex1] - input[tmpIndex2]);
								break;
							}
							case EQUALS:
							{
								output[tmpIndex] = 0.5*(input[tmpIndex1] - input[tmpIndex2]);
								break;
							}
						}
					}

					int tmpIndex_edge1 = this->index3DtoLinear(i, 0, k);
					int tmpIndex_edge2 = this->index3DtoLinear(i, 1, k);

					switch (s)
						{
							case PLUS:
							{
								output[tmpIndex_edge1] += 0.5*(input[tmpIndex_edge2]-input[tmpIndex_edge1]);
								break;
							}
							case MINUS:
							{
								output[tmpIndex_edge1] -= 0.5*(input[tmpIndex_edge2]-input[tmpIndex_edge1]);
								break;
							}
							case EQUALS:
							{
								output[tmpIndex_edge1] = 0.5*(input[tmpIndex_edge2]-input[tmpIndex_edge1]);
								break;
							}
						}

					int tmpIndex_edge3 = this->index3DtoLinear(i, sizeY-2, k);
					int tmpIndex_edge4 = this->index3DtoLinear(i, sizeY-1, k);

					switch (s)
						{
							case PLUS:
							{
								output[tmpIndex_edge4] += 0.5*(input[tmpIndex_edge4]-input[tmpIndex_edge3]);
								break;
							}
							case MINUS:
							{
								output[tmpIndex_edge4] -= 0.5*(input[tmpIndex_edge4]-input[tmpIndex_edge3]);
								break;
							}
							case EQUALS:
							{
								output[tmpIndex_edge4] = 0.5*(input[tmpIndex_edge4]-input[tmpIndex_edge3]);
								break;
							}
						}
					
				}
			}

			/*
			#pragma omp parallel for
			for (int k = 0; k < sizeZ; ++k)
			{
				for (int i = 0; i < sizeX; ++i)
				{
					for (int j = 1; j < sizeY; ++j)
					{
						const int tmpIndex = this->index3DtoLinear(i, j, k);
						const int tmpIndex1 = this->index3DtoLinear(i, j - 1, k);
						const int tmpIndex2 = this->index3DtoLinear(i, j + 1, k);
					
						this->updateValue(&output[tmpIndex], s, static_cast<T>(0.5) * (input[tmpIndex2] - input[tmpIndex1]));
					}
					int tmpIndex_edge1 = this->index3DtoLinear(i, 0, k);

					this->updateValue(&output[tmpIndex_edge1], s, static_cast<T>(0));

					int tmpIndex_edge2 = this->index3DtoLinear(i, sizeY, k);

					this->updateValue(&output[tmpIndex_edge2], s, static_cast<T>(0));
				}
			}
			*/
		}
	}

	void dzp3d(const Tdata &input, Tdata &output, mySign s)
	{
		int sizeZ = this->inputDimension[2] - 1;
		int sizeY = this->inputDimension[1];
		int sizeX = this->inputDimension[0];

		if(this->type == forward)
		{	
			#pragma omp parallel for
			for (int k = 0; k < sizeZ; ++k)
			{
				for (int j = 0; j < sizeY; ++j)
				{
					for (int i = 0; i < sizeX; ++i)
					{
						const int tmpIndex1 = this->index3DtoLinear(i, j, k);
						const int tmpIndex2 = this->index3DtoLinear(i, j, k + 1);

						this->updateValue(&output[tmpIndex1], s, input[tmpIndex2] - input[tmpIndex1]);
					}
				}
			}
		}
		else if(this->type == central)
		{	
			int sizeZ = this->inputDimension[2];
			int sizeY = this->inputDimension[1];
			int sizeX = this->inputDimension[0];

			#pragma omp parallel for
			for (int i = 0; i < sizeX; ++i)
			{
				for (int j = 0; j < sizeY; ++j)
				{
					for (int k = 1; k < sizeZ-1; ++k)
					{
						const int tmpIndex = this->index3DtoLinear(i, j, k);
						int tmpIndex1 = this->index3DtoLinear(i, j, k+1);
						int tmpIndex2 = this->index3DtoLinear(i, j, k-1);

						switch (s)
						{
							case PLUS:
							{
								output[tmpIndex] += 0.5*(input[tmpIndex1] - input[tmpIndex2]);
								break;
							}
							case MINUS:
							{
								output[tmpIndex] -= 0.5*(input[tmpIndex1] - input[tmpIndex2]);
								break;
							}
							case EQUALS:
							{
								output[tmpIndex] = 0.5*(input[tmpIndex1] - input[tmpIndex2]);
								break;
							}
						}
					}

					int tmpIndex_edge1 = this->index3DtoLinear(i, j, 0);
					int tmpIndex_edge2 = this->index3DtoLinear(i, j, 1);

					switch (s)
						{
							case PLUS:
							{
								output[tmpIndex_edge1] += 0.5*(input[tmpIndex_edge2]-input[tmpIndex_edge1]);
								break;
							}
							case MINUS:
							{
								output[tmpIndex_edge1] -= 0.5*(input[tmpIndex_edge2]-input[tmpIndex_edge1]);
								break;
							}
							case EQUALS:
							{
								output[tmpIndex_edge1] = 0.5*(input[tmpIndex_edge2]-input[tmpIndex_edge1]);
								break;
							}
						}

					int tmpIndex_edge3 = this->index3DtoLinear(i, j, sizeZ-2);
					int tmpIndex_edge4 = this->index3DtoLinear(i, j, sizeZ-1);

					switch (s)
						{
							case PLUS:
							{
								output[tmpIndex_edge4] += 0.5*(input[tmpIndex_edge4]-input[tmpIndex_edge3]);
								break;
							}
							case MINUS:
							{
								output[tmpIndex_edge4] -= 0.5*(input[tmpIndex_edge4]-input[tmpIndex_edge3]);
								break;
							}
							case EQUALS:
							{
								output[tmpIndex_edge4] = 0.5*(input[tmpIndex_edge4]-input[tmpIndex_edge3]);
								break;
							}
						}
					
				}
			}

			/*
			#pragma omp parallel for
			for (int i = 0; i < sizeX; ++i)
			{
				for (int j = 0; j < sizeY; ++j)
				{
					for (int k = 1; k < sizeZ; ++k)
					{
						const int tmpIndex = this->index3DtoLinear(i, j, k);
						const int tmpIndex1 = this->index3DtoLinear(i, j, k - 1);
						const int tmpIndex2 = this->index3DtoLinear(i, j, k + 1);

						this->updateValue(&output[tmpIndex], s, static_cast<T>(0.5) * (input[tmpIndex2] - input[tmpIndex1]));
					}
					int tmpIndex_edge1 = this->index3DtoLinear(i, j, 0);

					this->updateValue(&output[tmpIndex_edge1], s, static_cast<T>(0));

					int tmpIndex_edge2 = this->index3DtoLinear(i, j, sizeZ);

					this->updateValue(&output[tmpIndex_edge2], s, static_cast<T>(0));
				}
			}
			*/
		}
	}

	void dxp3dTransposed(const Tdata &input, Tdata &output, mySign s)
	{
		const int sizeZ = this->inputDimension[2];
		const int sizeY = this->inputDimension[1];
		const int sizeX = this->inputDimension[0] - 1;
		
		if(this->type == forward)
		{
			#pragma omp parallel for
			for (int k = 0; k < sizeZ; ++k)
			{
				for (int j = 0; j < sizeY; ++j)
				{
					for (int i = 1; i < sizeX; ++i)
					{
						int tmpIndex = this->index3DtoLinear(i, j, k);

						this->updateValue(&output[tmpIndex], s, -(input[tmpIndex] - input[tmpIndex - 1]));
					}
				}
			}

			#pragma omp parallel for
			for (int k = 0; k < this->inputDimension[2]; ++k)
			{
				for (int j = 0; j < this->inputDimension[1]; ++j)
				{
					const int index1 = this->index3DtoLinear(0, j, k);
					const int index2 = this->index3DtoLinear(this->inputDimension[0] - 1, j, k);
					const int index3 = this->index3DtoLinear(this->inputDimension[0] - 2, j, k);

					this->updateValue(&output[index1], s, -input[index1]);
					this->updateValue(&output[index2], s, input[index3]);
				}
			}
		}
		
		else if(this->type == central)
		{
			int sizeZ = this->inputDimension[2];
			int sizeY = this->inputDimension[1];
			int sizeX = this->inputDimension[0];

			#pragma omp parallel for
			for (int k = 0; k < sizeZ; ++k)
			{
				for (int j = 0; j < sizeY; ++j)
				{
					for (int i = 1; i < sizeX-1; ++i)
					{
						int tmpIndex = this->index3DtoLinear(i,j,k);
						switch (s)
						{
							case PLUS:
							{
								output[tmpIndex] += static_cast<T>(-0.5) *(input[tmpIndex + 1]-input[tmpIndex - 1]);
								break;
							}
							case MINUS:
							{
								output[tmpIndex] -= static_cast<T>(-0.5) *(input[tmpIndex + 1]-input[tmpIndex - 1]);
								break;
							}
							case EQUALS:
							{
								output[tmpIndex] = static_cast<T>(-0.5) *(input[tmpIndex + 1]-input[tmpIndex - 1]);
								break;
							}
						}
					}
					
					int tmpIndex_edge1 = this->index3DtoLinear(0, j, k);
					int tmpIndex_edge2 = this->index3DtoLinear(1, j, k);

					switch (s)
						{
							case PLUS:
							{
								output[tmpIndex_edge1] += -0.5*(input[tmpIndex_edge2]+input[tmpIndex_edge1]);
								break;
							}
							case MINUS:
							{
								output[tmpIndex_edge1] -= -0.5*(input[tmpIndex_edge2]+input[tmpIndex_edge1]);
								break;
							}
							case EQUALS:
							{
								output[tmpIndex_edge1] = -0.5*(input[tmpIndex_edge2]+input[tmpIndex_edge1]);
								break;
							}
						}

					int tmpIndex_edge3 = this->index3DtoLinear(sizeX-2, j, k);
					int tmpIndex_edge4 = this->index3DtoLinear(sizeX-1, j, k);

					switch (s)
						{
							case PLUS:
							{
								output[tmpIndex_edge4] += 0.5*(input[tmpIndex_edge4]+input[tmpIndex_edge3]);
								break;
							}
							case MINUS:
							{
								output[tmpIndex_edge4] -= 0.5*(input[tmpIndex_edge4]+input[tmpIndex_edge3]);
								break;
							}
							case EQUALS:
							{
								output[tmpIndex_edge4] = 0.5*(input[tmpIndex_edge4]+input[tmpIndex_edge3]);
								break;
							}
						}
				}
			}
			/*
			#pragma omp parallel for
			for (int k = 0; k < sizeZ; ++k)
			{
				for (int j = 0; j < sizeY; ++j)
				{
					for (int i = 1; i < sizeX; ++i)
					{
						int tmpIndex = this->index3DtoLinear(i,j,k);

						this->updateValue(&output[tmpIndex],s,static_cast<T>(-0.5) *(input[tmpIndex + 1]-input[tmpIndex - 1]));
					}

					int tmpIndex_edge1 = this->index3DtoLinear(0, j, k);

					this->updateValue(&output[tmpIndex_edge1], s, static_cast<T>(0));

					int tmpIndex_edge2 = this->index3DtoLinear(sizeX, j, k);

					this->updateValue(&output[tmpIndex_edge2], s, static_cast<T>(0));
				}
			}
			*/
		}
	}

	void dyp3dTransposed(const Tdata &input, Tdata &output, mySign s)
	{
		const int sizeZ = this->inputDimension[2];
		const int sizeY = this->inputDimension[1] - 1;
		const int sizeX = this->inputDimension[0];

		if(this->type == forward)
		{
			#pragma omp parallel for
			for (int k = 0; k < sizeZ; ++k)
			{
				for (int j = 1; j < sizeY; ++j)
				{
					for (int i = 0; i < sizeX; ++i)
					{
						const int tmpIndex1 = this->index3DtoLinear(i, j, k);
						const int tmpIndex2 = this->index3DtoLinear(i, j - 1, k);

						this->updateValue(&output[tmpIndex1], s, -(input[tmpIndex1] - input[tmpIndex2]));
					}
				}
			}

			#pragma omp parallel for
			for (int k = 0; k < this->inputDimension[2]; ++k)
			{
				for (int i = 0; i < this->inputDimension[0]; ++i)
				{
					const int index1 = this->index3DtoLinear(i, 0, k);
					const int index2 = this->index3DtoLinear(i, this->inputDimension[1] - 1, k);
					const int index3 = this->index3DtoLinear(i, this->inputDimension[1] - 2, k);

					this->updateValue(&output[index1], s, -input[index1]);
					this->updateValue(&output[index2], s, input[index3]);
				}
			}
		}
		else if(this->type == central)
		{
			int sizeZ = this->inputDimension[2];
			int sizeY = this->inputDimension[1];
			int sizeX = this->inputDimension[0];

			#pragma omp parallel for
			for (int k = 0; k < sizeZ; ++k)
			{
				for (int i = 0; i < sizeX; ++i)
				{
					for (int j = 1; j < sizeY-1; ++j)
					{
						int tmpIndex = this->index3DtoLinear(i,j,k);
						int tmpIndex1 = this->index3DtoLinear(i,j+1,k);
						int tmpIndex2 = this->index3DtoLinear(i,j-1,k);
						switch (s)
						{
							case PLUS:
							{
								output[tmpIndex] += static_cast<T>(-0.5) *(input[tmpIndex1]-input[tmpIndex2]);
								break;
							}
							case MINUS:
							{
								output[tmpIndex] -= static_cast<T>(-0.5) *(input[tmpIndex1]-input[tmpIndex2]);
								break;
							}
							case EQUALS:
							{
								output[tmpIndex] = static_cast<T>(-0.5) *(input[tmpIndex1]-input[tmpIndex2]);
								break;
							}
						}
					}
					
					int tmpIndex_edge1 = this->index3DtoLinear(i, 0, k);
					int tmpIndex_edge2 = this->index3DtoLinear(i, 1, k);

					switch (s)
						{
							case PLUS:
							{
								output[tmpIndex_edge1] += -0.5*(input[tmpIndex_edge2]+input[tmpIndex_edge1]);
								break;
							}
							case MINUS:
							{
								output[tmpIndex_edge1] -= -0.5*(input[tmpIndex_edge2]+input[tmpIndex_edge1]);
								break;
							}
							case EQUALS:
							{
								output[tmpIndex_edge1] = -0.5*(input[tmpIndex_edge2]+input[tmpIndex_edge1]);
								break;
							}
						}

					int tmpIndex_edge3 = this->index3DtoLinear(i, sizeY-2, k);
					int tmpIndex_edge4 = this->index3DtoLinear(i, sizeY-1, k);

					switch (s)
						{
							case PLUS:
							{
								output[tmpIndex_edge4] += 0.5*(input[tmpIndex_edge4]+input[tmpIndex_edge3]);
								break;
							}
							case MINUS:
							{
								output[tmpIndex_edge4] -= 0.5*(input[tmpIndex_edge4]+input[tmpIndex_edge3]);
								break;
							}
							case EQUALS:
							{
								output[tmpIndex_edge4] = 0.5*(input[tmpIndex_edge4]+input[tmpIndex_edge3]);
								break;
							}
						}
				}
			}



			/*
			#pragma omp parallel for
			for (int k = 0; k < sizeZ; ++k)
			{
				for (int i = 0; i < sizeX; ++i)
				{
					for (int j = 1; j < sizeY; ++j)
					{
						const int tmpIndex = this->index3DtoLinear(i, j, k);
						const int tmpIndex1 = this->index3DtoLinear(i, j - 1, k);
						const int tmpIndex2 = this->index3DtoLinear(i, j + 1, k);

						this->updateValue(&output[tmpIndex], s, static_cast<T>(-0.5) * (input[tmpIndex2] - input[tmpIndex1]));
					}

				int tmpIndex_edge1 = this->index3DtoLinear(i, 0, k);

				this->updateValue(&output[tmpIndex_edge1], s, static_cast<T>(0));

				int tmpIndex_edge2 = this->index3DtoLinear(i, sizeY, k);

				this->updateValue(&output[tmpIndex_edge2], s, static_cast<T>(0));
				}
			}
			*/
		}
	}

	void dzp3dTransposed(const Tdata &input, Tdata &output, mySign s)
	{
		const int sizeZ = this->inputDimension[2] - 1;
		const int sizeY = this->inputDimension[1];
		const int sizeX = this->inputDimension[0];

		if(this->type == forward)
		{
			#pragma omp parallel for
			for (int k = 1; k < sizeZ; ++k)
			{
				for (int j = 0; j < sizeY; ++j)
				{
					for (int i = 0; i < sizeX; ++i)
					{
						const int tmpIndex1 = this->index3DtoLinear(i, j, k);
						const int tmpIndex2 = this->index3DtoLinear(i, j, k - 1);

						this->updateValue(&output[tmpIndex1], s, -(input[tmpIndex1] - input[tmpIndex2]));
					}
				}
			}

			#pragma omp parallel for
			for (int j = 0; j < this->inputDimension[1]; ++j)
			{
				for (int i = 0; i < this->inputDimension[0]; ++i)
				{
					const int index1 = this->index3DtoLinear(i, j, 0);
					const int index2 = this->index3DtoLinear(i, j, this->inputDimension[2] - 1);
					const int index3 = this->index3DtoLinear(i, j, this->inputDimension[2] - 2);

					this->updateValue(&output[index1], s, -input[index1]);
					this->updateValue(&output[index2], s, input[index3]);
				}
			}
		}
		else if(this->type == central)
		{	
			int sizeZ = this->inputDimension[2];
			int sizeY = this->inputDimension[1];
			int sizeX = this->inputDimension[0];

			#pragma omp parallel for
			for (int i = 0; i < sizeX; ++i)
			{
				for (int j = 0; j < sizeY; ++j)
				{
					for (int k = 1; k < sizeZ-1; ++k)
					{
						int tmpIndex = this->index3DtoLinear(i,j,k);
						int tmpIndex1 = this->index3DtoLinear(i,j,k+1);
						int tmpIndex2 = this->index3DtoLinear(i,j,k-1);
						switch (s)
						{
							case PLUS:
							{
								output[tmpIndex] += static_cast<T>(-0.5) *(input[tmpIndex1]-input[tmpIndex2]);
								break;
							}
							case MINUS:
							{
								output[tmpIndex] -= static_cast<T>(-0.5) *(input[tmpIndex1]-input[tmpIndex2]);
								break;
							}
							case EQUALS:
							{
								output[tmpIndex] = static_cast<T>(-0.5) *(input[tmpIndex1]-input[tmpIndex2]);
								break;
							}
						}
					}
					
					int tmpIndex_edge1 = this->index3DtoLinear(i, j, 0);
					int tmpIndex_edge2 = this->index3DtoLinear(i, j, 1);

					switch (s)
						{
							case PLUS:
							{
								output[tmpIndex_edge1] += -0.5*(input[tmpIndex_edge2]+input[tmpIndex_edge1]);
								break;
							}
							case MINUS:
							{
								output[tmpIndex_edge1] -= -0.5*(input[tmpIndex_edge2]+input[tmpIndex_edge1]);
								break;
							}
							case EQUALS:
							{
								output[tmpIndex_edge1] = -0.5*(input[tmpIndex_edge2]+input[tmpIndex_edge1]);
								break;
							}
						}

					int tmpIndex_edge3 = this->index3DtoLinear(i, j, sizeZ-2);
					int tmpIndex_edge4 = this->index3DtoLinear(i, j, sizeZ-1);

					switch (s)
						{
							case PLUS:
							{
								output[tmpIndex_edge4] += 0.5*(input[tmpIndex_edge4]+input[tmpIndex_edge3]);
								break;
							}
							case MINUS:
							{
								output[tmpIndex_edge4] -= 0.5*(input[tmpIndex_edge4]+input[tmpIndex_edge3]);
								break;
							}
							case EQUALS:
							{
								output[tmpIndex_edge4] = 0.5*(input[tmpIndex_edge4]+input[tmpIndex_edge3]);
								break;
							}
						}
				}


			/*
			#pragma omp parallel for
			for (int i = 0; i < sizeX; ++i)
			{
				for (int j = 0; j < sizeY; ++j)
				{
					for (int k = 1; k < sizeZ; ++k)
					{
						const int tmpIndex = this->index3DtoLinear(i, j, k);
						const int tmpIndex1 = this->index3DtoLinear(i, j, k - 1);
						const int tmpIndex2 = this->index3DtoLinear(i, j, k + 1);

						this->updateValue(&output[tmpIndex], s, static_cast<T>(-0.5) * (input[tmpIndex2] - input[tmpIndex1]));
					}
						
					int tmpIndex_edge1 = this->index3DtoLinear(i, j, 0);

					this->updateValue(&output[tmpIndex_edge1], s, static_cast<T>(0));

					int tmpIndex_edge2 = this->index3DtoLinear(i, j, sizeZ);

					this->updateValue(&output[tmpIndex_edge2], s, static_cast<T>(0));
				}
			}
				*/
			}
		}
	}

	//2d cases
	void dxp2d(const Tdata &input, Tdata &output, mySign s)
	{
		if(this->type == forward)
		{
			int sizeY = this->inputDimension[1];
			int sizeX = this->inputDimension[0] - 1;
			
			#pragma omp parallel for
			for (int j = 0; j < sizeY; ++j)
			{
				for (int i = 0; i < sizeX; ++i)
				{
					const int tmpIndex = this->index2DtoLinear(i, j);

					switch (s)
					{
						case PLUS:
						{
							output[tmpIndex] += input[tmpIndex + 1] - input[tmpIndex];
							break;
						}
						case MINUS:
						{
							output[tmpIndex] -= input[tmpIndex + 1] - input[tmpIndex];
							break;
						}
						case EQUALS:
						{
							output[tmpIndex] = input[tmpIndex + 1] - input[tmpIndex];
							break;
						}
					}
				}
			}
		}
		if(this->type == central)
		{
			int sizeY = this->inputDimension[1];
			int sizeX = this->inputDimension[0];

			#pragma omp parallel for
			for (int j = 0; j < sizeY; ++j)
			{
				for (int i = 1; i < sizeX-1; ++i)
				{
					const int tmpIndex = this->index2DtoLinear(i, j);

					switch (s)
					{
						case PLUS:
						{
							output[tmpIndex] += 0.5*(input[tmpIndex + 1] - input[tmpIndex - 1]);
							break;
						}
						case MINUS:
						{
							output[tmpIndex] -= 0.5*(input[tmpIndex + 1] - input[tmpIndex - 1]);
							break;
						}
						case EQUALS:
						{
							output[tmpIndex] = 0.5*(input[tmpIndex + 1] - input[tmpIndex - 1]);
							break;
						}
					}
				}

				int tmpIndex_edge1 = this->index2DtoLinear(0, j);
				int tmpIndex_edge2 = this->index2DtoLinear(1, j);

				switch (s)
					{
						case PLUS:
						{
							output[tmpIndex_edge1] += 0.5*(input[tmpIndex_edge2]-input[tmpIndex_edge1]);
							break;
						}
						case MINUS:
						{
							output[tmpIndex_edge1] -= 0.5*(input[tmpIndex_edge2]-input[tmpIndex_edge1]);
							break;
						}
						case EQUALS:
						{
							output[tmpIndex_edge1] = 0.5*(input[tmpIndex_edge2]-input[tmpIndex_edge1]);
							break;
						}
					}

				int tmpIndex_edge3 = this->index2DtoLinear(sizeX-2, j);
				int tmpIndex_edge4 = this->index2DtoLinear(sizeX-1, j);

				switch (s)
					{
						case PLUS:
						{
							output[tmpIndex_edge4] += 0.5*(input[tmpIndex_edge4]-input[tmpIndex_edge3]);
							break;
						}
						case MINUS:
						{
							output[tmpIndex_edge4] -= 0.5*(input[tmpIndex_edge4]-input[tmpIndex_edge3]);
							break;
						}
						case EQUALS:
						{
							output[tmpIndex_edge4] = 0.5*(input[tmpIndex_edge4]-input[tmpIndex_edge3]);
							break;
						}
					}
			}
		}
	}

	void dyp2d(const Tdata &input, Tdata &output, mySign s)
	{

		if (this->type == forward)
		{
			int sizeY = this->inputDimension[1] - 1;
			int sizeX = this->inputDimension[0];

			#pragma omp parallel for
			for (int j = 0; j < sizeY; ++j)
			{
				for (int i = 0; i < sizeX; ++i)
				{
					const int tmpIndex = this->index2DtoLinear(i, j);

					switch (s)
					{
						case PLUS:
						{
							output[tmpIndex] += input[tmpIndex + sizeX] - input[tmpIndex];
							break;
						}
						case MINUS:
						{
							output[tmpIndex] -= input[tmpIndex + sizeX] - input[tmpIndex];
							break;
						}
						case EQUALS:
						{
							output[tmpIndex] = input[tmpIndex + sizeX] - input[tmpIndex];
							break;
						}
					}
				}
			}
		}
		if(this->type == central)
		{
			int sizeY = this->inputDimension[1];
			int sizeX = this->inputDimension[0];

			#pragma omp parallel for
			for (int i = 0; i < sizeX; ++i)
			{
				for (int j = 1; j < sizeY-1; ++j)
				{
					const int tmpIndex = this->index2DtoLinear(i, j);

					switch (s)
					{
						case PLUS:
						{
							output[tmpIndex] += 0.5*(input[tmpIndex + sizeX] - input[tmpIndex - sizeX]);
							break;
						}
						case MINUS:
						{
							output[tmpIndex] -= 0.5*(input[tmpIndex + sizeX] - input[tmpIndex - sizeX]);
							break;
						}
						case EQUALS:
						{
							output[tmpIndex] = 0.5*(input[tmpIndex + sizeX] - input[tmpIndex - sizeX]);
							break;
						}
					}
				}

				int tmpIndex_edge1 = this->index2DtoLinear(i, 0);
				int tmpIndex_edge2 = this->index2DtoLinear(i, 1);

				switch (s)
					{
						case PLUS:
						{
							output[tmpIndex_edge1] += 0.5*(input[tmpIndex_edge2]-input[tmpIndex_edge1]);
							break;
						}
						case MINUS:
						{
							output[tmpIndex_edge1] -= 0.5*(input[tmpIndex_edge2]-input[tmpIndex_edge1]);
							break;
						}
						case EQUALS:
						{
							output[tmpIndex_edge1] = 0.5*(input[tmpIndex_edge2]-input[tmpIndex_edge1]);
							break;
						}
					}

				int tmpIndex_edge3 = this->index2DtoLinear(i, sizeY-2);
				int tmpIndex_edge4 = this->index2DtoLinear(i, sizeY-1);

				switch (s)
					{
						case PLUS:
						{
							output[tmpIndex_edge4] += 0.5*(input[tmpIndex_edge4]-input[tmpIndex_edge3]);
							break;
						}
						case MINUS:
						{
							output[tmpIndex_edge4] -= 0.5*(input[tmpIndex_edge4]-input[tmpIndex_edge3]);
							break;
						}
						case EQUALS:
						{
							output[tmpIndex_edge4] = 0.5*(input[tmpIndex_edge4]-input[tmpIndex_edge3]);
							break;
						}
					}
			}
		}
	}

	void dxp2dTransposed(const Tdata &input, Tdata &output, mySign s)
	{
		

		if(this->type == forward)
		{
			int sizeY = this->inputDimension[1];
			int sizeX = this->inputDimension[0] - 1;

			#pragma omp parallel for
			for (int j = 0; j < sizeY; ++j)
			{
				for (int i = 1; i < sizeX; ++i)
				{
					int tmpIndex = this->index2DtoLinear(i, j);

					switch (s)
					{
						case PLUS:
						{
							output[tmpIndex] += -(input[tmpIndex] - input[tmpIndex - 1]);
							break;
						}
						case MINUS:
						{
							output[tmpIndex] -= -(input[tmpIndex] - input[tmpIndex - 1]);
							break;
						}
						case EQUALS:
						{
							output[tmpIndex] = -(input[tmpIndex] - input[tmpIndex - 1]);
							break;
						}
					}
				}
			}

			for (int j = 0; j < this->inputDimension[1]; ++j)
			{
				switch (s)
				{
					case PLUS:
					{
						output[this->index2DtoLinear(0, j)] += -input[this->index2DtoLinear(0, j)];
						output[this->index2DtoLinear(this->inputDimension[0] - 1, j)] += input[this->index2DtoLinear(this->inputDimension[0] - 2, j)];
						break;
					}
					case MINUS:
					{
						output[this->index2DtoLinear(0, j)] -= -input[this->index2DtoLinear(0, j)];
						output[this->index2DtoLinear(this->inputDimension[0] - 1, j)] -= input[this->index2DtoLinear(this->inputDimension[0] - 2, j)];
						break;
					}
					case EQUALS:
					{
						output[this->index2DtoLinear(0, j)] = -input[this->index2DtoLinear(0, j)];
						output[this->index2DtoLinear(this->inputDimension[0] - 1, j)] = input[this->index2DtoLinear(this->inputDimension[0] - 2, j)];
						break;
					}
				}
			}
		}
		else if(this->type == central)
		{
			int sizeY = this->inputDimension[1];
			int sizeX = this->inputDimension[0];

			#pragma omp parallel for
			for (int j = 0; j < sizeY; ++j)
			{
				for (int i = 1; i < sizeX-1; ++i)
				{
					int tmpIndex = this->index2DtoLinear(i,j);
					switch (s)
					{
						case PLUS:
						{
							output[tmpIndex] += static_cast<T>(-0.5) *(input[tmpIndex + 1]-input[tmpIndex - 1]);
							break;
						}
						case MINUS:
						{
							output[tmpIndex] -= static_cast<T>(-0.5) *(input[tmpIndex + 1]-input[tmpIndex - 1]);
							break;
						}
						case EQUALS:
						{
							output[tmpIndex] = static_cast<T>(-0.5) *(input[tmpIndex + 1]-input[tmpIndex - 1]);
							break;
						}
					}
				}
				
				int tmpIndex_edge1 = this->index2DtoLinear(0, j);
				int tmpIndex_edge2 = this->index2DtoLinear(1, j);

				switch (s)
					{
						case PLUS:
						{
							output[tmpIndex_edge1] += -0.5*(input[tmpIndex_edge2]+input[tmpIndex_edge1]);
							break;
						}
						case MINUS:
						{
							output[tmpIndex_edge1] -= -0.5*(input[tmpIndex_edge2]+input[tmpIndex_edge1]);
							break;
						}
						case EQUALS:
						{
							output[tmpIndex_edge1] = -0.5*(input[tmpIndex_edge2]+input[tmpIndex_edge1]);
							break;
						}
					}

				int tmpIndex_edge3 = this->index2DtoLinear(sizeX-2, j);
				int tmpIndex_edge4 = this->index2DtoLinear(sizeX-1, j);

				switch (s)
					{
						case PLUS:
						{
							output[tmpIndex_edge4] += 0.5*(input[tmpIndex_edge4]+input[tmpIndex_edge3]);
							break;
						}
						case MINUS:
						{
							output[tmpIndex_edge4] -= 0.5*(input[tmpIndex_edge4]+input[tmpIndex_edge3]);
							break;
						}
						case EQUALS:
						{
							output[tmpIndex_edge4] = 0.5*(input[tmpIndex_edge4]+input[tmpIndex_edge3]);
							break;
						}
					}
			}
		}
	}

	void dyp2dTransposed(const Tdata &input, Tdata &output, mySign s)
	{
		
		if(this->type == forward)
		{
			int sizeY = this->inputDimension[1] - 1;
			int sizeX = this->inputDimension[0];


			#pragma omp parallel for
			for (int j = 1; j < sizeY; ++j)
			{
				for (int i = 0; i < sizeX; ++i)
				{
					int tmpIndex = this->index2DtoLinear(i, j);

					switch (s)
					{
						case PLUS:
						{
							output[tmpIndex] += -(input[tmpIndex] - input[tmpIndex - sizeX]);
							break;
						}
						case MINUS:
						{
							output[tmpIndex] -= -(input[tmpIndex] - input[tmpIndex - sizeX]);
							break;
						}
						case EQUALS:
						{
							output[tmpIndex] = -(input[tmpIndex] - input[tmpIndex - sizeX]);
							break;
						}
					}
				}
			}

			for (int i = 0; i < this->inputDimension[0]; ++i)
			{
				switch (s)
				{
					case PLUS:
					{
						output[this->index2DtoLinear(i, 0)] += -input[this->index2DtoLinear(i, 0)];
						output[this->index2DtoLinear(i, this->inputDimension[1] - 1)] += input[this->index2DtoLinear(i, this->inputDimension[1] - 2)];
						break;
					}
					case MINUS:
					{
						output[this->index2DtoLinear(i, 0)] -= -input[this->index2DtoLinear(i, 0)];
						output[this->index2DtoLinear(i, this->inputDimension[1] - 1)] -= input[this->index2DtoLinear(i, this->inputDimension[1] - 2)];
						break;
					}
					case EQUALS:
					{
						output[this->index2DtoLinear(i, 0)] = -input[this->index2DtoLinear(i, 0)];
						output[this->index2DtoLinear(i, this->inputDimension[1] - 1)] = input[this->index2DtoLinear(i, this->inputDimension[1] - 2)];
						break;
					}
				}
			}
		}
		else if(this->type == central)
		{
			int sizeY = this->inputDimension[1];
			int sizeX = this->inputDimension[0];

			#pragma omp parallel for
			for (int i = 0; i < sizeX; ++i)
			{
				for (int j = 1; j < sizeY-1; ++j)
				{
					int tmpIndex = this->index2DtoLinear(i,j);
					int tmpIndex1 = this->index2DtoLinear(i,j-1);
					int tmpIndex2 = this->index2DtoLinear(i,j+1);

					switch (s)
					{
						case PLUS:
						{
							output[tmpIndex] += static_cast<T>(-0.5) *(input[tmpIndex2]-input[tmpIndex1]);
							break;
						}
						case MINUS:
						{
							output[tmpIndex] -= static_cast<T>(-0.5) *(input[tmpIndex2]-input[tmpIndex1]);
							break;
						}
						case EQUALS:
						{
							output[tmpIndex] = static_cast<T>(-0.5) *(input[tmpIndex2]-input[tmpIndex1]);
							break;
						}
					}
				}
				
				int tmpIndex_edge1 = this->index2DtoLinear(i, 0);
				int tmpIndex_edge2 = this->index2DtoLinear(i, 1);

				switch (s)
					{
						case PLUS:
						{
							output[tmpIndex_edge1] += -0.5*(input[tmpIndex_edge2]+input[tmpIndex_edge1]);
							break;
						}
						case MINUS:
						{
							output[tmpIndex_edge1] -= -0.5*(input[tmpIndex_edge2]+input[tmpIndex_edge1]);
							break;
						}
						case EQUALS:
						{
							output[tmpIndex_edge1] = -0.5*(input[tmpIndex_edge2]+input[tmpIndex_edge1]);
							break;
						}
					}

				int tmpIndex_edge3 = this->index2DtoLinear(i, sizeY-2);
				int tmpIndex_edge4 = this->index2DtoLinear(i, sizeY-1);

				switch (s)
					{
						case PLUS:
						{
							output[tmpIndex_edge4] += 0.5*(input[tmpIndex_edge4]+input[tmpIndex_edge3]);
							break;
						}
						case MINUS:
						{
							output[tmpIndex_edge4] -= 0.5*(input[tmpIndex_edge4]+input[tmpIndex_edge3]);
							break;
						}
						case EQUALS:
						{
							output[tmpIndex_edge4] = 0.5*(input[tmpIndex_edge4]+input[tmpIndex_edge3]);
							break;
						}
					}	
			}
		}
	}

	//have not checked if the upwind methods do sensible things on the edges
	float dxUpwind(const float *data,const float *v,  const size_t *sizeMat, int i, int j, int k)
	{
	if (k < sizeMat[2] - 1)
	{
		if (v[index3DtoLinear(sizeMat, i, j , k)]<0)
		{
			if (i < sizeMat[0] - 1)
			{
				return v[index3DtoLinear(sizeMat, i, j , k)]*(data[index3DtoLinear(sizeMat, i + 1, j , k)] - data[index3DtoLinear(sizeMat, i, j , k)]);
			}
			else
			{
				return 0.0f;
			}
		}
		else
		{
			if (i > 0)
			{
				return v[index3DtoLinear(sizeMat, i, j , k)]*(data[index3DtoLinear(sizeMat, i, j , k)] - data[index3DtoLinear(sizeMat, i - 1, j , k)]);
			}
			else
			{
				return 0.0f;
			}
		}
	}
	return 0.0f;
	}

	float dyUpwind(const float *data,const float *v, const size_t *sizeMat, int i, int j, int k)
	{
	if (k < sizeMat[2] - 1)
	{
		if (v[index3DtoLinear(sizeMat, i, j , k)]<0)
		{
			if (j < sizeMat[1] - 1)
			{
				return v[index3DtoLinear(sizeMat, i, j , k)]*(data[index3DtoLinear(sizeMat, i, j + 1, k)] - data[index3DtoLinear(sizeMat, i, j , k)]);
			}
			else
			{
				return 0.0f;
			}
		}
		else
		{
			if (j > 0)
			{
				return v[index3DtoLinear(sizeMat, i, j , k)]*(data[index3DtoLinear(sizeMat, i, j , k)] - data[index3DtoLinear(sizeMat, i , j - 1 , k)]);
			}
			else
			{
				return 0.0f;
			}
		}
	}
	return 0.0f;
	}

	float dxUpwindT(const float *data,const float *v,  const size_t *sizeMat, int i, int j, int k)
	{
	//int indexIJK = index3DtoLinear(sizeMat, i, j , k);

	if (k < sizeMat[2] - 1)
	{
		if (i == 0)
		{
			if (v[index3DtoLinear(sizeMat, i, j , k)] < 0)
			{
				return -v[index3DtoLinear(sizeMat, i, j , k)]*data[index3DtoLinear(sizeMat, i, j , k)];
			}
		}
		else if (i == sizeMat[0] - 1)
		{
			if (v[index3DtoLinear(sizeMat, i, j , k)] > 0)
			{
				return v[index3DtoLinear(sizeMat, i, j , k)]*data[index3DtoLinear(sizeMat, i, j , k)];
			}
		}
		else
		{
			float tmp = 0.0f;
			
			if (v[index3DtoLinear(sizeMat, i, j , k)] < 0)
			{
				tmp -= v[index3DtoLinear(sizeMat, i, j , k)]*data[index3DtoLinear(sizeMat, i, j , k)];
			}
			else
			{
				tmp += v[index3DtoLinear(sizeMat, i, j , k)]*data[index3DtoLinear(sizeMat, i, j , k)];
			}
			
			if (v[index3DtoLinear(sizeMat, i - 1, j , k)] < 0)
			{
				tmp += v[index3DtoLinear(sizeMat, i - 1, j , k)]*data[index3DtoLinear(sizeMat, i - 1, j , k)];
			}
			if (v[index3DtoLinear(sizeMat, i + 1, j , k)] > 0)
			{
				tmp -= v[index3DtoLinear(sizeMat, i + 1, j , k)]*data[index3DtoLinear(sizeMat, i + 1, j , k)];
			}
			return tmp;
		}
	}
	return 0.0f;
	}

	float dyUpwindT(const float *data,const float *v, const size_t *sizeMat, int i, int j, int k)
	{
	//int indexIJK = index3DtoLinear(sizeMat, i, j , k);

	if (k < sizeMat[2] - 1)
	{
		if (j == 0)
		{
			if (v[index3DtoLinear(sizeMat, i, j , k)] < 0)
			{
				return -v[index3DtoLinear(sizeMat, i, j , k)]*data[index3DtoLinear(sizeMat, i, j , k)];
			}
		}
		else if (j == sizeMat[1] - 1)
		{
			if (v[index3DtoLinear(sizeMat, i, j , k)] > 0)
			{
				return v[index3DtoLinear(sizeMat, i, j , k)]*data[index3DtoLinear(sizeMat, i, j , k)];
			}
		}
		else
		{
			float tmp = 0.0f;
			
			if (v[index3DtoLinear(sizeMat, i, j , k)] < 0)
			{
				tmp -= v[index3DtoLinear(sizeMat, i, j , k)]*data[index3DtoLinear(sizeMat, i, j , k)];
			}
			else
			{
				tmp += v[index3DtoLinear(sizeMat, i, j , k)]*data[index3DtoLinear(sizeMat, i, j , k)];
			}
			
			if (v[index3DtoLinear(sizeMat, i , j - 1 , k)] < 0)
			{
				tmp += v[index3DtoLinear(sizeMat, i , j - 1 , k)]*data[index3DtoLinear(sizeMat, i , j - 1 , k)];
			}
			if (v[index3DtoLinear(sizeMat, i , j + 1 , k)] > 0)
			{
				tmp -= v[index3DtoLinear(sizeMat, i , j + 1 , k)]*data[index3DtoLinear(sizeMat, i , j + 1 , k)];
			}
			return tmp;
		}
	}
	return 0.0f;
	}


	void doTimesCPU(bool transposed, const Tdata &input, Tdata &output, mySign s)
	{
		if (this->inputDimension.size() == 2)
		{
			if (this->gradDirection == 0)
			{
				if (transposed == false)
				{
					this->dxp2d(input, output, s);
				}
				else
				{
					this->dxp2dTransposed(input, output, s);
				}
			}
			else if (this->gradDirection == 1)
			{
				if (transposed == false)
				{
					this->dyp2d(input, output, s);
				}
				else
				{
					this->dyp2dTransposed(input, output, s);
				}
			}
		}
		else if (this->inputDimension.size() == 3)
		{
			if (this->gradDirection == 0)
			{
				if (transposed == false)
				{
					this->dxp3d(input, output, s);
				}
				else
				{
					this->dxp3dTransposed(input, output, s);
				}
			}
			else if (this->gradDirection == 1)
			{
				if (transposed == false)
				{
					this->dyp3d(input, output, s);
				}
				else
				{
					this->dyp3dTransposed(input, output, s);
				}
			}
			else if (this->gradDirection == 2)
			{
				if (transposed == false)
				{
					this->dzp3d(input, output, s);
				}
				else
				{
					this->dzp3dTransposed(input, output, s);
				}
			}
		}
		else
		{
			printf("Gradient not implemented for dim!={2,3}\n");
			//TODO: implement gradient for dim!={2,3} for CPU version
		}
	}

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
					dxp2dCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s);
				}
				else
				{
					dxp2dTransposedCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s);
				}
			}
			else if (this->gradDirection == 1)
			{
				if (transposed == false)
				{
					dyp2dCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s);
				}
				else
				{
					dyp2dTransposedCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s);
				}
			}
		}
		else if (this->inputDimension.size() == 3)
		{
			if (this->gradDirection == 0)
			{
				if (transposed == false)
				{
					dxp3dCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], this->inputDimension[2], s);
				}
				else
				{
					dxp3dTransposedCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], this->inputDimension[2], s);
				}
			}
			else if (this->gradDirection == 1)
			{
				if (transposed == false)
				{
					dyp3dCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], this->inputDimension[2], s);
				}
				else
				{
					dyp3dTransposedCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], this->inputDimension[2], s);
				}
			}
			else if (this->gradDirection == 2)
			{
				if (transposed == false)
				{
					dzp3dCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], this->inputDimension[2], s);
				}
				else
				{
					dzp3dTransposedCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], this->inputDimension[2], s);
				}
			}
		}
		else
		{
			printf("Gradient not implemented for dim!={2,3}\n");
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

		// flip sign and transposed
		if (this->type == backward)
		{
			transposed = !transposed;

			if (s == PLUS)
				s = MINUS;
			else if (s == MINUS)
				s = PLUS;
		}

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
		if(type==forward||type==backward)
		{
			return static_cast<T>(2);
		}
		else if(type==central)
		{
			return static_cast<T>(1);
		}
	}

	std::vector<T> getAbsRowSum(bool transposed)
	{	
		if(type==forward||type==backward)
		{
			std::vector<T> result(this->getNumRows(),(T)2);
			return result;
		}
		else if(type==central)
		{
			std::vector<T> result(this->getNumRows(),(T)1);
			return result;
		}
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
		if(type==forward||type==backward)
		{
			thrust::device_vector<T> result(this->getNumRows(), (T)2);
		}
		else if(type==central)
		{
			thrust::device_vector<T> result(this->getNumRows(), (T)1);
		}

		return result;
	}
	#endif
};

#endif
