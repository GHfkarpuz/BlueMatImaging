#ifndef flexSparseMatrixOperator_H
#define flexSparseMatrixOperator_H

#include <vector>
#include <cmath>
#include "flexLinearOperator.h"

#ifdef __CUDACC__
	#include <thrust/iterator/counting_iterator.h>
	#include <thrust/for_each.h>
#endif

//! represents a general sparse matrix in compressed row storage (CSR), on the CPU and with CUDA
/*!
	the transposed matrix is stored as a second CSR matrix, so both products are row-wise (no atomics on the GPU);
	the absolute row sums of A and A^T are computed once in the constructor (diagonal preconditioning)
*/
template <typename T>
class flexSparseMatrixOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
	typedef thrust::device_vector<int> Idata;
#else
	typedef std::vector<T> Tdata;
	typedef std::vector<int> Idata;
#endif

private:
	Idata rowPtr, colIdx, rowPtrT, colIdxT;
	Tdata values, valuesT;
	std::vector<T> absRowSum, absRowSumT;

	template <typename V, typename W>
	static void assign(V& target, const std::vector<W>& source)
	{
		target.resize(source.size());
		#ifdef __CUDACC__
			thrust::copy(source.begin(), source.end(), target.begin());
		#else
			std::copy(source.begin(), source.end(), target.begin());
		#endif
	}

	flexSparseMatrixOperator(int aNumRows, int aNumCols, bool aMinus) : flexLinearOperator<T>(aNumRows, aNumCols, matrixOp, aMinus) {}

public:
	//! initializes the operator from CSR arrays (0-based)
	/*!
		\param aNumRows number of rows
		\param aNumCols number of columns
		\param aRowPtr row pointer (aNumRows + 1 entries)
		\param aColIdx column index of every nonzero
		\param aValues value of every nonzero
		\param aMinus determines if operator is negated \sa isMinus
	*/
	flexSparseMatrixOperator(int aNumRows, int aNumCols, const std::vector<int>& aRowPtr, const std::vector<int>& aColIdx,
	                         const std::vector<T>& aValues, bool aMinus)
	: flexLinearOperator<T>(aNumRows, aNumCols, matrixOp, aMinus)
	{
		const int nnz = aRowPtr[aNumRows];

		// transposed CSR (= CSC of A) by counting sort over the columns
		std::vector<int> rpT(aNumCols + 1, 0), ciT(nnz);
		std::vector<T> vT(nnz);
		for (int k = 0; k < nnz; ++k) rpT[aColIdx[k] + 1]++;
		for (int j = 0; j < aNumCols; ++j) rpT[j + 1] += rpT[j];
		std::vector<int> next(rpT.begin(), rpT.end() - 1);
		for (int i = 0; i < aNumRows; ++i)
			for (int k = aRowPtr[i]; k < aRowPtr[i + 1]; ++k)
			{
				const int pos = next[aColIdx[k]]++;
				ciT[pos] = i;
				vT[pos] = aValues[k];
			}

		absRowSum.assign(aNumRows, static_cast<T>(0));
		absRowSumT.assign(aNumCols, static_cast<T>(0));
		for (int i = 0; i < aNumRows; ++i)
			for (int k = aRowPtr[i]; k < aRowPtr[i + 1]; ++k)
			{
				absRowSum[i] += std::abs(aValues[k]);
				absRowSumT[aColIdx[k]] += std::abs(aValues[k]);
			}

		assign(rowPtr, aRowPtr); assign(colIdx, aColIdx); assign(values, aValues);
		assign(rowPtrT, rpT); assign(colIdxT, ciT); assign(valuesT, vT);
	}

	flexSparseMatrixOperator<T>* copy()
	{
		auto A = new flexSparseMatrixOperator<T>(this->getNumRows(), this->getNumCols(), this->isMinus);
		A->rowPtr = rowPtr; A->colIdx = colIdx; A->values = values;
		A->rowPtrT = rowPtrT; A->colIdxT = colIdxT; A->valuesT = valuesT;
		A->absRowSum = absRowSum; A->absRowSumT = absRowSumT;
		return A;
	}

	#ifdef __CUDACC__
	struct flexSparseMatrixFunctor
	{
		const int* rp; const int* ci; const T* val; const T* in; T* out; mySign s;

		__host__ __device__
		flexSparseMatrixFunctor(const int* _rp, const int* _ci, const T* _val, const T* _in, T* _out, mySign _s)
		: rp(_rp), ci(_ci), val(_val), in(_in), out(_out), s(_s) {}

		__host__ __device__
		void operator()(int i) const
		{
			T sum = (T)0;
			for (int k = rp[i]; k < rp[i + 1]; ++k) sum += val[k] * in[ci[k]];
			switch (s)
			{
				case PLUS: out[i] += sum; break;
				case MINUS: out[i] -= sum; break;
				case EQUALS: out[i] = sum; break;
			}
		}
	};
	#endif

	void doTimes(bool transposed, const Tdata &input, Tdata &output, const mySign s)
	{
		const Idata& rp = transposed ? rowPtrT : rowPtr;
		const Idata& ci = transposed ? colIdxT : colIdx;
		const Tdata& val = transposed ? valuesT : values;
		const int n = transposed ? this->getNumCols() : this->getNumRows();
		#ifdef __CUDACC__
			thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n),
				flexSparseMatrixFunctor(thrust::raw_pointer_cast(rp.data()), thrust::raw_pointer_cast(ci.data()),
				                        thrust::raw_pointer_cast(val.data()), thrust::raw_pointer_cast(input.data()),
				                        thrust::raw_pointer_cast(output.data()), s));
		#else
			#pragma omp parallel for schedule(static) if(flexUseThreads(n))
			for (int i = 0; i < n; ++i)
			{
				T sum = static_cast<T>(0);
				for (int k = rp[i]; k < rp[i + 1]; ++k) sum += val[k] * input[ci[k]];
				switch (s)
				{
					case PLUS: output[i] += sum; break;
					case MINUS: output[i] -= sum; break;
					case EQUALS: output[i] = sum; break;
				}
			}
		#endif
	}

	void times(bool transposed, const Tdata &input, Tdata &output)
	{
		if (this->isMinus)
		{
			this->doTimes(transposed, input, output, EQUALS);
			vectorScalarProduct(output, static_cast<T>(-1));
		}
		else
		{
			this->doTimes(transposed, input, output, EQUALS);
		}
	}

	void timesPlus(bool transposed, const Tdata &input, Tdata &output)
	{
		this->doTimes(transposed, input, output, this->isMinus ? MINUS : PLUS);
	}

	void timesMinus(bool transposed, const Tdata &input, Tdata &output)
	{
		this->doTimes(transposed, input, output, this->isMinus ? PLUS : MINUS);
	}

	std::vector<T> getAbsRowSum(bool transposed)
	{
		return transposed ? absRowSumT : absRowSum;
	}

	T getMaxRowSumAbs(bool transposed)
	{
		const std::vector<T>& r = transposed ? absRowSumT : absRowSum;
		T m = static_cast<T>(0);
		for (const T& x : r) m = std::max(m, x);
		return m;
	}

	#ifdef __CUDACC__
	thrust::device_vector<T> getAbsRowSumCUDA(bool transposed)
	{
		const std::vector<T>& r = transposed ? absRowSumT : absRowSum;
		thrust::device_vector<T> result(r.begin(), r.end());
		return result;
	}
	#endif
};

#endif
