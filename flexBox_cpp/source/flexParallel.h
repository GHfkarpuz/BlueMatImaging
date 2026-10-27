#ifndef flexParallel_H
#define flexParallel_H

// Helpers to write an element-wise loop once and run it either on the CPU (OpenMP) or on the GPU (CUDA/thrust).
// A loop body is a functor with "FLEX_HD void operator()(int i) const" that only uses raw pointers and plain values.

#include <vector>
#include <cstddef>
#include "tools.h"

#ifdef __CUDACC__
	#include <thrust/device_vector.h>
	#include <thrust/for_each.h>
	#include <thrust/iterator/counting_iterator.h>
	#define FLEX_HD __host__ __device__
#else
	#define FLEX_HD
#endif

//! runs f(i) for i = 0..n-1 in parallel (OpenMP threads on the CPU, one CUDA thread per element on the GPU)
template <typename F>
void flexForEach(int n, const F& f)
{
#ifdef __CUDACC__
	thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n), f);
#else
	#pragma omp parallel for schedule(static) if(flexUseThreads(n))
	for (int i = 0; i < n; ++i)
	{
		f(i);
	}
#endif
}

//! raw pointer to the data of a host or device vector
template <typename T>
T* flexPtr(std::vector<T>& v) { return v.data(); }
template <typename T>
const T* flexPtr(const std::vector<T>& v) { return v.data(); }
#ifdef __CUDACC__
template <typename T>
T* flexPtr(thrust::device_vector<T>& v) { return thrust::raw_pointer_cast(v.data()); }
template <typename T>
const T* flexPtr(const thrust::device_vector<T>& v) { return thrust::raw_pointer_cast(v.data()); }
#endif

//! writes val into out according to s (out = val, out += val, out -= val)
template <typename T>
FLEX_HD inline void flexApplySign(T& out, const T val, const mySign s)
{
	if (s == PLUS) out += val;
	else if (s == MINUS) out -= val;
	else out = val;
}

//! flips PLUS and MINUS (used for negated operators), EQUALS stays
inline mySign flexFlipSign(const mySign s)
{
	if (s == PLUS) return MINUS;
	if (s == MINUS) return PLUS;
	return s;
}

template <typename T>
FLEX_HD inline T flexAbsHD(const T x) { return x < T(0) ? -x : x; }

//! geometry of a derivative along one axis of a column-major (Fortran ordered) array
struct flexAxis
{
	int stride;	//!< distance between neighbours along the axis
	int n;		//!< number of elements along the axis

	//! position along the axis of the linear index idx
	FLEX_HD int coord(const int idx) const { return (idx / stride) % n; }
};

inline flexAxis flexMakeAxis(const std::vector<int>& dims, const int direction)
{
	flexAxis a;
	a.stride = 1;
	for (int d = 0; d < direction; ++d)
		a.stride *= dims[d];
	a.n = direction < (int)dims.size() ? dims[direction] : 1;
	return a;
}

//! runs f.apply(i, c) for all linear indices i, where c is the position of i along the axis ax
/*!
	On the GPU every thread computes c itself (one integer division). On the CPU the loops are ordered
	so that c is known without divisions and the innermost loop runs over contiguous memory.
	f has to provide "FLEX_HD void apply(int i, int c) const" and "FLEX_HD void operator()(int i) const".
*/
template <typename F>
void flexForEachAxis(const flexAxis& ax, int n, const F& f)
{
#ifdef __CUDACC__
	flexForEach(n, f);
#else
	const int st = ax.stride;
	const int len = ax.n;
	const int outer = n / (st * len);

	if (st == 1)
	{
		// derivative along the first (contiguous) axis: one line per iteration
		#pragma omp parallel for schedule(static) if(flexUseThreads(n))
		for (int b = 0; b < outer; ++b)
		{
			const int base = b * len;
			for (int c = 0; c < len; ++c)
				f.apply(base + c, c);
		}
	}
	else
	{
		#pragma omp parallel for collapse(2) schedule(static) if(flexUseThreads(n))
		for (int b = 0; b < outer; ++b)
		{
			for (int c = 0; c < len; ++c)
			{
				const int base = (b * len + c) * st;
				for (int a = 0; a < st; ++a)
					f.apply(base + a, c);
			}
		}
	}
#endif
}

// ---- generic element-wise kernels ------------------------------------------------------------

//! out (s)= a[i] * b[i]
template <typename T>
struct flexMulFunctor
{
	T* out; const T* a; const T* b; mySign s;
	FLEX_HD void operator()(const int i) const { flexApplySign(out[i], a[i] * b[i], s); }
};

//! out (s)= a[i]
template <typename T>
struct flexCopyFunctor
{
	T* out; const T* a; mySign s;
	FLEX_HD void operator()(const int i) const { flexApplySign(out[i], a[i], s); }
};

//! out[i] = |a[i]| * b[i]
template <typename T>
struct flexAbsMulFunctor
{
	T* out; const T* a; const T* b;
	FLEX_HD void operator()(const int i) const { out[i] = flexAbsHD(a[i]) * b[i]; }
};

//! out[i] = c * b[i]
template <typename T>
struct flexScaleFunctor
{
	T* out; const T* b; T c;
	FLEX_HD void operator()(const int i) const { out[i] = c * b[i]; }
};

template <typename V>
void flexMul(V& out, const V& a, const V& b, const mySign s)
{
	typedef typename V::value_type T;
	flexMulFunctor<T> f = { flexPtr(out), flexPtr(a), flexPtr(b), s };
	flexForEach((int)out.size(), f);
}

template <typename V>
void flexCopy(V& out, const V& a, const mySign s)
{
	typedef typename V::value_type T;
	flexCopyFunctor<T> f = { flexPtr(out), flexPtr(a), s };
	flexForEach((int)out.size(), f);
}

//! copies a host vector into a host or device vector
template <typename T>
void flexAssign(std::vector<T>& out, const std::vector<T>& in) { out = in; }
#ifdef __CUDACC__
template <typename T>
void flexAssign(thrust::device_vector<T>& out, const std::vector<T>& in) { out = in; }
#endif

//! copies a host or device vector into a host vector
template <typename T>
std::vector<T> flexToHost(const std::vector<T>& in) { return in; }
#ifdef __CUDACC__
template <typename T>
std::vector<T> flexToHost(const thrust::device_vector<T>& in)
{
	std::vector<T> out(in.size());
	thrust::copy(in.begin(), in.end(), out.begin());
	return out;
}
#endif

#endif
