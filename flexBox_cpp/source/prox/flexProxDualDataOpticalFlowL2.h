#ifndef flexProxDualDataOpticalFlowL2_H
#define flexProxDualDataOpticalFlowL2_H

#include "flexProx.h"
//STill TODO!!!!!!
//! represents prox for a L2 optical flow term with v being the desired primal variable by using the flexProxDualDataL2
/*!
	\f$ \frac{1}{2}\|\u_t - grad(u)*\cdot\|_2^2 \f$
*/
template<typename T>
class flexProxDualDataOpticalFlowL2 : public flexProx<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	Tdata image1;
	Tdata image2;
	Tdata uT; //time derivative
	//std::vector<Tdata> gradients;//gradients[i] will hold the derivatives with respect to the i_th direction. i will be typically 2 or 3
	T timestep;
public:

	flexProxDualDataOpticalFlowL2(Tdata &image1, Tdata &image2,  T timeStep) :image1(image1), image2(image2), uT(), flexProx<T>(dualOpticalFlowL2DataProx)
	{
		uT.resize(image1.size());
	}

	~flexProxDualDataOpticalFlowL2()
	{
		if (VERBOSE > 0) printf("Destructor prox\n!");
	}

	//compute negative time derivative of u for the L2Data term to have \cdot -(-u_t)
	void computeNegUT(Tdata &image1, Tdata &image2, Tdata &uT, T &timestep)
	{
		for (int i = 0; i<uT.size(); i++)
		{
			uT[i] = -(image2[i]-image1[i])/timestep;
		}
	}

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers)
	{

	}

    #ifdef __CUDACC__
        struct flexProxDualDataOpticalFlowL2Functor
        {
            __host__ __device__
            flexProxDualDataL2Functor(T _alpha) : alpha(_alpha){};

            template <typename Tuple>
            __host__ __device__
            void operator()(Tuple t)
            {
                thrust::get<0>(t) = alpha / (thrust::get<2>(t) + alpha) * (thrust::get<1>(t) - thrust::get<2>(t) * thrust::get<3>(t));
            }

            const T alpha;
        };
    #endif

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers, Tdata &image1, Tdata &image2, T timeStep)
	{
		flexProxDualDataL2<T> L2DataProx();

		//calculate -u_t to give it in as the data term for L2DataProx
		computeNegUT(image1, image2, uT, timestep);
		
		L2DataProx.applyProx(alpha, data, dualNumbers, primalNumbers, uT);
	}
};

#endif