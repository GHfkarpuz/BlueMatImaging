/*
#ifndef flexProxDualDataOpticalFlowL2_H
#define flexProxDualDataOpticalFlowL2_H

#include "flexProx.h"
//STill TODO!!!!!!
//! represents prox for a L2 optical flow term with v being the desired primal variable
*/
/*!
	\f$ \frac{1}{2}\|\u_t - grad(u)*\cdot\|_2^2 \f$
*/
/*
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
	std::vector<Tdata> gradients;//gradients[i] will hold the derivatives with respect to the i_th direction. i will be typically 2 or 3
	T timestep;
public:

	flexProxDualDataOpticalFlowL2(Tdata &image1, Tdata &image2, std::vector<Tdata> gradients, T timeStep) :image1(image1), image2(image2), uT(), gradients(gradients), flexProx<T>(dualOpticalFlowL2DataProx)
	{
		uT.resize(image1.size());
	}

	~flexProxDualDataOpticalFlowL2()
	{
		if (VERBOSE > 0) printf("Destructor prox\n!");
	}

	void computeUT(Tdata &image1, Tdata &image2, Tdata &uT, T &timestep)
	{
		for (int i = 0; i<uT.size(); i++)
		{
			uT[i] = (image2[i]-image1[i])/timestep;
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

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers, Tdata &image1, Tdata &image2, std::vector<Tdata> gradients, T timeStep)
	{
		T* ptrTau = data->tauElt[dualNumbers[0]].data();
		computeUT(image1, image2, uT, timestep);
		
		Tdata c1(image1.size(), T(0));
		Tdata c2(image1.size(), T(0));
		Tdata c3(image1.size(), T(0));
		Tdata b1(image1.size(), T(0));
		Tdata b2(image1.size(), T(0));

		for(int i = 0; i<image1.size(); i++)
		{
			c1[i]=1+ptrTau[i]*gradients[0][i]*gradients[0][i];
		}

		for(int i = 0; i<image1.size(); i++)
		{
			c2[i]=ptrTau[i]*gradients[0][i]*gradients[1][i];
		}

		for(int i = 0; i<image1.size(); i++)
		{
			c3[i]=1+ptrTau[i]*gradients[1][i]*gradients[1][i];
		}

		T* ptrYtilde1 = data->yTilde[dualNumbers[0]].data();
		T* ptrSigma1 = data->sigmaElt[dualNumbers[0]].data();
		T* ptrTau1 = data->tauElt[dualNumbers[0]].data();

		for(int i = 0; i<image1.size(); i++)
		{
			b1[i]=ptrYtilde1[i]-ptrTau1[i]*gradients[0][i]*uT[i];
		}

		T* ptrYtilde2 = data->yTilde[dualNumbers[1]].data();
		T* ptrSigma2 = data->sigmaElt[dualNumbers[1]].data();
		T* ptrTau2 = data->tauElt[dualNumbers[1]].data();

		for(int i = 0; i<image1.size(); i++)
		{
			b2[i]=ptrYtilde2[i]-ptrTau2[i]*gradients[1][i]*uT[i];
		}
		// used different Tau. TODO: check if it causes problems
		

		//TODO correct this. 
		for (int i = 0; i < dualNumbers.size(); i++)
			{
				T* ptrY = data->y[dualNumbers[i]].data();
				T* ptrYtilde = data->yTilde[dualNumbers[i]].data();
				T* ptrSigma = data->sigmaElt[dualNumbers[i]].data();

				int numElements = (int)data->yTilde[dualNumbers[i]].size();

				#pragma omp parallel for
				for (int j = 0; j < numElements; j++)
				{
					ptrY1[j] = (b1[j]*c3[j]-c2[j]*b2[j])/(c1[j]*c3[j]-c2[j]*c2[j]);
				}

				#pragma omp parallel for
				for (int j = 0; j < numElements; j++)
				{
					ptrY2[j] =(b2[j]*c1[j]-c2[j]*b1[j])/(c1[j]*c3[j]-c2[j]*c2[j]);;
				}
			}
	
		/*
		#ifdef __CUDACC__
            for (int i = 0; i < dualNumbers.size(); i++)
			{
                auto startIterator = thrust::make_zip_iterator(thrust::make_tuple(data->y[dualNumbers[i]].begin(), data->yTilde[dualNumbers[i]].begin(), data->sigmaElt[dualNumbers[i]].begin(),  fList[i].begin()));
                auto endIterator = thrust::make_zip_iterator(  thrust::make_tuple(data->y[dualNumbers[i]].end(),   data->yTilde[dualNumbers[i]].end(),   data->sigmaElt[dualNumbers[i]].end(),    fList[i].end()));

                thrust::for_each(startIterator,endIterator,flexProxDualDataL2Functor(alpha));
            }
		#else
			for (int i = 0; i < dualNumbers.size(); i++)
			{
				T* ptrY = data->y[dualNumbers[i]].data();
				T* ptrYtilde = data->yTilde[dualNumbers[i]].data();
				T* ptrSigma = data->sigmaElt[dualNumbers[i]].data();

				T* ptrF = fList[i].data();

				int numElements = (int)data->yTilde[dualNumbers[i]].size();

				#pragma omp parallel for
				for (int j = 0; j < numElements; j++)
				{
					ptrY[j] = alpha / (ptrSigma[j] + alpha) * (ptrYtilde[j] - ptrSigma[j] * ptrF[j]);
				}
			}
		#endif
		*//*
	}
};

#endif
*/