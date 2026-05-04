#ifndef L1TVOpticalFlow_H
#define L1TVOpticalFlow_H

#include<vector>

/*Method to create a term for the optical flow v · ∇u + u_t = 0
input are an image sequence, the tolerance tol, the gradients ux, uy, ut,....*/
template<T>
void L2L2OpticalFlow(
    const std::vector<std::vector<T>> &imageSequence,
    const std::vector<T>& ux,
    const std::vector<T>& uy,
    const std::vector<T>& ut,

    double lambda,
    const double tol=1e-5,
    const int maxIterations=10000,

    std::vector<double>& v1,
    std::vector<double>& v2)
{
    if (imageSequence.empty()) {
        throw std::invalid_argument("Image sequence vector is empty.");
    }

    const size_t sizeImage = imageSequence[0].size();


}
