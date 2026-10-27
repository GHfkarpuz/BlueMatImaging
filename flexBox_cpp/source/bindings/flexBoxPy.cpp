#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include <cstdlib>
#include <algorithm>

#include <stdio.h>
#include <sys/types.h>
#include <string.h>
#include <cstddef>
#include <ctime>

#include "tools.h"

#include "operator/flexLinearOperator.h"
#include "operator/flexIdentityOperator.h"
#include "operator/flexZeroOperator.h"
#include "operator/flexDiagonalOperator.h"
#include "operator/flexMatrix.h"
#include "operator/flexSparseMatrixOperator.h"
#include "operator/flexMatrixLogical.h"
#include "operator/flexFullMatrix.h"
#include "operator/flexGradientOperator.h"
#include "operator/flexGradientDiagonalOperator.h"
#include "operator/flexUpwindGradientOperator.h"
#include "operator/flexUpwindDivergenceOperator.h"
#include "operator/flexWeightedGradientOperator.h"
#include "operator/flexWeightedDivergenceOperator.h"
#include "operator/flexSuperpixelOperator.h"
#include "operator/flexConcatOperator.h"
#include "operator/flexOpticalFlowOperator.h"
#include "operator/flexOpticalFlowPhaseFieldOperator.h"
#include "operator/flexMassPreservationOperator.h"
#include "operator/flexMassPreservationPhaseFieldOperator.h"
#ifndef __CUDACC__
    // CPU only operators (no GPU implementation)
    #include "operator/flexProjectionOperator.h"
    #include "operator/flexProjectionOutputOperator.h"
    #include "operator/flexConvolutionSquareMinusCircleOperator.h"
#endif

#include "flexBox.h"

#include "term/flexTerm.h"

//prox
#include "prox/flexProxDualDataL1.h"
#include "prox/flexProxDualDataL2.h"
#include "prox/flexProxDualDataKL.h"
#include "prox/flexProxDualDataHuber.h"
#include "prox/flexProxDualL1Aniso.h"
#include "prox/flexProxDualL1Iso.h"
#include "prox/flexProxDualL2.h"
#include "prox/flexProxDualL2Inf.h"
#include "prox/flexProxDualLInf.h"
#include "prox/flexProxDualHuber.h"
#include "prox/flexProxDualFrobenius.h"
#include "prox/flexProxDualBoxConstraint.h"
#include "prox/flexProxDualInnerProduct.h"
#include "prox/flexProxDualLabeling.h"


typedef double floatingType;

#ifdef __CUDACC__
	// (no "using namespace thrust" here: it makes cuda::/cub:: names ambiguous in the CUDA headers)
	#include "operator/flexMatrixGPU.h"

	typedef thrust::device_vector<floatingType> vectorData;
#else
	using namespace std;
	typedef std::vector<floatingType> vectorData;
#endif

namespace py = pybind11;

//! numpy array as contiguous double array (converts e.g. float32 or non contiguous arrays)
typedef py::array_t<double, py::array::c_style | py::array::forcecast> cArray;

//! copies a numpy array (or list) into a std::vector<double> with one memcpy instead of element-wise conversion
std::vector<double> toStdVector(const py::handle& obj)
{
    cArray arr = py::cast<cArray>(obj);
    return std::vector<double>(arr.data(), arr.data() + arr.size());
}


std::unique_ptr<flexBox<double>> mainObject = nullptr;

template <typename T, int Flags>
void copyNumpyToFlexMatrix(const py::array_t<T, Flags> &mat, flexMatrix<T>* A)
{
    py::buffer_info buf = mat.request();

    if (buf.ndim != 2)
        throw std::runtime_error("Matrix must be 2D");

    const int rows = static_cast<int>(buf.shape[0]);
    const int cols = static_cast<int>(buf.shape[1]);

    const T* ptr = static_cast<const T*>(buf.ptr);

    // wichtig: C-order flatten
    std::vector<T> data;
    data.reserve(rows * cols);

    for (int i = 0; i < rows; ++i)
    {
        const T* row_ptr = ptr + i * cols;
        for (int j = 0; j < cols; ++j)
        {
            data.push_back(row_ptr[j]);
        }
    }

    A->setValueList(data);
}

template <typename T, int Flags>
void copyNumpyToFlexFullMatrix(const py::array_t<T, Flags> &mat, flexFullMatrix<T>* A)
{
    py::buffer_info buf = mat.request();

    if (buf.ndim != 2)
        throw std::runtime_error("Matrix must be 2D");

    const int rows = static_cast<int>(buf.shape[0]);
    const int cols = static_cast<int>(buf.shape[1]);

    const T* ptr = static_cast<const T*>(buf.ptr);

    // 
    A->setNumRows(rows);
    A->setNumCols(cols);

    std::vector<T> data;
    data.reserve(rows * cols);

    // 
    for (int i = 0; i < rows; ++i)
    {
        const T* row_ptr = ptr + i * cols;
        for (int j = 0; j < cols; ++j)
        {
            data.push_back(row_ptr[j]);
        }
    }

    A->setValueList(data);
}

flexLinearOperator<double>* transformPythonToFlexOperator(py::dict operatorDict, int verbose, int operatorNumber, bool isGPU = false)
{
    flexLinearOperator<double>* A;
    bool isMinus = false;

    if (operatorDict.contains("isMinus")) {
        isMinus = operatorDict["isMinus"].cast<bool>();
    }

    if (verbose > 1) {
        printf("isMinus is set to %d\n", isMinus);
    }

    std::string type = operatorDict["type"].cast<std::string>();

    if (type == "functionHandleOperator") {
        throw std::runtime_error("Operator type functionHandleOperator not supported!");
    }
    else if (type == "gradientOperator") {
        if (verbose > 1) printf("Operator %d is type <gradientOperator>\n", operatorNumber);

        std::string gradTypeStr = operatorDict["gradType"].cast<std::string>();
        int gradDirection = operatorDict["gradDirection"].cast<int>();

        gradientType gradT = gradientType::forward;
        if (gradTypeStr == "backward") gradT = backward;
        else if (gradTypeStr == "central") gradT = central;

        std::vector<int> inputDimension = operatorDict["inputDimension"].cast<std::vector<int>>();

        A = new flexGradientOperator<double>(inputDimension, gradDirection, gradT, isMinus);
    }
    else if (type == "upwindGradientOperator") {
        if (verbose > 1) printf("Operator %d is type <upwindGradientOperator>\n", operatorNumber);

        int gradDirection = operatorDict["gradDirection"].cast<int>();

        std::vector<double> motionField = toStdVector(operatorDict["motionField"]);

        std::vector<int> inputDimension = operatorDict["inputDimension"].cast<std::vector<int>>();

        A = new flexUpwindGradientOperator<double>(inputDimension, gradDirection, motionField, isMinus);
    }
    else if (type == "upwindDivergenceOperator") {
        if (verbose > 1) printf("Operator %d is type <upwindDivergenceOperator>\n", operatorNumber);
        std::string gradTypeStr = operatorDict["gradType"].cast<std::string>();
        
        int gradDirection = operatorDict["gradDirection"].cast<int>();

        std::vector<double> weights = toStdVector(operatorDict["weights"]);

        gradientType gradT = gradientType::forward;
        if (gradTypeStr == "backward") gradT = backward;
        else if (gradTypeStr == "central") gradT = central;

        std::vector<int> inputDimension = operatorDict["inputDimension"].cast<std::vector<int>>();

        A = new flexUpwindDivergenceOperator<double>(weights, inputDimension, gradDirection, gradT, isMinus);
    }
    else if (type == "weightedGradientOperator") {
        if (verbose > 1) printf("Operator %d is type <weightedGradientOperator>\n", operatorNumber);
        std::string gradTypeStr = operatorDict["gradType"].cast<std::string>();
        int gradDirection = operatorDict["gradDirection"].cast<int>();
        std::vector<double> weights = toStdVector(operatorDict["weights"]);

        gradientType gradT = gradientType::forward;
        if (gradTypeStr == "backward") gradT = backward;
        else if (gradTypeStr == "central") gradT = central;

        std::vector<int> inputDimension = operatorDict["inputDimension"].cast<std::vector<int>>();

        A = new flexWeightedGradientOperator<double>(weights, inputDimension, gradDirection, gradT, isMinus);
    }
    else if (type == "weightedDivergenceOperator") {
        if (verbose > 1) printf("Operator %d is type <weightedDivergenceOperator>\n", operatorNumber);
        std::string gradTypeStr = operatorDict["gradType"].cast<std::string>();
        int gradDirection = operatorDict["gradDirection"].cast<int>();
        std::vector<double> weights = toStdVector(operatorDict["weights"]);

        gradientType gradT = gradientType::forward;
        if (gradTypeStr == "backward") gradT = backward;
        else if (gradTypeStr == "central") gradT = central;

        std::vector<int> inputDimension = operatorDict["inputDimension"].cast<std::vector<int>>();

        A = new flexWeightedDivergenceOperator<double>(weights, inputDimension, gradDirection, gradT, isMinus);
    }
    else if (type == "identityOperator") {
        if (verbose > 1) printf("Operator %d is type <identityOperator>\n", operatorNumber);
        int nPx = operatorDict["nPx"].cast<int>();
        A = new flexIdentityOperator<double>(nPx, nPx, isMinus);
    }
    else if (type == "zeroOperator") {
        if (verbose > 1) printf("Operator %d is type <zeroOperator>\n", operatorNumber);
        int nPx = operatorDict["nPx"].cast<int>();
        A = new flexZeroOperator<double>(nPx, nPx, isMinus);
    }
    else if (type == "diagonalOperator") {
        if (verbose > 1) printf("Operator %d is type <diagonalOperator>\n", operatorNumber);
        std::vector<double> diag = toStdVector(operatorDict["diagonalElements"]);
        A = new flexDiagonalOperator<double>(diag, isMinus);
    }
    else if (type == "gradientDiagonalOperator") {
        std::vector<double> weights = toStdVector(operatorDict["weights"]);
        std::string gradTypeStr = operatorDict["gradType"].cast<std::string>();
        int gradDirection = operatorDict["gradDirection"].cast<int>();

        gradientType gradT = gradientType::forward;
        if (gradTypeStr == "backward") gradT = backward;
        else if (gradTypeStr == "central") gradT = central;

        std::vector<int> inputDimension = operatorDict["inputDimension"].cast<std::vector<int>>();

        A = new flexGradientDiagonalOperator<double>(weights, inputDimension, gradDirection, gradT, isMinus);
    }
    else if (type == "concatOperator") {
        if (verbose > 1) printf("Operator %d is type <concatOperator>\n", operatorNumber);

        std::string operationStr = operatorDict["operation"].cast<std::string>();
        mySign operation;
        if (operationStr == "composition") operation = COMPOSE;
        else if (operationStr == "addition") operation = PLUS;
        else if (operationStr == "difference") operation = MINUS;
        else throw std::runtime_error("Unknown operation for concatOperator");

        py::dict opA = operatorDict["A"].cast<py::dict>();
        py::dict opB = operatorDict["B"].cast<py::dict>();

        auto operator1 = transformPythonToFlexOperator(opA, verbose, operatorNumber, isGPU);
        auto operator2 = transformPythonToFlexOperator(opB, verbose, operatorNumber, isGPU);

        A = new flexConcatOperator<double>(operator1, operator2, operation, isMinus);
    }
    else if (type == "superpixelOperator" && !isGPU) {
        if (verbose > 1) printf("Operator %d is type <superpixelOperator>\n", operatorNumber);

        float factor = operatorDict["factor"].cast<float>();
        std::vector<int> targetDim = operatorDict["targetDimension"].cast<std::vector<int>>();

        A = new flexSuperpixelOperator<double>(targetDim, factor, isMinus);
    }
    else if (type == "matrix") {
        if (verbose > 1) printf("Operator %d is type <matrix>\n", operatorNumber);

        cArray mat = operatorDict["matrix"].cast<cArray>();

        auto Atmp = new flexMatrix<double>(mat.shape(0), mat.shape(1), isMinus);
        copyNumpyToFlexMatrix(mat, Atmp);
        A = Atmp;
    }
    else if (type == "sparseMatrix") {
        // general sparse matrix in CSR format (e.g. scipy.sparse.csr_matrix: indptr, indices, data), CPU and GPU
        if (verbose > 1) printf("Operator %d is type <sparseMatrix>\n", operatorNumber);
        std::vector<int> shape = operatorDict["shape"].cast<std::vector<int>>();
        py::array_t<int, py::array::c_style | py::array::forcecast> indptr = operatorDict["indptr"].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
        py::array_t<int, py::array::c_style | py::array::forcecast> indices = operatorDict["indices"].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
        if ((int)indptr.size() != shape[0] + 1)
            throw std::runtime_error("sparseMatrix: indptr needs shape[0] + 1 entries");
        std::vector<int> rp(indptr.data(), indptr.data() + indptr.size());
        std::vector<int> ci(indices.data(), indices.data() + indices.size());
        std::vector<double> val = toStdVector(operatorDict["data"]);
        if ((int)ci.size() != rp[shape[0]] || (int)val.size() != rp[shape[0]])
            throw std::runtime_error("sparseMatrix: indices / data need indptr[-1] entries");
        A = new flexSparseMatrixOperator<double>(shape[0], shape[1], rp, ci, val, isMinus);
    }
    else if (type == "fullMatrix") {
        if (verbose > 1) printf("Operator %d is type <matrix>\n", operatorNumber);

        cArray mat = operatorDict["matrix"].cast<cArray>();

        auto Atmp = new flexFullMatrix<double>(mat.shape(0), mat.shape(1), isMinus);
        copyNumpyToFlexFullMatrix(mat, Atmp);
        A = Atmp;
    }
    else if(type == "opticalFlowOp"){
        if (verbose > 1) printf("Operator %d is type <opticalFlowOp>\n", operatorNumber);

        cArray img = operatorDict["image"].cast<cArray>();
        //py::array_t<double, py::array::f_style> img = operatorDict["image"].cast<py::array_t<double, py::array::f_style>>();
        int direction = operatorDict["direction"].cast<int>();

        //take the image
        auto buf = img.request();
        double* ptr = static_cast<double*>(buf.ptr);
        
        size_t size = 1;
        for (auto r : buf.shape) size *= r;

        std::vector<double> imageVec(ptr, ptr + size);

        //dimensions
        std::vector<int> inputDimension = operatorDict["inputDimension"].cast<std::vector<int>>();

        //create operator
        auto AOpticalFlow = new flexOpticalFlowOperator<double>(imageVec, inputDimension, direction, isMinus);
        A = AOpticalFlow;
    }
    else if(type == "massPreservationOp"){
        if (verbose > 1) printf("Operator %d is type <massPreservationOp>\n", operatorNumber);

        cArray img = operatorDict["image"].cast<cArray>();
        int direction = operatorDict["direction"].cast<int>();

        //take the image
        auto buf = img.request();

        double* ptr = static_cast<double*>(buf.ptr);
        size_t size = 1;
        for (auto r : buf.shape) size *= r;

        std::vector<double> imageVec(ptr, ptr + size);

        //dimensions
        std::vector<int> inputDimension = operatorDict["inputDimension"].cast<std::vector<int>>();

        //create operator
        auto AMassPreservation = new flexMassPreservationOperator<double>(imageVec, inputDimension, direction, isMinus);
        A = AMassPreservation;
    }
    else if(type == "opticalFlowPhaseFieldOp"){
        if (verbose > 1) printf("Operator %d is type <opticalFlowPhaseFieldOp>\n", operatorNumber);

        cArray img = operatorDict["image"].cast<cArray>();
        cArray mf0 = operatorDict["motionField_0"].cast<cArray>();
        cArray mf1 = operatorDict["motionField_1"].cast<cArray>();

        //take the image
        auto buf = img.request();
        double* ptr = static_cast<double*>(buf.ptr);
        
        size_t size = 1;
        for (auto r : buf.shape) size *= r;

        std::vector<double> imageVec(ptr, ptr + size);

        // take motionField_0
        auto mf0Buf = mf0.request();
        double* mf0Ptr = static_cast<double*>(mf0Buf.ptr);

        size_t mf0Size = 1;
        for (auto r : mf0Buf.shape)
            mf0Size *= r;

        std::vector<double> mf0Vec(mf0Ptr, mf0Ptr + mf0Size);


        // take motionField_1
        auto mf1Buf = mf1.request();
        double* mf1Ptr = static_cast<double*>(mf1Buf.ptr);

        size_t mf1Size = 1;
        for (auto r : mf1Buf.shape)
            mf1Size *= r;

        std::vector<double> mf1Vec(mf1Ptr, mf1Ptr + mf1Size);

        //dimensions
        std::vector<int> inputDimension = operatorDict["inputDimension"].cast<std::vector<int>>();

        //create operator
        auto AOpticalFlowPhaseField = new flexOpticalFlowPhaseFieldOperator<double>(imageVec, mf0Vec, mf1Vec, inputDimension, isMinus);
        A = AOpticalFlowPhaseField;
    }
    else if(type == "massPreservationPhaseFieldOp"){
        if (verbose > 1) printf("Operator %d is type <massPreservationPhaseFieldOp>\n", operatorNumber);

        cArray img = operatorDict["image"].cast<cArray>();
        cArray mf0 = operatorDict["motionField_0"].cast<cArray>();
        cArray mf1 = operatorDict["motionField_1"].cast<cArray>();

        //take the image
        auto buf = img.request();
        double* ptr = static_cast<double*>(buf.ptr);
        
        size_t size = 1;
        for (auto r : buf.shape) size *= r;

        std::vector<double> imageVec(ptr, ptr + size);

        // take motionField_0
        auto mf0Buf = mf0.request();
        double* mf0Ptr = static_cast<double*>(mf0Buf.ptr);

        size_t mf0Size = 1;
        for (auto r : mf0Buf.shape)
            mf0Size *= r;

        std::vector<double> mf0Vec(mf0Ptr, mf0Ptr + mf0Size);


        // take motionField_1
        auto mf1Buf = mf1.request();
        double* mf1Ptr = static_cast<double*>(mf1Buf.ptr);

        size_t mf1Size = 1;
        for (auto r : mf1Buf.shape)
            mf1Size *= r;

        std::vector<double> mf1Vec(mf1Ptr, mf1Ptr + mf1Size);

        //dimensions
        std::vector<int> inputDimension = operatorDict["inputDimension"].cast<std::vector<int>>();

        //create operator
        auto AMassPreservationPhaseField = new flexMassPreservationPhaseFieldOperator<double>(imageVec, mf0Vec, mf1Vec, inputDimension, isMinus);
        A = AMassPreservationPhaseField;
    }
#ifndef __CUDACC__
    else if(type == "projectionOp"){
        if (verbose > 1) printf("Operator %d is type <projectionOp>\n", operatorNumber);

        int index = operatorDict["index"].cast<int>();

        py::dict op = operatorDict["operator"].cast<py::dict>();

        auto ProjOp = transformPythonToFlexOperator(op, verbose, operatorNumber, isGPU);

        //dimensions
        std::vector<int> inputDimension = operatorDict["inputDimension"].cast<std::vector<int>>();

        //create operator
        A = new flexProjectionOperator<double>(ProjOp, index, inputDimension, isMinus);
    }
    else if(type == "projectionOutputOp"){
        if (verbose > 1) printf("Operator %d is type <projectionOutputOp>\n", operatorNumber);

        int index = operatorDict["index"].cast<int>();

        //dimensions
        std::vector<int> inputDimension = operatorDict["inputDimension"].cast<std::vector<int>>();

        //create operator
        A = new flexProjectionOutputOperator<double>(index, inputDimension, isMinus);
    }
    else if(type == "convolutionSquareMinusCircleOp"){
        if (verbose > 1) printf("Operator %d is type <convolutionSquareMinusCircleOp>\n", operatorNumber);

        int radius = operatorDict["radius"].cast<int>();
        bool withTimeDimension = operatorDict["withTimeDimension"].cast<bool>();
        //dimensions
        std::vector<int> inputDimension = operatorDict["inputDimension"].cast<std::vector<int>>();

        //create operator
        A = new flexConvolutionSquareMinusCircleOperator<double>(radius, withTimeDimension, inputDimension, isMinus);
    }
#else
    else if(type == "projectionOp" || type == "projectionOutputOp" || type == "convolutionSquareMinusCircleOp"){
        throw std::runtime_error("Operator type " + type + " is only available in the CPU version (flexBoxPy)");
    }
#endif
    else {
        throw std::runtime_error("Operator type not supported: " + type);
    }

    return A;
}

py::tuple flexBoxWrapper(py::dict problem) {
    bool firstRun = problem["firstRun"].cast<bool>();
    if (firstRun) {
        mainObject.reset(new flexBox<double>());
        mainObject->isMATLAB = false;
    } else {
        mainObject->setFirstRun(false);
    }


    // parameters
    py::dict params = problem["params"];
    if (params.contains("maxIt")) mainObject->maxIterations = params["maxIt"].cast<int>();
    if (params.contains("verbose")) mainObject->verbose = params["verbose"].cast<int>();
    if (params.contains("tol")) mainObject->tol = params["tol"].cast<double>();
    if (params.contains("checkError")) mainObject->checkError = params["checkError"].cast<int>();

    int verbose = mainObject->verbose;

    if (verbose > 0) {
        printf("Parameters:\n");
        printf("maxIterations: %d\n", mainObject->maxIterations);
        printf("verbose: %d\n", mainObject->verbose);
        printf("tol: %f\n", mainObject->tol);
        printf("checkError: %d\n", mainObject->checkError);
    }

    // primal variables
    py::list x = problem["x"];
    int numPrimalVars = x.size();
    for (size_t i = 0; i < x.size(); ++i) {
        cArray arr = x[i].cast<cArray>();
        //py::array_t<double, py::array::f_style> arr = x[i].cast<py::array_t<double, py::array::f_style>>();

        std::vector<int> dims(arr.shape(), arr.shape() + arr.ndim());
        mainObject->addPrimalVar(dims);

        std::vector<double> vec(arr.data(), arr.data() + arr.size());
        mainObject->setPrimal(i, vec);
    }

    // Dual Terms (only for first run)
    if (firstRun) {
        py::list duals = problem["duals"];

        for (size_t i = 0; i < duals.size(); ++i) {
            py::dict dualTerm = duals[i].cast<py::dict>();
            double alpha = dualTerm["factor"].cast<double>();

            // Prox
            flexProx<double>* myProx = nullptr;
            std::string proxType = dualTerm["type"].cast<std::string>();

            if (proxType == "L1IsoProxDual") {
                myProx = new flexProxDualL1Iso<double>();
            }
            else if (proxType == "L1AnisoProxDual") {
                myProx = new flexProxDualL1Aniso<double>();
            }
            else if (proxType == "L2proxDual") {
                myProx = new flexProxDualL2<double>();
            }
            else if (proxType == "L1proxDualData") {
                myProx = new flexProxDualDataL1<double>();
            }
            else if (proxType == "L2proxDualData") {
                myProx = new flexProxDualDataL2<double>();
            }
            else if (proxType == "BoxConstraintProxDual") {
                // indicator function of [minVal, maxVal], alpha is ignored
                double minVal = dualTerm["minVal"].cast<double>();
                double maxVal = dualTerm["maxVal"].cast<double>();
                myProx = new flexProxDualBoxConstraint<double>(minVal, maxVal);
            }
            else if(proxType == "dualOpticalFlowL2DataProx")
            {
                myProx = new flexProxDualDataL2<double>();
                /*
                // images is a list of two images
                py::list<py::array_t<double>> images = dualTerm["f"].cast<py::list<py::array_t<double>>>();

                double timeStep = dualTerm["timeStep"].cast<double>();

                auto buf1 = images[0].request();
                auto buf2 = images[1].request();

                double* ptr1 = static_cast<double*>(buf1.ptr);
                double* ptr2 = static_cast<double*>(buf2.ptr);

                size_t N = 1;
                for (auto r : buf1.shape) N *= r;

                vectorData negUT(N);

                for (size_t i = 0; i < N; ++i)
                {
                    negUT[i] = -(ptr2[i] - ptr1[i]) / timeStep;
                }

                // set fList here as -u_t
                std::vector<vectorData> fList;
                fList.push_back(negUT);

                //for this case the fList will not be read from the dual list
                */
            }
            else {
                throw std::runtime_error("Unknown prox type: " + proxType);
            }

            
            // fList
            py::list fListPy = dualTerm["f"];
            std::vector<std::vector<double>> fList(fListPy.size());

            for (size_t k = 0; k < fListPy.size(); ++k) {
                cArray arr = fListPy[k].cast<cArray>();
                fList[k] = std::vector<double>(arr.data(), arr.data() + arr.size());
            }
            

            // corresponding primals
            py::list corrPrimals = dualTerm["correspondingPrimals"];
            std::vector<int> correspondingPrimals;

            for (auto item : corrPrimals) {
                correspondingPrimals.push_back(item.cast<int>());
            }

            // Operators
            std::vector<flexLinearOperator<double>*> operatorList;

            if (dualTerm.contains("operators")) {
                py::list opListPy = dualTerm["operators"];

                for (size_t j = 0; j < opListPy.size(); ++j) {
                    py::dict opDict = opListPy[j].cast<py::dict>();

                    auto op = transformPythonToFlexOperator(opDict, verbose, j);
                    operatorList.push_back(op);
                }
            }

            // add a term 
            mainObject->addTerm(
                new flexTerm<double>(myProx, alpha, (int)correspondingPrimals.size(), operatorList, fList),
                correspondingPrimals
            );
        }
    }

    // start the algorithm
    mainObject->runAlgorithm();


    // return results
    py::list x_out, y_out;

    
    for (int i = 0; i < numPrimalVars; ++i) {
        auto vec = mainObject->getPrimal(i);
        x_out.append(py::array(vec.size(), vec.data()));
    }

    /*
    for (int i = 0; i < numPrimalVars; ++i) {
        auto vec = mainObject->getPrimal(i);

        py::array arr_in = x[i].cast<py::array>();
        std::vector<size_t> shape(arr_in.shape(), arr_in.shape() + arr_in.ndim());

        std::vector<size_t> strides;
        
        // Strides für Fortran-Layout berechnen (Spalten wandern zuerst)
        size_t acc = sizeof(double);
        for (size_t d : shape) {
            strides.push_back(acc);
            acc *= d;
        }

        x_out.append(py::array(py::buffer_info(
            vec.data(),
            sizeof(double),
            py::format_descriptor<double>::format(),
            shape.size(),
            shape,
            strides
        )));
    }
    */

    int numDualVars = mainObject->getNumDualVars();

    for (int i = 0; i < numDualVars; ++i) {
        auto vec = mainObject->getDual(i);
        y_out.append(py::array(vec.size(), vec.data()));
    }

    return py::make_tuple(x_out, y_out);
}

// applies a single operator: mode 0 = times, 1 = timesPlus, 2 = timesMinus (output starts with outputInit)
// used to test operators (CPU vs. GPU, adjointness)
py::array_t<double> applyOperatorWrapper(py::dict operatorDict, py::array_t<double> input, bool transposed, int mode, py::array_t<double> outputInit)
{
    std::unique_ptr<flexLinearOperator<double>> A(transformPythonToFlexOperator(operatorDict, 0, 0));

    std::vector<double> inHost(input.data(), input.data() + input.size());
    std::vector<double> outHost(outputInit.data(), outputInit.data() + outputInit.size());

    vectorData in(inHost.begin(), inHost.end());
    vectorData out(outHost.begin(), outHost.end());

    if (mode == 0) A->times(transposed, in, out);
    else if (mode == 1) A->timesPlus(transposed, in, out);
    else A->timesMinus(transposed, in, out);

    std::vector<double> result(out.size());
    #ifdef __CUDACC__
        thrust::copy(out.begin(), out.end(), result.begin());
    #else
        result = out;
    #endif
    return py::array_t<double>(result.size(), result.data());
}

// returns the absolute row sums of a single operator (used for the step sizes tau and sigma)
py::array_t<double> absRowSumWrapper(py::dict operatorDict, bool transposed)
{
    std::unique_ptr<flexLinearOperator<double>> A(transformPythonToFlexOperator(operatorDict, 0, 0));

    std::vector<double> result;
    #ifdef __CUDACC__
        thrust::device_vector<double> tmp = A->getAbsRowSumCUDA(transposed);
        result.resize(tmp.size());
        thrust::copy(tmp.begin(), tmp.end(), result.begin());
    #else
        result = A->getAbsRowSum(transposed);
    #endif
    return py::array_t<double>(result.size(), result.data());
}

#ifndef FLEXBOX_MODULE_NAME
    #define FLEXBOX_MODULE_NAME flexBoxPy
#endif

// default number of OpenMP threads if OMP_NUM_THREADS is not set: the operators are memory bound,
// more threads than this rarely help and hurt a lot on shared machines
#ifndef FLEX_DEFAULT_THREADS
    #define FLEX_DEFAULT_THREADS 8
#endif

void setNumThreads(int n)
{
    #ifdef _OPENMP
        omp_set_num_threads(n > 0 ? n : 1);
    #endif
}

int getNumThreads()
{
    #ifdef _OPENMP
        return omp_get_max_threads();
    #else
        return 1;
    #endif
}

PYBIND11_MODULE(FLEXBOX_MODULE_NAME, m) {
    #ifdef _OPENMP
        if (std::getenv("OMP_NUM_THREADS") == nullptr)
            omp_set_num_threads(std::min(FLEX_DEFAULT_THREADS, omp_get_num_procs()));
    #endif

    // free the solver (and its GPU memory) before the CUDA runtime is shut down at interpreter exit
    py::module_::import("atexit").attr("register")(py::cpp_function([]() { mainObject.reset(); }));

    m.def("run", &flexBoxWrapper, "FlexBox solver wrapper");
    m.def("setNumThreads", &setNumThreads, "sets the number of OpenMP threads used on the CPU");
    m.def("getNumThreads", &getNumThreads, "returns the number of OpenMP threads used on the CPU");
    m.def("applyOperator", &applyOperatorWrapper, "applies a single operator (mode 0: times, 1: timesPlus, 2: timesMinus)");
    m.def("absRowSum", &absRowSumWrapper, "absolute row sums of a single operator");
    #ifdef __CUDACC__
        m.attr("isGPU") = true;
    #else
        m.attr("isGPU") = false;
    #endif
}