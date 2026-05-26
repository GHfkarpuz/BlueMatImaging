#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>

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
#include "operator/flexMatrixLogical.h"
#include "operator/flexFullMatrix.h"
#include "operator/flexGradientOperator.h"
#include "operator/flexWeightedGradientOperator.h"
#include "operator/flexWeightedDivergenceOperator.h"
#include "operator/flexSuperpixelOperator.h"
#include "operator/flexConcatOperator.h"
#include "operator/flexOpticalFlowOperator.h"
#include "operator/flexMassPreservationOperator.h"

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
	using namespace thrust;
	#include "operator/flexMatrixGPU.h"

	typedef thrust::device_vector<floatingType> vectorData;
#else
	using namespace std;
	typedef std::vector<floatingType> vectorData;
#endif

namespace py = pybind11;


std::unique_ptr<flexBox<double>> mainObject = nullptr;

template <typename T>
void copyNumpyToFlexMatrix(const py::array_t<T> &mat, flexMatrix<T>* A)
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

template <typename T>
void copyNumpyToFlexFullMatrix(const py::array_t<T> &mat, flexFullMatrix<T>* A)
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
    else if (type == "weightedGradientOperator") {
        if (verbose > 1) printf("Operator %d is type <weightedGradientOperator>\n", operatorNumber);
        std::string gradTypeStr = operatorDict["gradType"].cast<std::string>();
        int gradDirection = operatorDict["gradDirection"].cast<int>();
        std::vector<double> weights = operatorDict["weights"].cast<std::vector<double>>();

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
        std::vector<double> weights = operatorDict["weights"].cast<std::vector<double>>();

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
        std::vector<double> diag = operatorDict["diagonalElements"].cast<std::vector<double>>();
        A = new flexDiagonalOperator<double>(diag, isMinus);
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

        py::array_t<double> mat = operatorDict["matrix"].cast<py::array_t<double>>();

        auto Atmp = new flexMatrix<double>(mat.shape(0), mat.shape(1), isMinus);
        copyNumpyToFlexMatrix(mat, Atmp);
        A = Atmp;
    }
    else if (type == "fullMatrix") {
        if (verbose > 1) printf("Operator %d is type <matrix>\n", operatorNumber);

        py::array_t<double> mat = operatorDict["matrix"].cast<py::array_t<double>>();

        auto Atmp = new flexFullMatrix<double>(mat.shape(0), mat.shape(1), isMinus);
        copyNumpyToFlexFullMatrix(mat, Atmp);
        A = Atmp;
    }
    else if(type == "opticalFlowOp"){
        //TODO Fehler fixen
        if (verbose > 1) printf("Operator %d is type <opticalFlowOp>\n", operatorNumber);

        py::array_t<double> img = operatorDict["image"].cast<py::array_t<double>>();
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
        //TODO Fehler fixen
        if (verbose > 1) printf("Operator %d is type <massPreservationOp>\n", operatorNumber);

        py::array_t<double> img = operatorDict["image"].cast<py::array_t<double>>();
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
    else {
        throw std::runtime_error("Operator type not supported!");
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
        py::array_t<double> arr = x[i].cast<py::array_t<double>>();

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

                std::vector<double> negUT(N);

                for (size_t i = 0; i < N; ++i)
                {
                    negUT[i] = -(ptr2[i] - ptr1[i]) / timeStep;
                }

                // set fList here as -u_t
                std::vector<std::vector<double>> fList;
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
                py::array_t<double> arr = fListPy[k].cast<py::array_t<double>>();
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

    int numDualVars = mainObject->getNumDualVars();

    for (int i = 0; i < numDualVars; ++i) {
        auto vec = mainObject->getDual(i);
        y_out.append(py::array(vec.size(), vec.data()));
    }

    return py::make_tuple(x_out, y_out);
}

PYBIND11_MODULE(flexBoxPy, m) {
    m.def("run", &flexBoxWrapper, "FlexBox solver wrapper");
}