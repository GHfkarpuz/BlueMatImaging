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
#include "operator/flexSuperpixelOperator.h"
#include "operator/flexConcatOperator.h"

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

void copyNumpyToFlexMatrix(py::array_t<double> mat, flexMatrix<double>* A)
{
    auto buf = mat.request();  // buffer info holen
    floatingType* ptr = static_cast<floatingType*>(buf.ptr);
    size_t size = 1;
    for (auto r : buf.shape) {
        size *= r;
    }

    vectorData mat_cpp(ptr, ptr + size);
    A->setValueList(mat_cpp);
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
        int gradDirection = operatorDict["gradDirection"].cast<int>() - 1;

        gradientType gradT = gradientType::forward;
        if (gradTypeStr == "backward") gradT = backward;
        else if (gradTypeStr == "central") gradT = central;

        std::vector<int> inputDimension = operatorDict["inputDimension"].cast<std::vector<int>>();

        A = new flexGradientOperator<double>(inputDimension, gradDirection, gradT, isMinus);
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

    // ======================
    // Parameter
    // ======================
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

    // ======================
    // Primal Variablen
    // ======================
    py::list x = problem["x"];
    int numPrimalVars = x.size();

    for (size_t i = 0; i < x.size(); ++i) {
        py::array_t<double> arr = x[i].cast<py::array_t<double>>();

        std::vector<int> dims(arr.shape(), arr.shape() + arr.ndim());
        mainObject->addPrimalVar(dims);

        std::vector<double> vec(arr.data(), arr.data() + arr.size());
        mainObject->setPrimal(i, vec);
    }

    // ======================
    // Dual Terms (nur beim ersten Run)
    // ======================
    if (firstRun) {
        py::list duals = problem["duals"];

        for (size_t i = 0; i < duals.size(); ++i) {
            py::dict dualTerm = duals[i].cast<py::dict>();

            double alpha = dualTerm["factor"].cast<double>();

            // ---------- Prox ----------
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
            else {
                throw std::runtime_error("Unknown prox type: " + proxType);
            }

            // ---------- fList ----------
            py::list fListPy = dualTerm["f"];
            std::vector<std::vector<double>> fList(fListPy.size());

            for (size_t k = 0; k < fListPy.size(); ++k) {
                py::array_t<double> arr = fListPy[k].cast<py::array_t<double>>();
                fList[k] = std::vector<double>(arr.data(), arr.data() + arr.size());
            }

            // ---------- corresponding primals ----------
            py::list corrPrimals = dualTerm["correspondingPrimals"];
            std::vector<int> correspondingPrimals;

            for (auto item : corrPrimals) {
                correspondingPrimals.push_back(item.cast<int>());
            }

            // ---------- Operatoren ----------
            std::vector<flexLinearOperator<double>*> operatorList;

            if (dualTerm.contains("operators")) {
                py::list opListPy = dualTerm["operators"];

                for (size_t j = 0; j < opListPy.size(); ++j) {
                    py::dict opDict = opListPy[j].cast<py::dict>();

                    auto op = transformPythonToFlexOperator(opDict, verbose, j);
                    operatorList.push_back(op);
                }
            }

            // ---------- Term hinzufügen ----------
            mainObject->addTerm(
                new flexTerm<double>(
                    myProx,
                    alpha,
                    (int)correspondingPrimals.size(),
                    operatorList,
                    fList
                ),
                correspondingPrimals
            );
        }
    }

    // ======================
    // Algorithmus starten
    // ======================
    mainObject->runAlgorithm();

    // ======================
    // Ergebnisse zurückgeben
    // ======================
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

