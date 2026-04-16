#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

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

namespace py = pybind11;

typedef double floatingType;

flexBox<floatingType>* mainObject = nullptr;

//////////////////////////////////////////////////////////////
// Help Funktions
//////////////////////////////////////////////////////////////

std::vector<floatingType> pyListToVector(py::array_t<double> arr)
{
    std::vector<floatingType> v(arr.size());
    std::memcpy(v.data(), arr.data(), arr.size()*sizeof(double));
    return v;
}

//////////////////////////////////////////////////////////////
// Operators
//////////////////////////////////////////////////////////////

flexLinearOperator<floatingType>* createOperator(py::dict op, int verbose)
{
    std::string type = py::str(op["type"]);

    bool isMinus = op.contains("isMinus") ? op["isMinus"].cast<bool>() : false;

    if (type == "gradient")
    {
        auto dims = op["inputDimension"].cast<std::vector<int>>();
        int dir = op["gradDirection"].cast<int>() - 1;
        std::string mode = op["mode"].cast<std::string>();

        gradientType gradT = forward;
        if (mode == "backward") gradT = backward;
        if (mode == "central") gradT = central;

        return new flexGradientOperator<floatingType>(dims, dir, gradT, isMinus);
    }

    else if (type == "identity")
    {
        int n = op["nPx"].cast<int>();
        return new flexIdentityOperator<floatingType>(n, n, isMinus);
    }

    else if (type == "zero")
    {
        int n = op["nPx"].cast<int>();
        return new flexZeroOperator<floatingType>(n, n, isMinus);
    }

    else if (type == "diagonal")
    {
        auto diag = op["diagonalElements"].cast<std::vector<floatingType>>();
        return new flexDiagonalOperator<floatingType>(diag, isMinus);
    }

    else if (type == "matrix")
    {
        auto mat = op["matrix"].cast<std::vector<std::vector<double>>>();

        int rows = mat.size();
        int cols = mat[0].size();

        auto A = new flexFullMatrix<floatingType>(rows, cols, isMinus);

        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                A->insertElement(i*cols + j, mat[i][j]);

        return A;
    }

    throw std::runtime_error("Unknown operator type");
}

//////////////////////////////////////////////////////////////
// Prox-Factory
//////////////////////////////////////////////////////////////

flexProx<floatingType>* createProx(py::dict dual)
{
    std::string type = py::str(dual["type"]);

    if (type == "L1Iso")
        return new flexProxDualL1Iso<floatingType>();

    if (type == "L1Aniso")
        return new flexProxDualL1Aniso<floatingType>();

    if (type == "L2")
        return new flexProxDualL2<floatingType>();

    if (type == "Huber")
    {
        float eps = dual["epsi"].cast<float>();
        return new flexProxDualHuber<floatingType>(eps);
    }

    throw std::runtime_error("Unknown prox");
}

//////////////////////////////////////////////////////////////
// Main Function
//////////////////////////////////////////////////////////////

py::dict run(py::dict input)
{
    bool firstRun = input["firstRun"].cast<bool>();

    if (firstRun)
    {
        if (mainObject) delete mainObject;
        mainObject = new flexBox<floatingType>();
    }
    else
    {
        mainObject->setFirstRun(false);
    }

    //////////////////////////////////////////////////////////
    // Parameter
    //////////////////////////////////////////////////////////

    py::dict params = input["params"];

    if (params.contains("maxIt"))
        mainObject->maxIterations = params["maxIt"].cast<int>();

    if (params.contains("verbose"))
        mainObject->verbose = params["verbose"].cast<int>();

    if (params.contains("tol"))
        mainObject->tol = params["tol"].cast<float>();

    //////////////////////////////////////////////////////////
    // Primal Variables
    //////////////////////////////////////////////////////////

    auto xList = input["x"].cast<py::list>();
    auto dimsList = input["dims"].cast<py::list>();

    int numPrimalVars = xList.size();

    if (firstRun)
    {
        for (int i = 0; i < numPrimalVars; ++i)
        {
            auto dims = dimsList[i].cast<std::vector<int>>();
            mainObject->addPrimalVar(dims);

            auto vec = pyListToVector(xList[i].cast<py::array_t<double>>());
            mainObject->setPrimal(i, vec);
        }
    }

    //////////////////////////////////////////////////////////
    // Dual Terms
    //////////////////////////////////////////////////////////

    if (firstRun)
    {
        auto duals = input["duals"].cast<py::list>();
        auto dcp   = input["DcP"].cast<py::list>();

        for (size_t i = 0; i < duals.size(); ++i)
        {
            py::dict d = duals[i].cast<py::dict>();

            float alpha = d["factor"].cast<float>();

            auto corrPrimals = dcp[i].cast<std::vector<int>>();
            for (auto &v : corrPrimals) v -= 1;

            ////////////////////////////////////////////////////
            // Operators
            ////////////////////////////////////////////////////

            std::vector<flexLinearOperator<floatingType>*> ops;

            for (auto op : d["operator"].cast<py::list>())
            {
                ops.push_back(createOperator(op.cast<py::dict>(), mainObject->verbose));
            }

            ////////////////////////////////////////////////////
            // Prox
            ////////////////////////////////////////////////////

            auto prox = createProx(d);

            ////////////////////////////////////////////////////
            // fList
            ////////////////////////////////////////////////////

            std::vector<std::vector<floatingType>> fList;

            for (auto f : d["f"].cast<py::list>())
            {
                fList.push_back(pyListToVector(f.cast<py::array_t<double>>()));
            }

            mainObject->addTerm(
                new flexTerm<floatingType>(prox, alpha, corrPrimals.size(), ops, fList),
                corrPrimals
            );
        }
    }

    //////////////////////////////////////////////////////////
    // Starting the Algoritm
    //////////////////////////////////////////////////////////

    mainObject->runAlgorithm();

    //////////////////////////////////////////////////////////
    // Output
    //////////////////////////////////////////////////////////

    py::dict result;

    py::list primals;
    for (int i = 0; i < numPrimalVars; ++i)
        primals.append(mainObject->getPrimal(i));

    py::list duals;
    int numDualVars = mainObject->getNumDualVars();
    for (int i = 0; i < numDualVars; ++i)
        duals.append(mainObject->getDual(i));

    result["primal"] = primals;
    result["dual"]   = duals;

    return result;
}

//////////////////////////////////////////////////////////////
// Moduls
//////////////////////////////////////////////////////////////

PYBIND11_MODULE(flexBoxPy, m)
{
    m.def("run", &run);
}