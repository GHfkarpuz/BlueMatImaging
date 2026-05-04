#ifndef BASIC_OPTICAL_FLOW_H
#define BASIC_OPTICAL_FLOW_H

#include <vector>
#include <stdexcept>
#include <cmath>

template<T>
class BasicOpticalFlow
{
public:
    using Image = std::vector<T>;

    enum TermType {
        BrightnessConstancy,
        GradientConstancy
    };

    struct DataTerm {
        std::vector<Image> A;  // diagonal operators (∇u)
        Image f;               // right handside (ut)
    };

    BasicOpticalFlow(
        T alpha,
        const Image& image1,
        const Image& image2,
        int width,
        int height,
        TermType type = BrightnessConstancy,
        int constancyDim = 0
    )
        : alpha(alpha),
          image1(image1),
          image2(image2),
          width(width),
          height(height),
          termType(type),
          constancyDimension(constancyDim)
    {
        if (image1.size() != image2.size())
            throw std::invalid_argument("Images must have same size");
    }

    DataTerm build(const std::vector<Image>& vTilde)
    {
        auto grads = generateGradients(vTilde);

        if (termType == BrightnessConstancy)
            return buildBrightness(grads, vTilde);
        else
            return buildGradient(grads, vTilde);
    }

private:

    struct Gradients {
        Image I2w;
        std::vector<Image> dI2w;
        std::vector<Image> dI1;
    };

    double alpha;
    Image image1, image2;
    int width, height;
    TermType termType;
    int constancyDimension;

    // -----------------------------
    // Gradient computation
    // -----------------------------
    Gradients generateGradients(const std::vector<Image>& vTilde)
    {
        Gradients g;
        int N = width * height;

        g.I2w.resize(N);
        g.dI2w.resize(2);
        g.dI1.resize(2);

        g.dI2w[0].resize(N);
        g.dI2w[1].resize(N);
        g.dI1[0].resize(N);
        g.dI1[1].resize(N);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int idx = y * width + x;

                double vx = vTilde[0][idx];
                double vy = vTilde[1][idx];

                double xw = x + vx;
                double yw = y + vy;

                // --- simple bilinear interpolation ---
                g.I2w[idx] = interpolate(image2, xw, yw);

                // derivatives (central differences)
                g.dI2w[0][idx] =
                    interpolate(image2, xw + 1, yw) -
                    interpolate(image2, xw - 1, yw);

                g.dI2w[1][idx] =
                    interpolate(image2, xw, yw + 1) -
                    interpolate(image2, xw, yw - 1);

                g.dI1[0][idx] =
                    get(image1, x + 1, y) - get(image1, x - 1, y);

                g.dI1[1][idx] =
                    get(image1, x, y + 1) - get(image1, x, y - 1);
            }
        }

        return g;
    }

    // -----------------------------
    // Brightness constancy
    // -----------------------------
    DataTerm buildBrightness(const Gradients& g,
                             const std::vector<Image>& vTilde)
    {
        DataTerm dt;

        int N = width * height;

        dt.A = g.dI2w;
        dt.f.resize(N);

        for (int i = 0; i < N; ++i)
        {
            double val = g.I2w[i] - image1[i];

            for (int d = 0; d < 2; ++d)
            {
                val -= vTilde[d][i] * g.dI2w[d][i];
            }

            dt.f[i] = -val;
        }

        return dt;
    }

    // -----------------------------
    // Gradient constancy
    // -----------------------------
    DataTerm buildGradient(const Gradients& g,
                           const std::vector<Image>& vTilde)
    {
        DataTerm dt;

        int N = width * height;

        dt.A.resize(2);
        dt.A[0].resize(N);
        dt.A[1].resize(N);
        dt.f.resize(N);

        for (int i = 0; i < N; ++i)
        {
            // simple version (1st order)
            dt.A[0][i] = g.dI2w[0][i];
            dt.A[1][i] = g.dI2w[1][i];

            double val = g.dI2w[constancyDimension][i]
                       - g.dI1[constancyDimension][i];

            for (int d = 0; d < 2; ++d)
            {
                val -= vTilde[d][i] * g.dI2w[d][i];
            }

            dt.f[i] = -val;
        }

        return dt;
    }

    // -----------------------------
    // Helpers
    // -----------------------------
    double get(const Image& img, int x, int y)
    {
        x = std::max(0, std::min(width - 1, x));
        y = std::max(0, std::min(height - 1, y));
        return img[y * width + x];
    }

    double interpolate(const Image& img, double x, double y)
    {
        int x0 = floor(x);
        int y0 = floor(y);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        double dx = x - x0;
        double dy = y - y0;

        double v00 = get(img, x0, y0);
        double v10 = get(img, x1, y0);
        double v01 = get(img, x0, y1);
        double v11 = get(img, x1, y1);

        return (1 - dx) * (1 - dy) * v00 +
               dx * (1 - dy) * v10 +
               (1 - dx) * dy * v01 +
               dx * dy * v11;
    }
};

#endif