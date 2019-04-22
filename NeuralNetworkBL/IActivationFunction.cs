using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkBL
{
    public interface IActivationFunction
    {
        double f(double x);
        double df(double x);
    }
}
