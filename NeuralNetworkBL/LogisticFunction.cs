using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkBL
{
    public class LogisticFunction : IActivationFunction
    {
        public double alpha = 1;
        
        public double df(double x)
        {
            double fx = f(x);
            return fx * (1 - fx);
        }

        public double f(double x)=> 1d / (1 + Math.Exp(-alpha* x));
    }
}
