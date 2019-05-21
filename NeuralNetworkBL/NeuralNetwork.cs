using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;
using SimpleMatrix;

namespace NeuralNetworkBL
{
    [Serializable]
    public class NeuralNetwork
    {
        int m_inputLayerLength;
        int[] m_layerLengths;

        Matrix[] W;
        Vector[] m_S;
        Vector[] m_Y;

        [NonSerialized]
        public IActivationFunction m_activationFunction = new LogisticFunction();

        public NeuralNetwork(int[] layerLengths)
        {
            m_inputLayerLength = layerLengths[0];
            m_layerLengths = layerLengths;

            W = new Matrix[layerLengths.Length - 1];

            for (int i = 0; i < layerLengths.Length - 1; i++)
                W[i] = new Matrix(layerLengths[i], layerLengths[i + 1]);

            RandomInitialization();
        }

        private void RandomInitialization()
        {
            Random r = new Random();
            for (int i = 0; i < W.Length; i++)
                for (int j = 0; j < W[i].Rows; j++)
                    for (int k = 0; k < W[i].Columns; k++)
                    {
                        double d = r.Next(2) == 1 ? -1 : 1;
                        W[i].data[j][k] = d * (r.Next(31) / 100d);
                    }
        }

        public static void Serialize(string filename, NeuralNetwork network)
        {
            FileStream fs = new FileStream(filename, FileMode.Create);

            BinaryFormatter formatter = new BinaryFormatter();
            formatter.Serialize(fs, network);

            fs.Dispose();
        }

        public static NeuralNetwork Deserialize(string filename)
        {
            FileStream fs = new FileStream(filename, FileMode.Open);
            BinaryFormatter formatter = new BinaryFormatter();

            NeuralNetwork result = formatter.Deserialize(fs) as NeuralNetwork;
            fs.Dispose();

            return result;


        }

        private Vector calcResult(Vector vector, NeuralNetworkCalcResultOptions options)
        {
            //ошибка по несовпадению размера первого слоя
            if (vector.Length != m_inputLayerLength)
                throw new ArgumentException("vector.Length != input layer length");

            //тут можно написать лямбду, которая уберет повторения кода

            if (options == NeuralNetworkCalcResultOptions.WithSavingSum)
            {
                m_S = new Vector[W.Length];
                m_Y = new Vector[W.Length];
            }
            Vector res = vector * W[0];

            //для обучения
            if (options == NeuralNetworkCalcResultOptions.WithSavingSum)
                m_S[0] = new Vector(res.GetCloneOfData());

            //активация
            res = activateVector(res);

            if (options == NeuralNetworkCalcResultOptions.WithSavingSum)
                m_Y[0] = new Vector(res.GetCloneOfData());

            for (int i = 1; i < m_layerLengths.Length - 1; i++)
            {
                res = res * W[i];

                //для обучения
                if (options == NeuralNetworkCalcResultOptions.WithSavingSum)
                    m_S[i] = new Vector(res.GetCloneOfData());     //для обучения

                //активация
                res = activateVector(res);

                if (options == NeuralNetworkCalcResultOptions.WithSavingSum)
                    m_Y[i] = new Vector(res.GetCloneOfData());
            }

            return res;
        }

        public Vector calcResult(Vector vector) =>
            calcResult(vector, NeuralNetworkCalcResultOptions.WithoutSavingSum);

        private Vector activateVector(Vector a)
        {
            double[] arr = new double[a.Length];

            for (int i = 0; i < a.Length; i++)
                arr[i] = m_activationFunction.f(a[i]);

            return new Vector(arr);
        }

        public double BackPropagation(Vector x, Vector a, double n)
        {
            Matrix[] WfixingArray = new Matrix[W.Length];

            Vector y = calcResult(x, NeuralNetworkCalcResultOptions.WithSavingSum);
            int lastLayer = m_S.Length - 1;

            Vector delta_kNext = Vector.Create.New(W[lastLayer].Columns);

            double err = 0;
            #region Last Layer Fixing
            {
                Vector eps = a - y;

                //debug log
                err = eps.GetCloneOfData().Select(el => el * el).Sum();

                Matrix lastWFixMatrix = new Matrix(W[lastLayer].Rows, W[lastLayer].Columns);

                for (int j = 0; j < lastWFixMatrix.Columns; j++)
                {
                    delta_kNext[j] = eps[j] * m_activationFunction.df(m_S[lastLayer][j]);
                    for (int i = 0; i < lastWFixMatrix.Rows; i++)
                        lastWFixMatrix[i, j] = delta_kNext[j] * m_Y[lastLayer - 1][i];

                }

                WfixingArray[lastLayer] = lastWFixMatrix;
            }
            #endregion

            #region another layers fixing
            for (int layer = lastLayer - 1; layer >= 0; layer--)
            {
                double[] eps = new double[W[layer].Columns];

                for (int i = 0; i < eps.Length; i++)
                    for (int m = 0; m < W[layer + 1].Columns; m++)
                        eps[i] += delta_kNext[m] * W[layer + 1][i, m];


                Vector delta_k = Vector.Create.New(W[layer].Columns);

                for (int i = 0; i < delta_k.Length; i++)
                    delta_k[i] = eps[i] * m_activationFunction.df(m_S[layer][i]);


                Matrix fixMatrix = new Matrix(W[layer].Rows, W[layer].Columns);
                for (int i = 0; i < fixMatrix.Rows; i++)
                    for (int j = 0; j < fixMatrix.Columns; j++)
                        if ((layer - 1) >= 0)
                            fixMatrix[i, j] = delta_k[j] * m_Y[layer - 1][i];
                        else
                            fixMatrix[i, j] = delta_k[j] * x[i];


                WfixingArray[layer] = fixMatrix;

                //присвоение рода delta_k+1 = delta_k
                delta_kNext = delta_k;
            }
            #endregion 

            //внесение поправок во все существующие веса
            for (int i = 0; i < W.Length; i++)
                W[i] += 2 * n * WfixingArray[i];


            return err;
        }

        public static void Test()
        {
            NeuralNetwork neuralNetwork = new NeuralNetwork(new[] { 3, 2, 2 });

            neuralNetwork.W[0] = new Matrix(3, 2, new double[] {
                 0.22,   0.10,
                 0.08,  -0.16,
                 -0.07,  0.15
            });

            neuralNetwork.W[1] = new Matrix(2, 2, new double[] {
                 0.10,   0.20,
                -0.05,   0.10
            });

            neuralNetwork.BackPropagation(new Vector(new[] { 5d, 10, 15 }),
                                          new Vector(new[] { 0d, 1 }),
                                          0.5);


        }
    }
}
