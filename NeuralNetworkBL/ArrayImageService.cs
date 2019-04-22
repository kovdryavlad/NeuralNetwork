
using SimpleMatrix;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace NeuralNetworkBL
{
    public class ArrayImageService
    {
        public static double[,,] getScaledImage(double[,,] imageArray, int scaleSize)
        {
            int height = imageArray.GetLength(1);
            int width = imageArray.GetLength(2);

            double[,,] scaledImage = new double[3, height * scaleSize, width * scaleSize];

            for (int y = 0; y < height; y++)
            {

                int y_bottom = y * scaleSize;
                int y_top = (y + 1) * scaleSize;

                for (int w = 0; w < width; w++)
                {
                    int w_bottom = w * scaleSize;
                    int w_top = (w + 1) * scaleSize;



                    for (int i = y_bottom; i < y_top; i++)
                    {
                        for (int j = w_bottom; j < w_top; j++)
                        {
                            scaledImage[0, i, j] = imageArray[0, y, w];
                            scaledImage[1, i, j] = imageArray[1, y, w];
                            scaledImage[2, i, j] = imageArray[2, y, w];
                        }
                    }
                }
            }

            return scaledImage;
        }

        public static double[,,] GetNoisedImage(double[,,] imageArray, double alpha)
        {
            int height = imageArray.GetLength(1);
            int width = imageArray.GetLength(2);

            double[,,] result = new double[3, height, width];

            Random random = new Random();

            for (int y = 0; y < height; y++)
                for (int w = 0; w < width; w++)
                    if (random.NextDouble() <= alpha){

                        double value = 255 - imageArray[0, y, w];

                        result[0, y, w] = value;
                        result[1, y, w] = value;
                        result[2, y, w] = value;
                    }
                    else {
                        double value = imageArray[0, y, w];

                        result[0, y, w] = value;
                        result[1, y, w] = value;
                        result[2, y, w] = value;
                    }

            return result;
        }
        public static Vector formInputVectorForNeuralNetwork(double[,,] imageArray)
        {
            int imwidth = imageArray.GetLength(2);
            int imheight = imageArray.GetLength(1);

            double[] input = new double[imheight * imwidth];

            for (int y = 0; y < imheight; y++)
                for (int w = 0; w < imwidth; w++)
                    input[y * w + w] = imageArray[0, y, w];

            return new Vector(input);
        }

    }
}