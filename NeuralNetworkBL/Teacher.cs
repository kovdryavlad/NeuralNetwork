using SimpleMatrix;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ImageProcessor;
using System.Drawing;

namespace NeuralNetworkBL
{
    public class Teacher
    {
        string m_folder;
        NeuralNetwork m_network;

        public Teacher(NeuralNetwork network, string folder)
        {
            m_network = network;
            m_folder = folder;            
        }

        public int EPOCH_NUMBER = 1000;
        public List<double> Errors = new List<double>();

        public void Teach()
        {
            Errors.Clear();

            DateTime t1 = DateTime.Now;
            for (int epoсh = 0; epoсh < EPOCH_NUMBER; epoсh++)
            {
                double epohError = 0;
                //System.Diagnostics.Debug.WriteLine("----" + "epoсh" + epoсh+"----");

                string[] folders = Directory.GetDirectories(m_folder);
                for (int i = 0; i < folders.Length; i++)
                {
                    string[] files = Directory.GetFiles(folders[i]);
                    for (int j = 0; j < files.Length; j++)
                    {
                        char ch = Path.GetFileName(folders[i])[0];

                        double[,,] arrImage = BitmapConverter.BitmapToDoubleRgb(new Bitmap(files[j]));
                        Vector input = ArrayImageService.formInputVectorForNeuralNetwork(arrImage);

                        double err = m_network.BackPropagation(input, LettersInformator.GetVectorForLetter(ch), 0.5);
                        //System.Diagnostics.Debug.WriteLine(String.Format("letter: '{0}'  error: {1:0.0000}", ch, err));

                        epohError += err;
                    }
                    
                }

                Errors.Add(epohError);
                //System.Diagnostics.Debug.WriteLine(String.Format("EPOCH: '{0}'  ERROR: {1:0.0000}", epoсh, epohError));
                //System.Diagnostics.Debug.WriteLine("_________________________________");
            }

            DateTime t2 = DateTime.Now;
            System.Diagnostics.Debug.WriteLine("Время обучения: "+ (t2-t1));

        }
    }
}
