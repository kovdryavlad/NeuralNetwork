using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using ImageProcessor;
using NeuralNetworkBL;
using SimpleMatrix;

namespace NeuralNetwork
{
    public partial class Form1 : Form
    {
        double[,,] m_image;

        //файл с весами
        string neuralNetworkFile = "NeuralNetwork.dat";
        NeuralNetworkBL.NeuralNetwork m_neuralNetwork;

        string m_sampleDirectory;

        public Form1()
        {
            InitializeComponent();
            this.FormClosing += Form1_FormClosing;        

            if (File.Exists(neuralNetworkFile))
                m_neuralNetwork = NeuralNetworkBL.NeuralNetwork.Deserialize(neuralNetworkFile);

            else
                m_neuralNetwork = new NeuralNetworkBL.NeuralNetwork(new[] { 16 * 16, 100, 4 });

            m_neuralNetwork.m_activationFunction = new LogisticFunction();

            findSampleDirectory();
        }

        private void findSampleDirectory()
        {
            //need go up 3 times to folder "sample"
            string nededdFolder = Environment.CurrentDirectory;
            for (int i = 0; i < 3; i++)
                nededdFolder = new DirectoryInfo(nededdFolder).Parent.FullName;

            m_sampleDirectory = Path.Combine(nededdFolder, "sample");
        }


        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            NeuralNetworkBL.NeuralNetwork.Serialize(neuralNetworkFile, m_neuralNetwork);
        }

        private void вілкритиToolStripMenuItem_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.InitialDirectory = m_sampleDirectory;

            if (ofd.ShowDialog() != DialogResult.OK)
                return;

            m_image = BitmapConverter.BitmapToDoubleRgb(new Bitmap(ofd.FileName));

            OutputImageOnForm();
        }

        private void OutputImageOnForm()
        {
            double[,,] scaledImage = ArrayImageService.getScaledImage(m_image, 10);

            Bitmap scaledBitmap = BitmapConverter.DoubleRgbToBitmap(scaledImage);

            pictureBox1.Image = scaledBitmap;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Vector input = ArrayImageService.formInputVectorForNeuralNetwork(m_image);

            Vector res = m_neuralNetwork.calcResult(input);

            label2.Text = LettersInformator.GetLetterByVector(res).ToString();
            label3.Text = res.ToString();
        }
        
        private void аНуБігомВчитисяToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Teacher teacher = new Teacher(m_neuralNetwork, m_sampleDirectory);

            teacher.Teach();

            chart1.Series[0].Points.DataBindXY(Enumerable.Range(0, teacher.EPOCH_NUMBER).ToArray(),
                                               teacher.Errors);
        }
                
        private void Form1_Load(object sender, EventArgs e)
        {
            //NeuralNetworkBL.NeuralNetwork.Test();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            m_image = ArrayImageService.GetNoisedImage(m_image, 0.05);

            OutputImageOnForm();
        }
    }
}
