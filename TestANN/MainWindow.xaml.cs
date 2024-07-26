using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows;

using SmallBasicANN;
using static System.Net.Mime.MediaTypeNames;

namespace TestANN
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        bool isTrained = false;
        private double scale = 100;
        string name = "";
        string AnnData = "";
        Random rand = new Random();
        ANN aNN = null;

        public MainWindow()
        {
            InitializeComponent();

            name = "Mean";
            AnnData = Directory.GetCurrentDirectory() + "/" + name + "_ANN.txt";
        }

        private void testANN()
        {
            textBoxReport.Text = "";
            textBoxReport.Text += "epoch=" + aNN.Trained + '\n';
            textBoxReport.Text += "BinaryOutput=" + aNN.BinaryOutput + '\n';
            textBoxReport.Text += "Epochs=" + aNN.Epochs + '\n';
            textBoxReport.Text += "LearningRate=" + aNN.LearningRate + '\n';
            textBoxReport.Text += "Momentum=" + aNN.Momentum + '\n';
            textBoxReport.Text += "SigmoidResponse=" + aNN.SigmoidResponse + '\n';
            textBoxReport.Text += "ErrorRequired=" + aNN.ErrorRequired + '\n';
            textBoxReport.Text += '\n';
            for (int i = 0; i < 10; i++)
            {
                int A = rand.Next((int)scale);
                int B = rand.Next((int)scale);
                double C = (A + B) / 2.0;
                string[] result = aNN.Use((A / scale) + "," + (B / scale)).Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                double _C = double.Parse(result[0]) * scale;
                string report = "Mean(" + A + "," + B + ") = " + C + " (" + _C + ")";
                textBoxReport.Text += report + '\n';
            }
        }

        private void buttonTrain_Click(object sender, RoutedEventArgs e)
        {
            int inputNode = 2;
            int hiddenNode = 3;
            int outputNode = 1;
            int[] structure = new int[3] { inputNode, hiddenNode, outputNode };
            string trainingData = Directory.GetCurrentDirectory() + "/" + name + ".txt";

            aNN = new ANN(name, structure);
            aNN.BinaryOutput = false;
            aNN.Epochs = 1000;

            using (StreamWriter sw = new StreamWriter(trainingData))
            {
                for (int i = 0; i < 1000; i++)
                {
                    int A = rand.Next((int)scale);
                    int B = rand.Next((int)scale);
                    double C = (A + B) / 2.0;
                    sw.WriteLine((A / scale) + "," + (B / scale) + "," + (C / scale));
                }
            }

            aNN.Train(trainingData, false);
            aNN.Save(AnnData);
            isTrained = true;

            testANN();
        }

        private void buttonUse_Click(object sender, RoutedEventArgs e)
        {
            if (!isTrained)
            {
                textBoxReport.Text = "ANN is not trained";
            }
            else
            {
                aNN = new ANN(AnnData);
                name = aNN.Name;

                testANN();
            }
        }
    }
}
