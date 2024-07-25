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
        private double scale = 100;
        private bool train = true;
        private string path = "";

        public MainWindow()
        {
            InitializeComponent();

            path = Directory.GetCurrentDirectory();
        }

        private void buttonTest_Click(object sender, RoutedEventArgs e)
        {
            Random rand = new Random();
            string name = "Mean";
            string AnnData = path + "/" + name + "_ANN.txt";
            ANN aNN;

            if (train)
            {
                int inputNode = 2;
                int hiddenNode = 3;
                int outputNode = 1;
                int[] structure = new int[3] { inputNode, hiddenNode, outputNode };
                string trainingData = path + "/" + name + ".txt";

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
                aNN.Save(path);
            }
            else
            {
                aNN = new ANN(AnnData);
                name = aNN.Name;
            }

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
    }
}
