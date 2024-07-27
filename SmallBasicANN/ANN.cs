using Microsoft.SmallBasic.Library;
using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices.ComTypes;
using System.Threading;

namespace SmallBasicANN
{
    public class ANN
    {
        private string name;
        private int[] structure;
        private double[][,] connections;
        private int trained;
        private bool binaryOutput;
        private int epochs;
        private double learningRate;
        private double momentum;
        private double sigmoidResponse;
        private double errorRequired;
        private double[][] nodes;
        private int numLayers;
        private int numHidden;
        private int numConnections;
        private int outputLayer;
        private double[][] error;
        private double[][] oldError;
        private double[] idealOutput;
        private double RMSerror;
        private int numBias;
        private int randomSeed;
        private bool isInitialised;

        public int InputNodes
        {
            get
            {
                return structure[0];
            }
        }

        public int HiddenLayers
        {
            get
            {
                return numHidden;
            }
        }

        public int OutputNodes
        {
            get
            {
                return structure[structure.Length - 1];
            }
        }

        public int Trained
        {
            get
            {
                return trained;
            }
        }

        public string Name
        {
            get
            {
                return name;
            }
            set
            {
                name = value;
            }
        }

        public int Epochs
        {
            get
            {
                return epochs;
            }
            set
            {
                epochs = value;
            }
        }

        public double LearningRate
        {
            get
            {
                return learningRate;
            }
            set
            {
                learningRate = value;
            }
        }

        public double Momentum
        {
            get
            {
                return momentum;
            }
            set
            {
                momentum = value;
            }
        }

        public double SigmoidResponse
        {
            get
            {
                return sigmoidResponse;
            }
            set
            {
                sigmoidResponse = value;
            }
        }

        public double ErrorRequired
        {
            get
            {
                return errorRequired;
            }
            set
            {
                errorRequired = value;
            }
        }

        public bool BinaryOutput
        {
            get
            {
                return binaryOutput;
            }
            set
            {
                binaryOutput = value;
            }
        }

        public int RandomSeed
        {
            get
            {
                return randomSeed;
            }
            set
            {
                randomSeed = value;
            }
        }

        private void InitialiseConnections()
        {
            Random random;
            if (randomSeed < 0)
                random = new Random();
            else
                random = new Random(randomSeed);
            connections = new double[numConnections][,];
            for (int i = 0; i < numConnections; i++)
            {
                connections[i] = new double[structure[i] + numBias, structure[i + 1]];
                for (int j = 0; j < structure[i] + numBias; j++)
                {
                    for (int k = 0; k < structure[i + 1]; k++)
                    {
                        connections[i][j, k] = random.NextDouble();
                        if (random.Next(2) == 0)
                        {
                            connections[i][j, k] *= -1.0;
                        }
                    }
                }
            }
            isInitialised = true;
        }

        private void FeedForward()
        {
            for (int i = 0; i < numLayers - 1; i++)
            {
                for (int j = 0; j < structure[i + 1]; j++)
                {
                    double value = 0;
                    for (int k = 0; k < structure[i]; k++)
                    {
                        value += nodes[i][k] * connections[i][k, j];
                    }
                    value += connections[i][structure[i], j];
                    nodes[i + 1][j] = 1.0 / (1.0 + System.Math.Exp(-1.0 * (value / sigmoidResponse)));
                }
            }
        }

        private void Backpropagation()
        {
            for (int i = 0; i < structure[outputLayer]; i++)
            {
                error[outputLayer - 1][i] = nodes[outputLayer][i] * (1.0 - nodes[outputLayer][i]) * (idealOutput[i] - nodes[outputLayer][i]);
                int j = 0;
                while (j < structure[numHidden])
                {
                    connections[numConnections - 1][j, i] += learningRate * error[outputLayer - 1][i] * nodes[numHidden][j] + momentum * oldError[outputLayer - 1][i];
                    j++;
                }
                connections[numConnections - 1][structure[numHidden], i] += learningRate * error[outputLayer - 1][i] * 1.0 + momentum * oldError[outputLayer - 1][i];
                RMSerror += error[outputLayer - 1][i] * error[outputLayer - 1][i];
            }
            for (int i = numHidden; i > 0; i--)
            {
                for (int j = 0; j < structure[i]; j++)
                {
                    double value = 0;
                    for (int k = 0; k < structure[i + 1]; k++)
                    {
                        value += error[i][k] * connections[i][j, k];
                    }
                    error[i - 1][j] = nodes[i][j] * (1.0 - nodes[i][j]) * value;
                    for (int k = 0; k < structure[i - 1]; k++)
                    {
                        connections[i - 1][k, j] += learningRate * error[i - 1][j] * nodes[i - 1][k] + momentum * oldError[i - 1][j];
                    }
                    connections[i - 1][structure[i - 1], j] += learningRate * error[i - 1][j] * 1.0 + momentum * oldError[i - 1][j];
                }
            }
        }

        public ANN(string n, int[] l)
        {
            Reset();
            name = n;
            structure = l;
            numLayers = structure.Length;
            numHidden = numLayers - 2;
            numConnections = numLayers - 1;
            outputLayer = numLayers - 1;
            nodes = new double[numLayers][];
            for (int i = 0; i < numLayers; i++)
            {
                nodes[i] = new double[structure[i]];
            }
        }

        private void Reset()
        {
            trained = 0;
            binaryOutput = true;
            epochs = 100000;
            learningRate = 0.7;
            momentum = 0.3;
            sigmoidResponse = 1.0;
            errorRequired = 0.0001;
            numBias = 1;
            randomSeed = -1;
            isInitialised = false;
        }

        public ANN(string fn)
        {
            Reset();

            int i, j, k;
            FileStream fileStream;
            try
            {
                fileStream = new FileStream(fn, FileMode.Open, FileAccess.Read);
            }
            catch
            {
                return;
            }
            StreamReader streamReader = new StreamReader(fileStream);
            do
            {
                string text = streamReader.ReadLine();
                string[] array = text.Split(new char[] { ':' }, StringSplitOptions.RemoveEmptyEntries);
                if (array.Length == 2)
                {
                    array[0] = array[0].Trim();
                    array[0] = array[0].ToLower();
                    array[1] = array[1].Trim();
                    if (array[0] == "name")
                    {
                        name = array[1];
                    }
                    else if (array[0] == "structure")
                    {
                        string[] array2 = array[1].Split(new char[] { ',', ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                        structure = new int[array2.Length];
                        for (i = 0; i < structure.Length; i++)
                        {
                            array2[i] = array2[i].Trim();
                            structure[i] = Convert.ToInt32(array2[i]);
                        }
                    }
                    else if (array[0] == "trained")
                    {
                        trained = Convert.ToInt32(array[1]);
                    }
                    else if (array[0] == "binary output")
                    {
                        binaryOutput = Convert.ToBoolean(array[1]);
                    }
                    else if (array[0] == "epochs")
                    {
                        epochs = Convert.ToInt32(array[1]);
                    }
                    else if (array[0] == "learning rate")
                    {
                        learningRate = Convert.ToDouble(array[1]);
                    }
                    else if (array[0] == "momentum")
                    {
                        momentum = Convert.ToDouble(array[1]);
                    }
                    else if (array[0] == "sigmoid response")
                    {
                        sigmoidResponse = Convert.ToDouble(array[1]);
                    }
                    else if (array[0] == "error required")
                    {
                        errorRequired = Convert.ToDouble(array[1]);
                    }
                    else if (array[0] == "random seed")
                    {
                        randomSeed = Convert.ToInt32(array[1]);
                    }
                }
            }
            while (!streamReader.EndOfStream);
            numLayers = (int)structure.Length;
            numHidden = numLayers - 2;
            numConnections = numLayers - 1;
            outputLayer = numLayers - 1;
            nodes = new double[numLayers][];
            for (i = 0; i < numLayers; i++)
            {
                nodes[i] = new double[structure[i]];
            }
            connections = new double[numConnections][,];
            i = 0;
            while (i < numConnections)
            {
                connections[i] = new double[structure[i] + numBias, structure[i + 1]];
                i++;
            }
            streamReader.BaseStream.Position = 0L;
            do
            {
                string text2 = streamReader.ReadLine();
                text2 = text2.ToLower();
                text2 = text2.Trim();
                if (text2.IndexOf("synapses:") > -1 && text2.Length == 9)
                {
                    i = 0;
                    while (i < numConnections)
                    {
                        j = 0;
                        while (j < structure[i] + numBias)
                        {
                            k = 0;
                            while (k < structure[i + 1])
                            {
                                connections[i][j, k] = Convert.ToDouble(streamReader.ReadLine());
                                k++;
                            }
                            j++;
                        }
                        i++;
                    }
                }
            }
            while (!streamReader.EndOfStream);
            streamReader.Close();
            fileStream.Close();
            isInitialised = true;
        }

        public bool Save(string fn)
        {
            if (!isInitialised) InitialiseConnections();
            FileStream fileStream;
            try
            {
                fileStream = new FileStream(fn, FileMode.Create, FileAccess.Write);
            }
            catch
            {
                return false;
            }
            StreamWriter streamWriter = new StreamWriter(fileStream);
            streamWriter.WriteLine("Name: " + name);
            string text = "";
            int[] array = structure;
            for (int i = 0; i < array.Length; i++)
            {
                int num = array[i];
                text = text + num + ",";
            }
            text = text.Remove(text.Length - 1);
            streamWriter.WriteLine("Structure: " + text);
            streamWriter.WriteLine("Trained: " + trained);
            streamWriter.WriteLine("Binary Output: " + binaryOutput);
            streamWriter.WriteLine("Epochs: " + epochs);
            streamWriter.WriteLine("Learning Rate: " + learningRate);
            streamWriter.WriteLine("Momentum: " + momentum);
            streamWriter.WriteLine("Sigmoid Response: " + sigmoidResponse);
            streamWriter.WriteLine("Error Required: " + errorRequired);
            streamWriter.WriteLine("Random Seed: " + randomSeed);
            streamWriter.WriteLine();
            streamWriter.WriteLine("Synapses:");
            for (int i = 0; i < connections.Length; i++)
            {
                double[,] array1 = connections[i];
                for (int j = array1.GetLowerBound(0); j <= array1.GetUpperBound(0); j++)
                {
                    for (int k = array1.GetLowerBound(1); k <= array1.GetUpperBound(1); k++)
                    {
                        streamWriter.WriteLine(array1[j, k]);
                    }
                }
            }
            streamWriter.Write(text);
            streamWriter.Close();
            fileStream.Close();
            return true;
        }

        public Primitive TrainAsync(string fn, bool fm)
        {
            Thread thread = new Thread(new ParameterizedThreadStart(DoTrain));
            thread.Start(new Object[] { fn, fm });
            return 0;
        }

        private void DoTrain(object obj)
        {
            string fn = (string)((object[])obj)[0];
            bool fm = (bool)((object[])obj)[1];
            Train(fn, fm);
        }

        public static double NextData(StreamReader streamReader, ref string[] data, ref int index)
        {
            index++;
            while (data == null || index >= data.Length)
            {
                index = 0;
                data = streamReader.ReadLine().Split(new char[] { ',', ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            }
            return Convert.ToDouble(data[index]);
        }

        public string Train(string fn, bool fm)
        {
            if (!isInitialised) InitialiseConnections();
            trained = epochs;
            if (!System.IO.File.Exists(fn))
            {
                return "Error: training set file doesn't exist";
            }
            FileStream fileStream = new FileStream(fn, FileMode.Open, FileAccess.Read);
            BinaryReader binaryReader = new BinaryReader(fileStream);
            StreamReader streamReader = new StreamReader(fileStream);
            int numTrial = 0;
            double[,] trialInput;
            double[,] trialOutput;
            if (fm)
            {
                if (fileStream.Length % (structure[0] + structure[numLayers - 1]) * 8 != 0L)
                {
                    fileStream.Close();
                    binaryReader.Close();
                    streamReader.Close();
                    return "Error: training set data size not valid for this network";
                }
                numTrial = (int)(fileStream.Length / ((structure[0] + structure[numLayers - 1]) * 8));
                binaryReader.BaseStream.Position = 0L;
                trialInput = new double[numTrial, structure[0]];
                trialOutput = new double[numTrial, structure[outputLayer]];
                for (int trial = 0; trial < numTrial; trial++)
                {
                    for (int i = 0; i < structure[0]; i++)
                    {
                        trialInput[trial, i] = binaryReader.ReadDouble();
                    }
                    for (int i = 0; i < structure[outputLayer]; i++)
                    {
                        trialOutput[trial, i] = binaryReader.ReadDouble();
                    }
                }
                binaryReader.Close();
            }
            else
            {
                int i = 0;
                do
                {
                    string text = streamReader.ReadLine();
                    if (text.Length > 0)
                    {
                        i += text.Split(new char[] { ',', ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries).Length;
                    }
                }
                while (!streamReader.EndOfStream);
                if (i % (structure[0] + structure[numLayers - 1]) != 0)
                {
                    fileStream.Close();
                    binaryReader.Close();
                    streamReader.Close();
                    return "Error: training set data size not valid for this network";
                }
                numTrial = i / (structure[0] + structure[numLayers - 1]);
                streamReader.BaseStream.Position = 0L;
                trialInput = new double[numTrial, structure[0]];
                trialOutput = new double[numTrial, structure[outputLayer]];
                int index = 0;
                string[] data = null;

                for (int trial = 0; trial < numTrial; trial++)
                {
                    for (i = 0; i < structure[0]; i++)
                    {
                        trialInput[trial, i] = NextData(streamReader, ref data, ref index);
                    }
                    for (i = 0; i < structure[outputLayer]; i++)
                    {
                        trialOutput[trial, i] = NextData(streamReader, ref data, ref index);
                    }
                }
                streamReader.Close();
            }
            fileStream.Close();
            error = new double[numLayers - 1][];
            oldError = new double[numLayers - 1][];
            for (int i = 0; i < numLayers - 1; i++)
            {
                error[i] = new double[structure[i + 1] + numBias];
                oldError[i] = new double[structure[i + 1] + numBias];
            }
            idealOutput = new double[structure[outputLayer]];
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                RMSerror = 0.0;
                for (int i = 0; i < numLayers - 1; i++)
                {
                    for (int j = 0; j < structure[i + 1] + numBias; j++)
                    {
                        oldError[i][j] = 0.0;
                    }
                }
                for (int trial = 0; trial < numTrial; trial++)
                {
                    for (int i = 0; i < structure[0]; i++)
                    {
                        nodes[0][i] = trialInput[trial, i];
                    }
                    for (int i = 0; i < structure[outputLayer]; i++)
                    {
                        idealOutput[i] = trialOutput[trial, i];
                    }
                    FeedForward();
                    Backpropagation();
                    for (int i = 0; i < numLayers - 1; i++)
                    {
                        int j = 0;
                        while (j < structure[i + 1] + numBias)
                        {
                            oldError[i][j] = error[i][j];
                            j++;
                        }
                    }
                    RMSerror /= numTrial;
                    RMSerror = System.Math.Sqrt(RMSerror);
                    if (null != NeuralNetwork._ANNProgressDelegate && NeuralNetwork.progressReportFrequency > 0 && trial % NeuralNetwork.progressReportFrequency == 0)
                    {
                        NeuralNetwork.lastANN = name;
                        NeuralNetwork.lastANNProgress["Epoch"] = epoch + 1;
                        NeuralNetwork.lastANNProgress["% trained"] = 100 * (double)(trial) / (double)(numTrial);
                        NeuralNetwork.lastANNProgress["RMS error"] = RMSerror;
                        NeuralNetwork._ANNProgressDelegate();
                    }
                }
                if (null != NeuralNetwork._ANNProgressDelegate)
                {
                    NeuralNetwork.lastANN = name;
                    NeuralNetwork.lastANNProgress["Epoch"] = epoch + 1;
                    NeuralNetwork.lastANNProgress["% trained"] = 100;
                    NeuralNetwork.lastANNProgress["RMS error"] = RMSerror;
                    NeuralNetwork._ANNProgressDelegate();
                }
                if (RMSerror < errorRequired)
                {
                    epoch = epochs;
                }
            }
            if (null != NeuralNetwork._ANNProgressDelegate)
            {
                NeuralNetwork.lastANNProgress = "Finished";
                NeuralNetwork._ANNProgressDelegate();
            }
            return trained.ToString();
        }

        public string Use(string Input)
        {
            if (!isInitialised) InitialiseConnections();
            int i;
            string[] array = Input.ToString().Split(new char[] { ',', ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            if (array.Length != nodes[0].Length)
            {
                return "Error: An invalid amount of inputs was specified";
            }
            for (i = 0; i < array.Length; i++)
            {
                array[i].Trim();
                try
                {
                    nodes[0][i] = Convert.ToDouble(array[i]);
                }
                catch (FormatException)
                {
                    return "Error: An invalid Input parameter was passed";
                }
            }
            FeedForward();
            if (binaryOutput)
            {
                i = 0;
                while (i < structure[outputLayer])
                {
                    if (nodes[outputLayer][i] < 0.5)
                    {
                        nodes[outputLayer][i] = 0.0;
                    }
                    else
                    {
                        nodes[outputLayer][i] = 1.0;
                    }
                    i++;
                }
            }
            string text = "";
            i = 0;
            while (i < structure[outputLayer])
            {
                text = text + nodes[outputLayer][i].ToString() + " ";
                i++;
            }
            return text;
        }

        public int HiddenNodes(int i)
        {
            return structure[i];
        }

        public string DumpWeights()
        {
            if (!isInitialised) InitialiseConnections();
            string text = "";
            int i = 0;
            while (i < numConnections)
            {
                int j = 0;
                while (j < structure[i] + numBias)
                {
                    int k = 0;
                    while (k < structure[i + 1])
                    {
                        string text2 = text;
                        text = string.Concat(new string[]
                        {
                            text2,
                            "[",
                            i.ToString(),
                            "][",
                            j.ToString(),
                            ",",
                            k.ToString(),
                            "]\t",
                            connections[i][j, k].ToString(),
                            "\n"
                        });
                        k++;
                    }
                    j++;
                }
                i++;
            }
            return text.Remove(text.Length - 1);
        }

        public Primitive DumpWeights2()
        {
            if (!isInitialised) InitialiseConnections();
            Primitive weights = "";
            int i = 0;
            while (i < numConnections)
            {
                Primitive arr1 = "";
                int j = 0;
                while (j < structure[i] + numBias)
                {
                    Primitive arr2 = "";
                    int k = 0;
                    while (k < structure[i + 1])
                    {
                        arr2[k + 1] = connections[i][j, k];
                        k++;
                    }
                    arr1[++j] = arr2;
                }
                weights[++i] = arr1;
            }
            return weights;

        }
    }
}
