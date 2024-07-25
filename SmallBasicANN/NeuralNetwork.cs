using Microsoft.SmallBasic.Library;
using System;
using System.Collections.Generic;
using System.IO;

namespace SmallBasicANN
{
	/// <summary>
	/// Artificial Neural Network (ANN).
	/// </summary>
	[SmallBasicType]
	public static class NeuralNetwork
	{
		private static Dictionary<Primitive, ANN> dictionaryANN = new Dictionary<Primitive, ANN>();

        /// <summary>
        /// Create a new ANN.
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <param name="Structure">Number of nodes in each layer (comma or space separated list).</param>
        /// <returns>An error message or "True".</returns>
        public static Primitive New(Primitive Name, Primitive Structure)
		{
			if (Name == "")
			{
				return "Error: no network name was specified";
			}
			string[] array = Structure.ToString().Split(new char[] { ',', ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
			if (array.Length < 3)
			{
				return "Error: an invalid number of layers were specified";
			}
			int[] array2 = new int[array.Length];
			for (int i = 0; i < array.Length; i++)
			{
				array[i].Trim();
				try
				{
					array2[i] = Convert.ToInt32(array[i]);
				}
				catch
				{
					return "Error: an invalid Structure parameter was passed";
				}
			}
			ANN value;
			if (dictionaryANN.TryGetValue(Name, out value))
			{
				return "Error: a neural network with that name already exists";
			}
			value = new ANN(Name, array2);
			dictionaryANN.Add(Name, value);
			return "True";
		}

		/// <summary>
		/// Load a trained ANN from a file.
		/// </summary>
		/// <param name="Filename"></param>
		/// <returns>An error message or ANN.</returns>
		public static Primitive Load(Primitive Filename)
		{
			if (Filename == "")
			{
				return "Error: no filename was specified";
			}
			ANN aNN;
			if (dictionaryANN.TryGetValue(Filename, out aNN))
			{
				return "Error: a neural network with that name already exists";
			}
			aNN = new ANN(Filename);
			dictionaryANN.Add(aNN.Name, aNN);
			return aNN.Name;
		}

        /// <summary>
        /// Save a trained ANN to a file.
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <param name="Filename">The text file to save ANN to.</param>
        /// <returns>An error message or "True".</returns>
        public static Primitive Save(Primitive Name, Primitive Filename)
		{
			ANN aNN;
			if (!dictionaryANN.TryGetValue(Name, out aNN))
			{
				return "Error: a neural network with that name doesn't exist";
			}
			return aNN.Save(Filename);
		}

        /// <summary>
        /// Train an ANN.
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <param name="TrainingSet">A file containing training data.</param>
        /// <param name="FileMode">"True" for binary format.</param>
        /// <returns>An error message or the epoch the training finished.</returns>
        public static Primitive Train(Primitive Name, Primitive TrainingSet, Primitive FileMode)
        {
            ANN aNN;
            if (!dictionaryANN.TryGetValue(Name, out aNN))
            {
                return "Error: a neural network with that name doesn't exist";
            }
            return aNN.Train(TrainingSet, FileMode);
        }

        /// <summary>
        /// Train an ANN async (start the training an return immediately.
		/// Use the training event to detect progress and when it has finished.
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <param name="TrainingSet">A file containing training data.</param>
        /// <param name="FileMode">"True" for binary format.</param>
        /// <returns>An error message or 0.</returns>
        [HideFromIntellisense]
        public static Primitive TrainAsync(Primitive Name, Primitive TrainingSet, Primitive FileMode)
        {
            ANN aNN;
            if (!dictionaryANN.TryGetValue(Name, out aNN))
            {
                return "Error: a neural network with that name doesn't exist";
            }
            return aNN.TrainAsync(TrainingSet, FileMode);
        }

        /// <summary>
        /// Use ANN to calculate output from input.
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <param name="Input">A comma or space separated list of input data.</param>
        /// <returns>An error message or space separated list of result data.</returns>
        public static Primitive Use(Primitive Name, Primitive Input)
		{
			if (Input.ToString().Length == 0)
			{
				return "Error: No input specified";
			}
			ANN aNN;
			if (!dictionaryANN.TryGetValue(Name, out aNN))
			{
				return "Error: a neural network with that name doesn't exist";
			}
			return aNN.Use(Input);
		}

        /// <summary>
        /// Delete an ANN.
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <returns>An error message or "True".</returns>
        public static Primitive Delete(Primitive Name)
		{
			ANN aNN;
			if (!dictionaryANN.TryGetValue(Name, out aNN))
			{
				return "Error: a neural network with that name doesn't exist";
			}
			dictionaryANN.Remove(Name);
			return "True";
		}

        /// <summary>
        /// Check if ANN exists.
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <returns>An error message or "True".</returns>
        public static Primitive Exists(Primitive Name)
		{
			ANN aNN;
			if (!dictionaryANN.TryGetValue(Name, out aNN))
			{
				return "Error: a neural network with that name doesn't exist";
			}
			return "True";
		}

        /// <summary>
        /// Conver the format of ANN training data.
        /// </summary>
        /// <param name="Input">The file to convert.</param>
        /// <param name="Output">The file to save converted data to.</param>
        /// <param name="Binary">"True" for binary to ascii.</param>
        /// <returns>An error message or "True".</returns>
        public static Primitive ConvertFile(Primitive Input, Primitive Output, Primitive Binary)
		{
			if (!System.IO.File.Exists(Input))
			{
				return "Error: input file doesn't exist";
			}
			FileStream fileStream;
			try
			{
				fileStream = new FileStream(Input, FileMode.Open, FileAccess.ReadWrite);
			}
			catch
			{
				Primitive result = "Error: couldn't open input file";
				return result;
			}
			FileStream fileStream2;
			try
			{
				fileStream2 = new FileStream(Output, FileMode.Create, FileAccess.ReadWrite);
			}
			catch
			{
				Primitive result = "Error: couldn't create output file";
				return result;
			}
			if (!Binary)
			{
				StreamReader streamReader = new StreamReader(fileStream);
				BinaryWriter binaryWriter = new BinaryWriter(fileStream2);
                int index = 0;
                string[] data = null;
                do
                {
                    binaryWriter.Write(ANN.NextData(streamReader, ref data, ref index));
                }
				while (!streamReader.EndOfStream);
				streamReader.Close();
				binaryWriter.Close();
			}
			else
			{
				BinaryReader binaryReader = new BinaryReader(fileStream);
				StreamWriter streamWriter = new StreamWriter(fileStream2);
				do
				{
					streamWriter.WriteLine(binaryReader.ReadDouble().ToString());
				}
				while (binaryReader.BaseStream.Position < binaryReader.BaseStream.Length - 1L);
				streamWriter.Flush();
				streamWriter.BaseStream.SetLength(streamWriter.BaseStream.Length - 2L);
				binaryReader.Close();
				streamWriter.Close();
			}
			fileStream.Close();
			fileStream2.Close();
			return "True";
		}

        /// <summary>
        /// Get the number of input nodes.
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <returns>An error message or the number of input nodes.</returns>
        public static Primitive InputNodes(Primitive Name)
		{
			ANN aNN;
			if (!dictionaryANN.TryGetValue(Name, out aNN))
			{
				return "Error: a neural network with that name doesn't exist";
			}
			return (double)aNN.InputNodes;
		}

        /// <summary>
        /// Get the number of hidden layers.
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <returns>An error message or the number of hidden layers.</returns>
        public static Primitive HiddenLayers(Primitive Name)
		{
			ANN aNN;
			if (!dictionaryANN.TryGetValue(Name, out aNN))
			{
				return "Error: a neural network with that name doesn't exist";
			}
			return (double)aNN.HiddenLayers;
		}

        /// <summary>
        /// Get the number of nodes on a hidden layer.
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <param name="Layer">The hidden layer number.</param>
        /// <returns>An error message or the number of nodes on hidden layer.</returns>
        public static Primitive HiddenNodes(Primitive Name, Primitive Layer)
		{
			ANN aNN;
			if (!dictionaryANN.TryGetValue(Name, out aNN))
			{
				return "Error: a neural network with that name doesn't exist";
			}
			return (double)aNN.HiddenNodes(Layer);
		}

        /// <summary>
        /// Get the number of output nodes.
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <returns>An error message or the number of output nodes.</returns>
        public static Primitive OutputNodes(Primitive Name)
		{
			ANN aNN;
			if (!dictionaryANN.TryGetValue(Name, out aNN))
			{
				return "Error: a neural network with that name doesn't exist";
			}
			return (double)aNN.OutputNodes;
		}

        /// <summary>
        /// Get the epoch of a trained ANN.
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <returns>An error message or the epoch of a trained ANN.</returns>
        public static Primitive Trained(Primitive Name)
		{
			ANN aNN;
			if (!dictionaryANN.TryGetValue(Name, out aNN))
			{
				return "Error: a neural network with that name doesn't exist";
			}
			return (double)aNN.Trained;
		}

        /// <summary>
        /// Change the name of a ANN.
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <param name="Value">The new name.</param>
        /// <returns>An error message or the new ANN name.</returns>
        public static Primitive Name(Primitive Name, Primitive Value)
		{
			ANN aNN;
			if (!dictionaryANN.TryGetValue(Name, out aNN))
			{
				return "Error: a neural network with that name doesn't exist";
			}
			aNN.Name = Value;
			dictionaryANN.Remove(Name);
			dictionaryANN.Add(Value, aNN);
			return aNN.Name;
		}

        /// <summary>
        /// Get or set the max number of epochs.
		/// The epoch is the number of times the training data is re-used, default 100000.
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <param name="Value">The requested data.</param>
        /// <param name="Get">"True" to get the requested data..</param>
        /// <returns>An error message or the requested data.</returns>
        public static Primitive Epochs(Primitive Name, Primitive Value, Primitive Get)
		{
			ANN aNN;
			if (!dictionaryANN.TryGetValue(Name, out aNN))
			{
				return "Error: a neural network with that name doesn't exist";
			}
			if (!Get)
			{
				aNN.Epochs = Value;
			}
			return (double)aNN.Epochs;
		}

        /// <summary>
        /// Get or set the learning rate.
		/// The learning rate is the amount errors are back propagated to node weights, default 0.7.
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <param name="Value">The requested data.</param>
        /// <param name="Get">"True" to get the requested data.</param>
        /// <returns>An error message or the requested data.</returns>
		public static Primitive LearningRate(Primitive Name, Primitive Value, Primitive Get)
		{
			ANN aNN;
			if (!dictionaryANN.TryGetValue(Name, out aNN))
			{
				return "Error: a neural network with that name doesn't exist";
			}
			if (!Get)
			{
				aNN.LearningRate = Value;
			}
			return aNN.LearningRate;
		}

        /// <summary>
        /// Get or set the momentum.
		/// The momentun is the amount previous iteration errors are preserved during back propagation, default 0.3.
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <param name="Value">The requested data.</param>
        /// <param name="Get">"True" to get the requested data.</param>
        /// <returns>An error message or the requested data.</returns>
        public static Primitive Momentum(Primitive Name, Primitive Value, Primitive Get)
		{
			ANN aNN;
			if (!dictionaryANN.TryGetValue(Name, out aNN))
			{
				return "Error: a neural network with that name doesn't exist";
			}
			if (!Get)
			{
				aNN.Momentum = Value;
			}
			return aNN.Momentum;
		}

        /// <summary>
        /// Get or set the sigmoid response.
		/// The sigmoid response is the smoothing of values outside [-1,1] in the forward feed, defaut 1.0.
		/// Sigmoid = 1/(1+Exp(-weight/sigmoid))
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <param name="Value">The requested data.</param>
        /// <param name="Get">"True" to get the requested data.</param>
        /// <returns>An error message or the requested data.</returns>
		public static Primitive SigmoidResponse(Primitive Name, Primitive Value, Primitive Get)
		{
			ANN aNN;
			if (!dictionaryANN.TryGetValue(Name, out aNN))
			{
				return "Error: a neural network with that name doesn't exist";
			}
			if (!Get)
			{
				aNN.SigmoidResponse = Value;
			}
			return (decimal)aNN.SigmoidResponse;
		}

        /// <summary>
        /// Get or set the required RMS error.
		/// The RMS error is current RootMeanSquare deviation in trained output layer values, default 0.0001.
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <param name="Value">The requested data.</param>
        /// <param name="Get">"True" to get the requested data.</param>
        /// <returns>An error message or the requested data.</returns>
        public static Primitive ErrorRequired(Primitive Name, Primitive Value, Primitive Get)
        {
            ANN aNN;
            if (!dictionaryANN.TryGetValue(Name, out aNN))
            {
                return "Error: a neural network with that name doesn't exist";
            }
            if (!Get)
            {
                aNN.ErrorRequired = Value;
            }
            return (decimal)aNN.ErrorRequired;
        }

        /// <summary>
        /// Get or set a random seed for the initial node weights, to have reproducible results.
		/// If &lt; 0, then a time dependent seed is used, default -1.
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <param name="Value">The requested data.</param>
        /// <param name="Get">"True" to get the requested data.</param>
        /// <returns>An error message or the requested data.</returns>
        public static Primitive RandomSeed(Primitive Name, Primitive Value, Primitive Get)
        {
            ANN aNN;
            if (!dictionaryANN.TryGetValue(Name, out aNN))
            {
                return "Error: a neural network with that name doesn't exist";
            }
            if (!Get)
            {
                aNN.RandomSeed = Value;
            }
            return (decimal)aNN.RandomSeed;
        }

        /// <summary>
        /// Get or set wether the data is binary "True" or not.
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <param name="Value">The requested data.</param>
        /// <param name="Get">"True" to get the requested data.</param>
        /// <returns>An error message or the requested data.</returns>
        public static Primitive BinaryOutput(Primitive Name, Primitive Value, Primitive Get)
        {
            ANN aNN;
            if (!dictionaryANN.TryGetValue(Name, out aNN))
            {
                return "Error: a neural network with that name doesn't exist";
            }
            if (!Get)
            {
                aNN.BinaryOutput = Value;
            }
            return aNN.BinaryOutput;
        }

		internal static int progressReportFrequency = 1000;
        internal static Primitive lastANN = "";
        internal static Primitive lastANNProgress = "";
        internal static SmallBasicCallback _ANNProgressDelegate = null;
		/// <summary>
		/// The progress reporting event frequency, number of test data consumed between each report, default 1000.
		/// Zero or a negative number only reports at the end of each epoch.
		/// </summary>
		public static Primitive ProgressReportFrequency
		{
			get { return progressReportFrequency; }
			set { progressReportFrequency = value; }
		}
        /// <summary>
        /// The last ANN to report progress.
        /// </summary>
        public static Primitive LastANN
        {
            get { return lastANN; }
        }
        /// <summary>
        /// The last progress message while training.
		/// Array of "Epoch", "% trained", "RMS error".
		/// Note that RMS error doesn't have much significance until 100% trained at the end of each epoch.
        /// </summary>
        public static Primitive LastANNMessage
        {
            get { return lastANNProgress; }
        }
        /// <summary>
        /// Progress event every 1000 trained items.
		/// Change the default progress report frequency with ProgressReportFrequency
        /// </summary>
        public static event SmallBasicCallback ANNProgress
        {
            add { _ANNProgressDelegate = value; }
            remove { _ANNProgressDelegate = null; }
        }

        /// <summary>
        /// Get the trained node weights.
        /// </summary>
        /// <param name="Name">The ANN name.</param>
        /// <returns>An array with node (and bias node) weights.
        /// array[layer][node][node+]
        /// layer is input and hidden layers, indexed from 1
        /// node is the nodes in layer, indexed from 1
		/// node+ is the nodes in layer+1, indexed from 1
        /// </returns>
        public static Primitive GetWeights(Primitive Name)
		{
            ANN aNN;
            if (!dictionaryANN.TryGetValue(Name, out aNN))
            {
                return "Error: a neural network with that name doesn't exist";
            }
            return aNN.DumpWeights2();
        }
    }
}
