/*
*    This program is free software; you can redistribute it and/or modify
*    it under the terms of the GNU General Public License as published by
*    the Free Software Foundation; either version 2 of the License, or
*    (at your option) any later version.
*
*    This program is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU General Public License for more details.
*
*    You should have received a copy of the GNU General Public License
*    along with this program; if not, write to the Free Software
*    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/

/*
*    BPMLL.java
*    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
*
*/

package mulan.classifier.neural;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.List;
import java.util.Set;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.neural.model.ActivationTANH;
import mulan.classifier.neural.model.BasicNeuralNet;
import mulan.classifier.neural.model.NeuralNet;
import mulan.core.WekaException;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.experiment.Stats;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;

/**
 * The implementation of Back-Propagation Multi-Label Learning (BPMLL) learner.
 * The learned model is stored in {@link NeuralNet} neural network. The models of the
 * learner is built by {@link BPMLLAlgorithm} from given training data set.
 * 
 * @author Jozef Vilcek
 * @see BPMLLAlgorithm
 */
public class BPMLL extends MultiLabelLearnerBase {

	private static final long serialVersionUID = 2153814250172139021L;
	private static final double NET_BIAS = 1;
	private static final double ERROR_SMALL_CHANGE = 0.000001;
	
	// filter used to convert nominal input attributes into binary-numeric 
	private NominalToBinary nominalToBinaryFilter;
	
	// algorithm parameters 
	private int epochs = 100;
	private double weightsDecayCost = 0.00001;
	private double learningRate = 0.05;
	private int[] hiddenLayersTopology;
	
	// members related to normalization or attributes
	private boolean normalizeAttributes = true;
	private Normalizer normalizer; 

	private NeuralNet model;
	private ThresholdFunction thresholdF;
	
	/**
	 * Sets the topology of hidden layers for neural network. 
	 * The length of passed array defines number of hidden layers. 
	 * The value at particular index of array defines number of neurons in that layer.
	 * If <code>null</code> is specified, no hidden layers will be created.
	 * <br/>
	 * The network is created when learner is being built.
	 * The input and output layer is determined from input training data.
	 * 
	 * @param hiddenLayers
	 * @throws IllegalArgumentException if any value in the array is less or equal to zero
	 */
	public void setHiddenLayers(int[] hiddenLayers){
		if(hiddenLayers != null){
			for(int value : hiddenLayers){
				if(value <= 0){
					throw new IllegalArgumentException("Invalid hidden layer topology definition. " +
							"Number of neurons in hidden layer must be larger than zero.");
				}
			}
		}
		hiddenLayersTopology = hiddenLayers;
	}
	
	/**
	 * Sets the learning rate. Must be greater than 0 and no more than 1.<br/>
	 * Default value is 0.05.
	 * 
	 * @param learningRate the learning rate
	 * @throws IllegalArgumentException if passed value is invalid
	 */
	public void setLearningRate(double learningRate){
		if(learningRate <= 0 || learningRate > 1){
			throw new IllegalArgumentException("The learning rate must be greater than 0 and no more than 1. " +
					"Entered value is : " + learningRate);
		}
		this.learningRate = learningRate;
	}
	
	/**
	 * Sets the regularization cost term for weights decay. 
	 * Must be greater than 0 and no more than 1.<br/>
	 * Default value is 0.00001.
	 * 
	 * @param weightsDecayCost the weights decay cost term
	 * @throws IllegalArgumentException if passed value is invalid
	 */
	public void setWeightsDecayRegularization(double weightsDecayCost){
		if(weightsDecayCost <= 0 || weightsDecayCost > 1){
			throw new IllegalArgumentException("The weights decay regularization cost " +
					"term must be greater than 0 and no more than 1. " +
					"The passed  value is : " + weightsDecayCost);
		}
		this.weightsDecayCost = weightsDecayCost;
	}
	
	/**
	 * Sets the number of training epochs. Must be greater than 0.<br/>
	 * Default value is 100.
	 * 
	 * @param epochs the number of training epochs
	 * @throws IllegalArgumentException if passed value is invalid
	 */
	public void setTrainingEpochs(int epochs){
		if(epochs <= 0){
			throw new IllegalArgumentException("The learning rate must be greater than zero. " +
					"Entered value is : " + epochs);
		}
		this.epochs = epochs;
	}
	
	/**
	 * Sets whether attributes of instances data (except label attributes) should 
	 * be normalized prior to building the learner. Normalization is performed 
	 * on numeric attributes to the range {-1,1}).<br/> 
	 * When making prediction, attributes of passed input instance are also 
	 * normalized prior to making prediction.<br/>
	 * Default is true (normalization of attributes takes place).
	 * 
	 * @param normalize flag if normalization of attributes should be used
	 * @throws IllegalArgumentException if passed value is invalid
	 */
	public void setNormalizeAttributes(boolean normalize){
		normalizeAttributes = normalize;
	}

	protected void buildInternal(final MultiLabelInstances instances) throws Exception {
		
		if(instances == null){
			throw new IllegalArgumentException("Instances must not be null.");
		}
		
		// delete filter if available from previous build, a new one will be created if necessary
		nominalToBinaryFilter = null;
		
		MultiLabelInstances trainInstances = instances.clone();
		List<DataPair> trainData = prepareData(trainInstances);
		int inputsDim = trainData.get(0).getInput().length;
		model = buildNeuralNetwork(inputsDim); 
		BPMLLAlgorithm learnAlg = new BPMLLAlgorithm(model, weightsDecayCost);
		
		int numInstances = trainData.size();
		int processedInstances = 0;
		double prevError = Double.MAX_VALUE;
		double error = 0;
		for(int epoch = 0; epoch < epochs; epoch++){
			Collections.shuffle(trainData);
			for(int index = 0; index < numInstances; index++){
				DataPair trainPair = trainData.get(index);
				double result = learnAlg.learn(trainPair.getInput(), trainPair.getOutput(), learningRate);
				if(!Double.isNaN(result)){
					error += result;
					processedInstances++;
				}
			}

			if(getDebug()){
				if(epoch%10 == 0)
				debug("Training epoch : " + epoch + "  Model error : " + error/processedInstances);
			}

			double errorDiff = prevError - error;
			if(errorDiff <= ERROR_SMALL_CHANGE * prevError){
				if(getDebug()){
					debug("Global training error does not decrease enough. Training terminated.");
				}
				break;
			}
		}
		
		thresholdF = buildThresholdFunction(trainData);
	}
	
	@Override
	public TechnicalInformation getTechnicalInformation() {
		
		TechnicalInformation technicalInfo = new TechnicalInformation(Type.ARTICLE);
	    technicalInfo.setValue(Field.AUTHOR, "Zhang, M.L., Zhou, Z.H.");
	    technicalInfo.setValue(Field.YEAR, "2006");
	    technicalInfo.setValue(Field.TITLE, "Multi-label neural networks with applications to functional genomics and text categorization");
	    technicalInfo.setValue(Field.JOURNAL, "IEEE Transactions on Knowledge and Data Engineering");
	    technicalInfo.setValue(Field.VOLUME, "18");
	    technicalInfo.setValue(Field.PAGES, "1338-1351");
	    return technicalInfo;
	}
	
	private ThresholdFunction buildThresholdFunction(List<DataPair> trainData){
		
		int numExamples = trainData.size();
		double[][] idealLabels = new double[numExamples][numLabels];
		double[][] modelConfidences = new double[numExamples][numLabels];
		
		for(int example = 0; example < numExamples; example++){
			DataPair dataPair = trainData.get(example);
			idealLabels[example] = dataPair.getOutput();
			modelConfidences[example] = model.feedForward(dataPair.getInput());
		}
		
		return new ThresholdFunction(idealLabels, modelConfidences);
	}
	
	private NeuralNet buildNeuralNetwork(int inputsDim){
		
		int[] networkTopology;
		if(hiddenLayersTopology == null){
			networkTopology = new int[]{inputsDim, numLabels};
		}
		else {
			networkTopology = new int[hiddenLayersTopology.length + 2];
			networkTopology[0] = inputsDim;
			System.arraycopy(hiddenLayersTopology, 0, networkTopology, 1, hiddenLayersTopology.length);
			networkTopology[networkTopology.length - 1] = numLabels;
		}
		NeuralNet model = new BasicNeuralNet(networkTopology,NET_BIAS, ActivationTANH.class);
		
		return model;
	}
	
	/**
	 * Prepares {@link Instances} data for the learning algorithm.
	 * <br/>
	 * The data are checked for correct format, label attributes 
	 * are converted to bipolar values. Finally {@link Instance} instances are 
	 * converted to {@link DataPair} instances, which will be used for the algorithm. 
	 */
	private List<DataPair> prepareData(MultiLabelInstances mlData){
		
		Instances data = mlData.getDataSet();
		if(!checkAttributesFormat(data, mlData.getFeatureAttributes())){
			throw new IllegalArgumentException("Attributes are not in correct format. " +
					"Input attributes (all but the label attributes) must be numeric.");
		}
		
		if(normalizeAttributes){
			normalizer = new Normalizer(mlData, true);
		}
		
		int numInstances = data.numInstances();
		 
		
		int[] featureIndices = mlData.getFeatureIndices();
		int numFeatures = featureIndices.length;
		List<DataPair> dataPairs = new ArrayList<DataPair>();
		for(int index = 0; index < numInstances; index++){
			Instance instance = data.instance(index);
			
			double[] input = new double[numFeatures];
			for(int i = 0; i < numFeatures; i++){
				int featureIndex = featureIndices[i];
				Attribute featureAttr = instance.attribute(featureIndex);
				if(featureAttr.isNominal()){
					input[i] = Double.parseDouble(instance.stringValue(featureIndex));
				}
				else{
					input[i] = instance.value(featureIndex);
				}
			}
			
			double[] output = new double[numLabels];
			for(int i = 0; i < numLabels; i++){
				double value = instance.value(labelIndices[i]);
				output[i] = value == 0 ? -1 : value;
			}

			dataPairs.add(new DataPair(input, output));
		}
		
		return dataPairs;
	}
	
	
	/**
	 * Checks {@link Instances} data if attributes (all but the label attributes) 
	 * are numeric or nominal. Nominal attributes are transformed to binary by use of
	 * {@link NominalToBinary} filter.
	 * 
	 * @param dataSet instances data to be checked
	 * @param inputAttributes input/feature attributes which format need to be checked
	 * @return true if attributes are in correct format, false otherwise
	 */
	private boolean checkAttributesFormat(Instances dataSet, Set<Attribute> inputAttributes){
		
		StringBuilder nominalAttrRange = new StringBuilder();
		String rangeDelimiter = ",";
		for(Attribute attribute : inputAttributes){
			if(attribute.isNumeric() == false){
				if(attribute.isNominal()){
					nominalAttrRange.append((attribute.index() + 1) + rangeDelimiter);
				}
				else{
					// fail check if any other attribute type than nominal or numeric is used
					return false;
				}
			}
		}
		
		// convert any nominal attributes to binary
		if(nominalAttrRange.length() > 0){
			nominalAttrRange.deleteCharAt(nominalAttrRange.lastIndexOf(rangeDelimiter));
			try {
				nominalToBinaryFilter = new NominalToBinary();
				nominalToBinaryFilter.setAttributeIndices(nominalAttrRange.toString());
				nominalToBinaryFilter.setInputFormat(dataSet);
				dataSet = Filter.useFilter(dataSet, nominalToBinaryFilter);
			} catch (Exception exception) {
				nominalToBinaryFilter = null;
				if(getDebug()){
					debug("Failed to apply NominalToBinary filter to the input instances data. " +
							"Error message: " + exception.getMessage());
				}
				throw new WekaException("Failed to apply NominalToBinary filter to the input instances data.", exception);
			}
		}

		return true;
	}
	
    public MultiLabelOutput makePrediction(Instance instance) throws Exception {

		if(instance == null){
			throw new IllegalArgumentException("Input instance for prediction is null.");
		}
		int numAttributes = instance.numAttributes();
		if(numAttributes < model.getNetInputSize()){
			throw new IllegalArgumentException("Input instance do not have enough attributes " +
					"to be processed by the model. Instance is not consistent with the data the model was built for.");
		}

		Instance inputInstance = (instance instanceof SparseInstance) ?
				new SparseInstance(instance) : new Instance(instance);

		if(nominalToBinaryFilter != null){
			nominalToBinaryFilter.input(inputInstance);
			inputInstance = nominalToBinaryFilter.output();
			inputInstance.setDataset(null);
		}

		// if instance has more attributes than model input, we assume that true outputs 
		// are there, so we remove them
		List<Integer> labelIndices = new ArrayList<Integer>();
		boolean labelsAreThere = false;
		if(numAttributes > model.getNetInputSize()){
			for(int index : this.labelIndices)
				labelIndices.add(index);
			
			labelsAreThere = true;
		}

		if(normalizeAttributes){
			normalizer.normalize(inputInstance);
		}
		
		int inputDim = model.getNetInputSize();
		double[] inputPattern = new double[inputDim];
		int indexCounter = 0;
		for(int attrIndex = 0; attrIndex < numAttributes; attrIndex++){
			if(labelsAreThere && labelIndices.contains(attrIndex)){
					continue;
			}
			inputPattern[indexCounter] = inputInstance.value(attrIndex);
			indexCounter++;
		}
	
		double[] labelConfidences = model.feedForward(inputPattern);
		double threshold = thresholdF.computeThreshold(labelConfidences);
		boolean[] labelPredictions = new boolean[numLabels];
		Arrays.fill(labelPredictions, false);

		for(int labelIndex = 0; labelIndex < numLabels; labelIndex++){
			if(labelConfidences[labelIndex] > threshold){
				labelPredictions[labelIndex] = true;
			}
			// translate from bipolar output to binary
			labelConfidences[labelIndex] = (labelConfidences[labelIndex] + 1) / 2;
		}

        MultiLabelOutput mlo = new MultiLabelOutput(labelPredictions, labelConfidences);
        return mlo;
    }
    
    /**
     * Class for holding data pair for neural network. 
     * The data pair contains the input pattern and respected 
     * expected/ideal network output/response pattern for the input.
     */
    private class DataPair {

    	private final double[] input;
    	private final double[] output;
    	
    	/**
    	 * Creates a {@link DataPair} instance.
    	 * @param input the input pattern
    	 * @param output the ideal/expected output pattern for the input
    	 */
    	public DataPair(final double[] input, final double[] output){
    		if(input == null || output == null){
    			throw new IllegalArgumentException("Failed to create an instance. Either input or output pattern is null.");
    		}
    		this.input = Arrays.copyOf(input, input.length);
    		this.output = Arrays.copyOf(output, output.length);
    	}
    	
    	/**
    	 * Gets the input pattern.
    	 * @return the input pattern
    	 */
    	public double[] getInput(){
    		return input;
    	}
    	
    	/**
    	 * Gets the idel/expected output pattern.
    	 * @return the output pattern
    	 */
    	public double[] getOutput(){
    		return output;
    	}
    }
    
    /**
     * Normalizer performas a normalization of numeric attributes of the data set.
     * It is initialized based on given {@link MultiLabelInstances} data set and then can 
     * be used to normalize {@link Instance} instances which conform to the format of the 
     * data set the {@link Normalizer} was initialized from. 
     * 
     * @author Jozef Vilcek
     */
    public class Normalizer implements Serializable {

		/** Serial UID for serialization */
		private static final long serialVersionUID = -2575012048861337275L;
		private final double maxValue;
		private final double minValue;
		private Hashtable<Integer, double[]> attStats;
		
		/**
		 * Creates a new instance of {@link Normalizer} class for given data set.
		 * 
		 * @param mlData the {@link MultiLabelInstances} data set from which normalizer
		 * should be initialized.
		 * @param performNormalization indicates whether normalization of instances contained 
		 * in the data set used for initialization should be performed
		 * @param minValue the minimum value of the normalization range for numerical attributes
		 * @param maxValue the maximum value of the normalization range for numerical attributes
		 */
		public Normalizer(MultiLabelInstances mlData, boolean performNormalization, double minValue, double maxValue){
			if(mlData == null){
				throw new IllegalArgumentException("Parameter 'mlData' is null.");
			}
			if(maxValue <= minValue){
				throw new IllegalArgumentException(
						String.format("Parameters 'minValue=%f' and 'maxValue=%f' does not define valid range.",
								minValue, maxValue));
			}
			
			this.minValue = minValue;
			this.maxValue = maxValue;
			this.attStats = new Hashtable<Integer, double[]>();
			
			Initialize(mlData);
			
			if(performNormalization){
				Instances instances = mlData.getDataSet();
				int numInstance = instances.numInstances();
				for(int i=0; i<numInstance; i++){
					normalize(instances.instance(i));
				}
			}
		}
		
		/**
		 * Creates a new instance of {@link Normalizer} class for given data set.
		 * The normalizer will be initialized to perform normalization to the default
		 * range <-1,1>.  
		 * 
		 * @param mlData the {@link MultiLabelInstances} data set from which normalizer
		 * should be initialized.
		 * @param performNormalization indicates whether normalization of instances contained 
		 * in the data set used for initialization should be performed
		 */
		public Normalizer(MultiLabelInstances mlData, boolean performNormalization) {
			this(mlData, performNormalization, -1, 1);
		}
		
		/**
		 * Performs a normalization of numerical attributes on given instance.
		 * The instance must conform to format of instances data the {@link Normalizer} 
		 * was initialized with.
		 * @param instance the instance to be normalized
		 */
		public void normalize(Instance instance){
			Set<Integer> normScope = attStats.keySet(); 
			for(Integer attIndex : normScope){
				double[] stats = attStats.get(attIndex);
				double attMin = stats[0];
				double attMax = stats[1];
				double value = instance.value(attIndex);
				if(attMin == attMax){
					instance.setValue(attIndex, minValue);
				}else{
					instance.setValue(attIndex, (((value - stats[0]) / (stats[1] - stats[0])) * (maxValue - minValue)) + minValue);
				}
			}
		}
		
		private void Initialize(MultiLabelInstances mlData){
			Instances dataSet = mlData.getDataSet();
			int[] featureIndices = mlData.getFeatureIndices();
			for(int attIndex : featureIndices){
				Attribute feature = dataSet.attribute(attIndex);
				if(feature.isNumeric()){
					Stats stats = dataSet.attributeStats(attIndex).numericStats;
					attStats.put(attIndex, new double[]{stats.min, stats.max});
				}
			}
		}
    }
}
