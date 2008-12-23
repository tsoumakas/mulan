package mulan.classifier.neural;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;

import mulan.classifier.MultiLabelClassifierBase;
import mulan.classifier.Prediction;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;



/**
 * 
 * The network is created when classifier is being build. The input and output layer is determined from input training data.
 * 
 * @author Jozef Vilcek
 */
public class BPMLL extends MultiLabelClassifierBase {

	private static final long serialVersionUID = 2153814250172139021L;
	private static final double NET_BIAS = 1;
	private static final double ERROR_SMALL_CHANGE = 0.000001;
	
	// filter used to convert nominal input attributes into binary-numeric 
	NominalToBinary nominalToBinaryFilter;
	
	// algorithm parameters 
	private int epochs = 100;
	private double weightsDecayCost = 0.00001;
	private double learningRate = 0.05;
	private int[] hiddenLayersTopology;
	
	// members related to normalization or attributes
	private boolean normalizeAttributes = true;
	private double[] attRanges;
	private double[] attBases;

	private NeuralNet model;
	private ThresholdFunction thresholdF;
	
	/**
	 * Creates a {@link BPMLL} instance.
	 * 
	 * @param numLabels number of labels of the classifier
	 */
	public BPMLL(int numLabels) {
		super(numLabels);
	}
	
	/**
	 * Sets the topology of hidden layers for neural network. 
	 * The length of passed array defines number of hidden layers. 
	 * The value at particular index of array defines number of neurons in that layer.
	 * If null is specified, no hidden layers will be created.
	 * <br/>
	 * The network is created when classifier is being built.
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
	 * be normalized prior to building the classifier. Normalization is performed 
	 * on numeric attributes to the range {-1,1}).<br/> 
	 * When making prediction, attributes of passed input instance are also 
	 * normalized prior to making prediction.<br/>
	 * Default is false (no normalization takes place).
	 * 
	 * @param epochs the number of training epochs
	 * @throws IllegalArgumentException if passed value is invalid
	 */
	public void setNormalizeAttributes(boolean normalize){
		normalizeAttributes = normalize;
	}

	@Override
	public void buildClassifier(final Instances instances) throws Exception {
		
		if(instances == null){
			throw new IllegalArgumentException("Instances must not be null.");
		}
		// delete filter if available from previous build, a new one will be created if necessary
		nominalToBinaryFilter = null;
		
		Instances trainInstances = new Instances(instances);
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
	
	protected Prediction makePrediction(final Instance instance) throws Exception {

		if(instance == null){
			throw new IllegalArgumentException("Input instance for prediction is null.");
		}
		if(instance.numAttributes() < model.getNetInputSize()){
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
		
		// remove label attributes from the end of instance if they are available there
		if(inputInstance.numAttributes() > model.getNetInputSize()){
			int diff = inputInstance.numAttributes() - model.getNetInputSize();
			for(int i = 0; i < diff; i++)
				inputInstance.deleteAttributeAt(model.getNetInputSize());
		}
		
		if(normalizeAttributes){
			int numAttributes = inputInstance.numAttributes();
			for (int attIndex = 0; attIndex < numAttributes; attIndex++) {
				normalizeAttribute(inputInstance, attIndex, attRanges[attIndex], attBases[attIndex]);
			}
		}
		
		double[] inputPattern = Arrays.copyOfRange(inputInstance.toDoubleArray(), 0, inputInstance.numAttributes());
		double[] labelConfidences = model.feedForward(inputPattern);
		double threshold = thresholdF.computeThreshold(labelConfidences);
		double[] labelPredictions = new double[numLabels];
		
		for(int labelIndex = 0; labelIndex < numLabels; labelIndex++){
			if(labelConfidences[labelIndex] > threshold){
				labelPredictions[labelIndex] = 1;
			}
			// translate from bipolar output to binary
			labelConfidences[labelIndex] = (labelConfidences[labelIndex] + 1) / 2;
		}
		
		return new Prediction(labelPredictions, labelConfidences);
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
	private List<DataPair> prepareData(Instances data){
		
		if(!checkLabelAttributesFormat(data)){
			throw new IllegalArgumentException("Label attributes are not in correct format. " +
					"Label attributes must be nominal with binary values.");
		}
		if(!checkAttributesFormat(data)){
			throw new IllegalArgumentException("Attributes are not in correct format. " +
					"Input attributes (all but the label attributes) must be numeric.");
		}
				
		if(normalizeAttributes){
			normalizeAttributes(data);
		}
		else{
			attBases = null;
			attRanges = null;
		}
		int numInstances = data.numInstances();
		int inputDim = data.numAttributes() - numLabels;
		List<DataPair> dataPairs = new ArrayList<DataPair>();
		for(int index = 0; index < numInstances; index++){
			double[] instance = data.instance(index).toDoubleArray();
			double[] input = Arrays.copyOfRange(instance, 0, inputDim);
			double[] output = Arrays.copyOfRange(instance, inputDim, instance.length);
			for(int i = 0; i < output.length; i++){
				if(output[i] == 0){
					output[i] = -1;
				}
			}
			dataPairs.add(new DataPair(input, output));
		}
		
		return dataPairs;
	}
	
	/**
	 * Checks {@link Instances} data if label attributes are nominal and have binary values.
	 * 
	 * @param data instances data to be checked
	 * @return true if label attributes are in correct format, false otherwise
	 */
	@SuppressWarnings("unchecked")
	private boolean checkLabelAttributesFormat(Instances data){
		
		int numAttributes = data.numAttributes();
		for(int labelIndex = numAttributes - numLabels; labelIndex < numAttributes; labelIndex++){
			Attribute labelAttr = data.attribute(labelIndex);
			if(labelAttr.isNominal() != true){
				return false;
			}
			List<String> allowedValues = new ArrayList<String>();
			allowedValues.add("0");
			allowedValues.add("1");
			Enumeration labelValues = labelAttr.enumerateValues();
			while (labelValues.hasMoreElements()) {
				String value = (String)labelValues.nextElement();
				if(allowedValues.contains(value)){
					allowedValues.remove(value);
				}
			}
			if(allowedValues.size() != 0){
				return false;
			}
		}
		
		return true;
	}
	
	/**
	 * Checks {@link Instances} data if attributes (all but the label attributes) 
	 * are numeric or nominal. Nominal attributes are transformed to binary by use of
	 * {@link NominalToBinary} filter.
	 * 
	 * @param data instances data to be checked
	 * @return true if attributes are in correct format, false otherwise
	 */
	private boolean checkAttributesFormat(Instances data){
		
		StringBuilder nominalAttrRange = new StringBuilder();
		String rangeDelimiter = ",";
		int numAttributes = data.numAttributes();
		for(int attrIndex = 0; attrIndex < numAttributes - numLabels; attrIndex++){
			Attribute attribute = data.attribute(attrIndex);
			if(attribute.isNumeric() == false){
				if(attribute.isNominal()){
					nominalAttrRange.append((attrIndex + 1) + rangeDelimiter);
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
				nominalToBinaryFilter.setInputFormat(data);
				data = Filter.useFilter(data, nominalToBinaryFilter);
			} catch (Exception exception) {
				nominalToBinaryFilter = null;
				if(getDebug()){
					debug("Failed to apply NominalToBinary filter to the input instances data. " +
							"Error message: " + exception.getMessage());
				}
				throw new RuntimeException("Failed to apply NominalToBinary filter to the input instances data.", exception);
			}
		}

		return true;
	}
	
	/**
	 * Performs normalization of all but label attributes into the range <-1,1>
	 * 
	 * @param data instances data of which attributes should be normalized
	 */
	private void normalizeAttributes(Instances data){
		
		int numInstances = data.numInstances();
		int numAttributes = data.numAttributes();
		attRanges = new double[numAttributes - numLabels];
		attBases = new double[numAttributes - numLabels];
		for(int attIndex = 0; attIndex < numAttributes - numLabels; attIndex++){
			// get min max values of attribute
			double min = Double.POSITIVE_INFINITY;
			double max = Double.NEGATIVE_INFINITY;
			for (int i = 0; i < numInstances; i++) {
				Instance instance = data.instance(i);
				if (!instance.isMissing(attIndex)) {
					double value = instance.value(attIndex);
					if (value < min) {
						min = value;
					}
					if (value > max) {
						max = value;
					}
				}
			}
			// normalize values of the attribute to <-1,1>
			attRanges[attIndex] = (max - min) / 2;
			attBases[attIndex] = (max + min) / 2;
			for (int i = 0; i < numInstances; i++) {
				Instance instance = data.instance(i);
				normalizeAttribute(instance, attIndex, attRanges[attIndex], attBases[attIndex]);
			}
		}
	}
	
	private void normalizeAttribute(Instance instance, int attIndex, double attRange, double attBase){
		
		if (attRange != 0) {
			double value = instance.value(attIndex);
			value = (value - attBase) / attRange;
			instance.setValue(attIndex, value);
		} 
		else {
			double value = instance.value(attIndex);
			value = value - attBase;
			instance.setValue(attIndex, value);
		}
	}
}
