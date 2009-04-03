package mulan.classifier.transformation;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import mulan.classifier.MultiLabelOutput;
import mulan.evaluation.PhiCoefficient;
import mulan.transformations.BinaryRelevanceTransformation;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class MultiLabelStacking extends TransformationBasedMultiLabelLearner implements Serializable {

	/**
	 * the BR transformed datasets of the original dataset
	 */
	protected Instances[] baseLevelData;
	/**
	 * the dataset produced by first level
	 */
	// protected Instances metaData;
	/**
	 * the BR transformed datasets of the meta dataset
	 */
	protected Instances[] metaLevelData;

	/**
	 * the enseble of BR classifiers of the original dataset
	 */
	protected Classifier[] baseLevelEnsemble;

	/**
	 * the enseble of BR classifiers of the meta dataset
	 */
	protected Classifier[] metaLevelEnsemble;
	/**
	 * the enseble of pruned BR classifiers of the meta dataset
	 */
	protected FilteredClassifier[] metaLevelFilteredEnsemble;
	/**
	 * the number of folds used in the first level
	 */
	int numFolds;

	/**
	 * a table holding the predictions of the first level classifiers for each
	 * class-label of every instance
	 */
	double[][] baseLevelPredictions;

	protected BinaryRelevanceTransformation transformation;
	
	/**
	 * whether to normalize baseLevelPredictions or not.
	 */
	boolean normalize;

	public boolean isNormalize() {
		return normalize;
	}

	public void setNormalize(boolean normalize) {
		this.normalize = normalize;
	}
	
	boolean includeAttrs;

	public boolean isIncludeAttrs() {
		return includeAttrs;
	}

	public void setIncludeAttrs(boolean includeAttrs) {
		this.includeAttrs = includeAttrs;
	}

	PhiCoefficient phi;

	double phival;
	
	double maxProb[];
	double minProb[];

	int[] numUncorrelated;

    public int[] getNumUncorrelated() {
        return numUncorrelated;
    }

	public double getPhival() {
		return phival;
	}

	public void setPhival(double phival) {
		this.phival = phival;
	}

	public MultiLabelStacking(Classifier baseClassifier,
			Classifier metaClassifier, int numFolds, int numLabels)
			throws Exception {
		super(baseClassifier, numLabels);
		transformation = new BinaryRelevanceTransformation(numLabels);
		baseLevelData = new Instances[numLabels];
		metaLevelData = new Instances[numLabels];
		debug("BR: making classifier copies");
		baseLevelEnsemble = Classifier.makeCopies(baseClassifier, numLabels);
		metaLevelEnsemble = Classifier.makeCopies(metaClassifier, numLabels);
		metaLevelFilteredEnsemble = new FilteredClassifier[numLabels];
		this.numFolds = numFolds;
		phival = 0;
		normalize = false;
		includeAttrs = false;
	}

	public void buildBaseLevel(Instances train) throws Exception {
		if(normalize){
			maxProb = new double[numLabels];
			minProb = new double[numLabels];
			Arrays.fill(minProb, 1);
		}
		// initialize the table holding the predictions of the first level
		// classifiers for each label for every instance of the training set
		baseLevelPredictions = new double[train.numInstances()][numLabels];
		// attach indexes in order to keep track of the original positions
		Instances trainData = new Instances(attachIndexes(train));
		
		for (int labelIndex = 0; labelIndex < numLabels; labelIndex++) {
			debug("Label: " + labelIndex);
			// transform the dataset according to the BR method
			baseLevelData[labelIndex] = transformation.transformInstances(trainData, labelIndex);
			// prepare the transformed dataset for stratified x-fold cv	
			Random random = new Random(1);
			baseLevelData[labelIndex].randomize(random);
			baseLevelData[labelIndex].stratify(numFolds);
			debug("Creating meta-data");
            for (int j = 0; j < numFolds; j++) {
                debug("Label=" + labelIndex + ", Fold=" + j);
				Instances subtrain = baseLevelData[labelIndex].trainCV(
						numFolds, j, random);
				// create a filtered meta classifier, used to ignore
				// the index attribute in the build process
				// perform stratified x-fold cv and get predictions for class l for
				// every instance
				FilteredClassifier fil = new FilteredClassifier();
				fil.setClassifier(baseLevelEnsemble[labelIndex]);
				Remove remove = new Remove();
				remove.setAttributeIndices("first");
				remove.setInputFormat(subtrain);
				fil.setFilter(remove);
				fil.buildClassifier(subtrain);

				// Classify test instance
				Instances subtest = baseLevelData[labelIndex].testCV(numFolds,
						j);
				for (int i = 0; i < subtest.numInstances(); i++) {
					double distribution[] = new double[2];
					distribution = fil.distributionForInstance(subtest
							.instance(i));
					// Ensure correct predictions both for class values {0,1}
					// and {1,0}
					Attribute classAttribute = baseLevelData[labelIndex]
							.classAttribute();
					baseLevelPredictions[(int) subtest.instance(i).value(0)][labelIndex] = distribution[classAttribute
							.indexOfValue("1")];
					if(normalize){
						if(distribution[classAttribute.indexOfValue("1")]> maxProb[labelIndex]){
							maxProb[labelIndex] = distribution[classAttribute.indexOfValue("1")];
						}
						if(distribution[classAttribute.indexOfValue("1")]< minProb[labelIndex]){
							minProb[labelIndex] = distribution[classAttribute.indexOfValue("1")];
						}
						
					}
				}
			}
    		// now we can detach the indexes from the first level datasets
			baseLevelData[labelIndex] = detachIndexes(baseLevelData[labelIndex]);

            debug("Building base classifier on full data");
    		// build base classifier on the full training data
			baseLevelEnsemble[labelIndex].buildClassifier(baseLevelData[labelIndex]);
			baseLevelData[labelIndex].delete();
        }

		if(normalize){
			normalizePredictions();
		}
		
		//calculate the PhiCoefficient, used in the meta-level
		phi = new PhiCoefficient();
		phi.calculatePhi(train, numLabels);		
	}
	
	private void normalizePredictions(){
		for(int i=0;i<baseLevelPredictions.length;i++){
			for(int j=0;j<numLabels;j++){
				baseLevelPredictions[i][j]=baseLevelPredictions[i][j]-minProb[j]/maxProb[j]-minProb[j];
			}
		}
	}

    public void saveObject(String filename) {
        try {
            ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename));
            out.writeObject(this);
        } catch (IOException ex) {
            Logger.getLogger(MultiLabelStacking.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

	public void buildMetaLevel(Instances train,double phival) throws Exception {
		this.phival = phival;

		debug("Building the ensemle of the meta level classifers");
		// After that we can build the metalevel ensemble of classifiers
		// we apply a filtered classifier to prune uncorrelated labels
		numUncorrelated = new int[numLabels];
        for (int i = 0; i < numLabels; i++) {
            // creating meta-level data
        	if(includeAttrs){
        		metaLevelData[i] = metaFormat2(baseLevelData[i]);
        	}
        	else{
        		metaLevelData[i] = metaFormat(baseLevelData[i]);
        	}
        	
			for (int l = 0; l < train.numInstances(); l++) {
				// add the meta instances
				Instance metaInstance;
				if(includeAttrs){
					metaInstance = metaInstance2(train.instance(l),i,l);
				}
				else{
					metaInstance = metaInstance(train.instance(l),i,l);
				}
				metaLevelData[i].add(metaInstance);
			}


			metaLevelFilteredEnsemble[i] = new FilteredClassifier();
			metaLevelFilteredEnsemble[i].setClassifier(metaLevelEnsemble[i]);
			Remove remove = new Remove();
			int[] attributes = phi.uncorrelatedIndices(i, phival);
			numUncorrelated[i] = attributes.length;
			remove.setAttributeIndicesArray(attributes);
			remove.setInputFormat(metaLevelData[i]);
			metaLevelFilteredEnsemble[i].setFilter(remove);
            debug("Building classifier for meta training set" + i);
			metaLevelFilteredEnsemble[i].buildClassifier(metaLevelData[i]);
            metaLevelData[i].delete();
		}
	}

	@Override
	public void build(Instances train) throws Exception {
		buildBaseLevel(train);
		buildMetaLevel(train,phival);
	}

	/**
	 * Makes the format for the meta-level data.
	 * The predictions of the base level classifiers + the class attribute
	 * 
	 * @param instances the base-level format
	 * @return the format for the meta data
	 * @throws Exception
	 *             if the format generation fails
	 */
	protected Instances metaFormat(Instances instances) throws Exception {

		FastVector attributes = new FastVector();
		Instances metaFormat;

		for (int k = 0; k < baseLevelEnsemble.length; k++) {
			String name = baseLevelData[k].classAttribute().toString();
			attributes.addElement(new Attribute(name));
		}
		attributes.addElement(instances.classAttribute().copy());
		metaFormat = new Instances("Meta format", attributes, 0);
		metaFormat.setClassIndex(metaFormat.numAttributes() - 1);
		return metaFormat;
	}
	
	/**
	 * Makes the format for the meta-level data.
	 * The predictions of the base level classifiers + the class attribute
	 * 
	 * @param instances the base-level format
	 * @return the format for the meta data
	 * @throws Exception
	 *             if the format generation fails
	 */
	protected Instances metaFormat2(Instances instances) throws Exception {

		FastVector attributes = new FastVector();
		Instances metaFormat;

		for (int i = 0; i < baseLevelEnsemble.length; i++) {
			String name = baseLevelData[i].classAttribute().toString(); //.name
			attributes.addElement(new Attribute(name)); //constructor for a numeric attribute
		}
		
		for (int i = 0; i < instances.numAttributes()-1; i++) {
			attributes.addElement(instances.attribute(i));
		}
		
		attributes.addElement(instances.classAttribute());
		metaFormat = new Instances("Meta format", attributes, 0);
		metaFormat.setClassIndex(metaFormat.numAttributes() - 1);
		return metaFormat;
	}
	
	/**
	 * 
	 * @param instance the base-level instance
	 * @param labelIndex
	 * @param index
	 * @return
	 * @throws Exception
	 */
	protected Instance metaInstance(Instance instance, int labelIndex, int index) throws Exception {

		double[] values = new double[metaLevelData[labelIndex].numAttributes()];
		int k = 0;
		for (k = 0; k < numLabels; k++) {
			values[k] = baseLevelPredictions[index][k];
		}
		values[k] = instance.value(
				instance.numAttributes() - numLabels + labelIndex);
		Instance metaInstance = new Instance(1, values);
		metaInstance.setDataset(metaLevelData[labelIndex]);
		
		return metaInstance;
	}
	
	/**
	 * 
	 * @param instance the base-level instance
	 * @param labelIndex
	 * @param index
	 * @return
	 * @throws Exception
	 */
	protected Instance metaInstance2(Instance instance, int labelIndex, int index) throws Exception {

		double[] values = new double[metaLevelData[labelIndex].numAttributes()];
		int k = 0;
		for (k = 0; k < numLabels; k++) {
			values[k] = baseLevelPredictions[index][k];
		}
		for (k = numLabels ; k < instance.numAttributes(); k++){
			values[k] = instance.value(k-numLabels);
		}
		values[k] = instance.value(
				instance.numAttributes() - numLabels + labelIndex);
		Instance metaInstance = new Instance(1, values);
		metaInstance.setDataset(metaLevelData[labelIndex]);
		
		return metaInstance;
	}

	/**
	 * Attaches an index attribute at the beginning of each instance
	 * 
	 * @param original
	 * @return
	 */
	protected Instances attachIndexes(Instances original) {

		FastVector attributes = new FastVector(original.numAttributes() + 1);

		for (int i = 0; i < original.numAttributes(); i++) {
			attributes.addElement(original.attribute(i));
		}
		// Add attribute for holding the index at the end.
		attributes.insertElementAt(new Attribute("Index"), 0);
		Instances transformed = new Instances("Meta format", attributes, 0);
		for (int i = 0; i < original.numInstances(); i++) {
			Instance newInstance;
			newInstance = (Instance) original.instance(i).copy();
			newInstance.setDataset(null);
			newInstance.insertAttributeAt(0);
			newInstance.setValue(0, i);

			transformed.add(newInstance);
		}

		transformed.setClassIndex(transformed.numAttributes() - 1);
		return transformed;
	}

	/**
	 * Detaches the index attribute from the beginning of each instance
	 * 
	 * @param original
	 * @return
	 * @throws Exception
	 */
	protected Instances detachIndexes(Instances original) throws Exception {

		Remove remove = new Remove();
		remove.setAttributeIndices("first");
		remove.setInputFormat(original);
		// remove.setInvertSelection(true);
		Instances result = Filter.useFilter(original, remove);
		// result.setClassIndex(result.numAttributes() - 1);
		return result;

	}

	@Override
	public MultiLabelOutput makePrediction(Instance instance) throws Exception {
		boolean[] bipartition = new boolean[numLabels];
		// the confidences given as final output
		double[] metaconfidences = new double[numLabels];
		// the confidences produced by the first level ensemble of classfiers
		double[] confidences = new double[numLabels];
		// the meta instance consisting of the above confidences and the actual
		// labelset
		// numlabels + numlabels attributes
		Instance metaInstance;

		// getting the confidences for each label
		for (int labelIndex = 0; labelIndex < numLabels; labelIndex++) {
			Instance newInstance = transformation.transformInstance(instance,
					labelIndex);
			newInstance.setDataset(baseLevelData[labelIndex]);

			double distribution[] = new double[2];
			try {
				distribution = baseLevelEnsemble[labelIndex]
						.distributionForInstance(newInstance);
			} catch (Exception e) {
				System.out.println(e);
				return null;
			}

			// Ensure correct predictions both for class values {0,1} and {1,0}
			Attribute classAttribute = baseLevelData[labelIndex]
					.classAttribute();

			// The confidence of the label being equal to 1
			confidences[labelIndex] = distribution[classAttribute
					.indexOfValue("1")];
		}

		/* creation of the meta-instance with the appropriate values */
		double[] classes = new double[numLabels];
		double[] values;
		if(includeAttrs){
			double[] attributes = new double[instance.numAttributes()-numLabels];
			for(int i=0;i<instance.numAttributes()-numLabels;i++){
				attributes[i] = instance.value(i);
			}
			// Concatenation of the three tables (confidences, attributes and classes)
			values = new double[confidences.length + attributes.length + classes.length];
			System.arraycopy(confidences, 0, values, 0, confidences.length);
			System.arraycopy(attributes, 0, values, confidences.length, attributes.length);
			System
					.arraycopy(classes, 0, values, confidences.length+ attributes.length,
							classes.length);
		}
		else{
			// Concatenation of the two tables (confidences and classes)
			values = new double[confidences.length + classes.length];
			System.arraycopy(confidences, 0, values, 0, confidences.length);
			System
				.arraycopy(classes, 0, values, confidences.length,
						classes.length);
		}
		metaInstance = new Instance(1, values);

		/* application of the meta-level ensemble to the metaInstance */
		for (int labelIndex = 0; labelIndex < numLabels; labelIndex++) {
			Instance newmetaInstance = transformation.transformInstance(
					metaInstance, labelIndex);
			newmetaInstance.setDataset(metaLevelData[labelIndex]);

			double distribution[] = new double[2];
			try {
				distribution = metaLevelFilteredEnsemble[labelIndex]
						.distributionForInstance(newmetaInstance);
			} catch (Exception e) {
				System.out.println(e);
				return null;
			}
			int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

			// Ensure correct predictions both for class values {0,1} and {1,0}
			Attribute classAttribute = metaLevelData[labelIndex]
					.classAttribute();
			bipartition[labelIndex] = (classAttribute.value(maxIndex)
					.equals("1")) ? true : false;

			// The confidence of the label being equal to 1
			metaconfidences[labelIndex] = distribution[classAttribute
					.indexOfValue("1")];
		}

		MultiLabelOutput mlo = new MultiLabelOutput(bipartition,
				metaconfidences);
		return mlo;
	}
	
}
