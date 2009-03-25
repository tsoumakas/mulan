package mulan.classifier.transformation;

import java.util.Arrays;
import java.util.Random;

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

public class MultiLabelStacking extends TransformationBasedMultiLabelLearner {

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

	PhiCoefficient phi;

	double phival;
	
	double maxProb[];
	double minProb[];

	int numUncorrelated;

    public int getNumUncorrelated() {
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
		
		debug("Building the ensemle of the base level classifers");
		for (int labelIndex = 0; labelIndex < numLabels; labelIndex++) {
			debug("BR: Building classifier for base training set " + labelIndex);
			// transform the dataset according to the BR method
			baseLevelData[labelIndex] = transformation.transformInstances(trainData, labelIndex);
			// prepare the transformed dataset for stratified x-fold cv	
			Random random = new Random(1);
			baseLevelData[labelIndex].randomize(random);
			baseLevelData[labelIndex].stratify(numFolds);

			for (int j = 0; j < numFolds; j++) {
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
		}

		if(normalize){
			normalizePredictions();
		}
		
		// now we can detach the indexes from the first level datasets
		for (int i = 0; i < numLabels; i++) {
			baseLevelData[i] = detachIndexes(baseLevelData[i]);
		}
		// build all the base classifiers on the full training data
		for (int i = 0; i < numLabels; i++) {
			baseLevelEnsemble[i].buildClassifier(baseLevelData[i]);
			baseLevelData[i].delete();
		}

		//calculate the PhiCoefficient, used in the meta-level
		phi = new PhiCoefficient();
		phi.calculatePhi(train, numLabels);
		
		//build the l meta-level datasets
		for (int i = 0; i < numLabels; i++) {
			metaLevelData[i] = metaFormat(baseLevelData[i]);
			for (int l = 0; l < train.numInstances(); l++) { // add the meta
				// instances
				double[] values = new double[metaLevelData[i].numAttributes()];
				int k = 0;
				for (k = 0; k < numLabels; k++) {
					values[k] = baseLevelPredictions[l][k];
				}

				values[k] = train.instance(l).value(
						train.numAttributes() - numLabels + i);
				Instance metaInstance = new Instance(1, values);
				metaInstance.setDataset(metaLevelData[i]);

				metaLevelData[i].add(metaInstance);
			}
		}

	}
	
	private void normalizePredictions(){
		for(int i=0;i<baseLevelPredictions.length;i++){
			for(int j=0;j<numLabels;j++){
				baseLevelPredictions[i][j]=baseLevelPredictions[i][j]-minProb[j]/maxProb[j]-minProb[j];
			}
		}
	}

	public void buildMetaLevel(Instances train,double phival) throws Exception {
		this.phival = phival;
		debug("Building the ensemle of the meta level classifers");
		// After that we can build the metalevel ensemble of classifiers
		// we apply a filtered classifier to prune uncorrelated labels
		for (int i = 0; i < numLabels; i++) {
			debug("BR: Building classifier for meta training set" + i);
			metaLevelFilteredEnsemble[i] = new FilteredClassifier();
			metaLevelFilteredEnsemble[i].setClassifier(metaLevelEnsemble[i]);
			Remove remove = new Remove();
			int[] attributes = phi.uncorrelatedIndices(i, phival);
			numUncorrelated = attributes.length;
			remove.setAttributeIndicesArray(attributes);
			remove.setInputFormat(metaLevelData[i]);
			metaLevelFilteredEnsemble[i].setFilter(remove);
			metaLevelFilteredEnsemble[i].buildClassifier(metaLevelData[i]);
		}
	}

	@Override
	public void build(Instances train) throws Exception {
		buildBaseLevel(train);
		buildMetaLevel(train,phival);
		/*// initialize the table holding the predictions of the first level
		// classifiers for each label for every instance of the training set
		baseLevelPredictions = new double[train.numInstances()][numLabels];
		for (int xxx1 = 0; xxx1 < train.numInstances(); xxx1++)
			Arrays.fill(baseLevelPredictions[xxx1], -1.0);

		for (int labelIndex = 0; labelIndex < numLabels; labelIndex++) {
			// for each label
			debug("BR: transforming training set for label " + labelIndex);
			// transform the dataset according to the BR method
			// and attach indexes in order to keep track of the original
			// positions
			baseLevelData[labelIndex] = attachIndexes(transformation
					.transformInstances(train, labelIndex));
			// prepare the transformed dataset for stratified x-fold cv
			Random random = new Random(1);
			baseLevelData[labelIndex].randomize(random);
			// if (firstlevelData[labelIndex].classAttribute().isNominal()) {
			baseLevelData[labelIndex].stratify(numFolds);
			// }
			// create a filtered meta classifier, used to ignore
			// the index attribute in the build process
			// perform stratified x-fold cv and get predictions for class l for
			// every instance
			for (int j = 0; j < numFolds; j++) {

				Instances subtrain = baseLevelData[labelIndex].trainCV(
						numFolds, j, random);
				// Build base classifier (one classifier in our case)

				FilteredClassifier fil = new FilteredClassifier();
				fil.setClassifier(baseLevelEnsemble[j]);
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
					// System.out.println(subtest.instance(i).value(0));
					baseLevelPredictions[(int) subtest.instance(i).value(0)][labelIndex] = distribution[classAttribute
							.indexOfValue("1")];
				}
			}
		}
		// System.out.println(Arrays.deepToString(firstlevelpredictions));

		// now we can detach the indexes from the first level datasets
		for (int i = 0; i < numLabels; i++) {
			baseLevelData[i] = detachIndexes(baseLevelData[i]);
		}
		// now we are ready to build the l meta-level datasets

		// TO DO: here we should apply the pruning of uncorrelated labels.
		// We can also leave the meta-level datasets intact
		// and use a filtered classifier to ignore the
		// uncorrelated attributes
		for (int i = 0; i < numLabels; i++) {
			metaLevelData[i] = metaFormat(baseLevelData[i]);
			for (int l = 0; l < train.numInstances(); l++) { // add the meta
				// instances
				double[] values = new double[metaLevelData[i].numAttributes()];
				int k = 0;
				for (k = 0; k < numLabels; k++) {
					values[k] = baseLevelPredictions[l][k];
				}

				values[k] = train.instance(l).value(
						train.numAttributes() - numLabels + i);
				// values[k] = firstlevelData[i].instance(l).classValue();
				Instance metaInstance = new Instance(1, values);
				metaInstance.setDataset(metaLevelData[i]);

				metaLevelData[i].add(metaInstance);
			}
		}

		// After that we can build the metalevel ensemble of classifiers
		// TO DO: we can apply a filtered classifier to prune uncorrelated
		// labels here.
		if (prune) {
			PhiCoefficient phi = new PhiCoefficient();
			double[][] correlations = phi.calculatePhi(train, numLabels);
			for (int i = 0; i < numLabels; i++) {
				metaLevelFilteredEnsemble[i] = new FilteredClassifier();
				metaLevelFilteredEnsemble[i]
						.setClassifier(metaLevelEnsemble[i]);
				Remove remove = new Remove();
				int[] attributes = phi.uncorrelatedIndices(i, 0.3);
				remove.setAttributeIndicesArray(attributes);
				remove.setInputFormat(metaLevelData[i]);
				metaLevelFilteredEnsemble[i].setFilter(remove);
			}
		}

		// rebuilt all the base classifiers on the full training data
		for (int i = 0; i < numLabels; i++) {
			baseLevelEnsemble[i].buildClassifier(baseLevelData[i]);
			if (!prune) {
				metaLevelEnsemble[i].buildClassifier(metaLevelData[i]);
			} else {
				metaLevelFilteredEnsemble[i].buildClassifier(metaLevelData[i]);
			}

		}*/
	}

	/**
	 * Makes the format for the level-1 data.
	 * 
	 * @param instances
	 *            the level-0 format
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
			// attributes.addElement(firstlevelData[k].classAttribute());
		}
		attributes.addElement(instances.classAttribute().copy());
		metaFormat = new Instances("Meta format", attributes, 0);
		metaFormat.setClassIndex(metaFormat.numAttributes() - 1);
		return metaFormat;
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
		// Concatenation of the two tables (confidences and classes)
		double[] values = new double[confidences.length + classes.length];
		System.arraycopy(confidences, 0, values, 0, confidences.length);
		System
				.arraycopy(classes, 0, values, confidences.length,
						classes.length);

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
