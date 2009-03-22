package mulan.classifier.transformation;

import java.util.Arrays;
import java.util.Random;

import mulan.classifier.MultiLabelOutput;
import mulan.transformations.BinaryRelevanceTransformation;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.meta.ClassificationViaRegression;
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
	protected Instances[] firstlevelData;
	/**
	 * the dataset produced by first level
	 */
	// protected Instances metaData;
	/**
	 * the BR transformed datasets of the meta dataset
	 */
	protected Instances[] metalevelData;

	/**
	 * the enseble of BR classifiers of the original dataset
	 */
	protected Classifier[] firstlevelEnsemble;

	/**
	 * the enseble of BR classifiers of the meta dataset
	 */
	protected Classifier[] metalevelEnsemble;
	/**
	 * the number of folds used in the first level
	 */
	int numFolds;

	/**
	 * a table holding the predictions of the first level classifiers for each
	 * class-label of every instance
	 */
	double[][] firstlevelpredictions;

	protected BinaryRelevanceTransformation transformation;

	public MultiLabelStacking(Classifier classifier, int numFolds, int numLabels)
			throws Exception {
		super(classifier, numLabels);
		transformation = new BinaryRelevanceTransformation(numLabels);
		firstlevelData = new Instances[numLabels];
		metalevelData = new Instances[numLabels];
		debug("BR: making classifier copies");
		firstlevelEnsemble = Classifier.makeCopies(classifier, numLabels);
		metalevelEnsemble = new Classifier[numLabels];
		for (int i = 0; i < numLabels; i++) {
			ClassificationViaRegression cvr = new ClassificationViaRegression();
			LinearRegression lr = new LinearRegression();
			cvr.setClassifier(lr);
			metalevelEnsemble[i] = cvr;
		}
		// metalevelEnsemble = Classifier.makeCopies(classifier, numLabels);
		this.numFolds = numFolds;
	}

	@Override
	public void build(Instances train) throws Exception {
		// initialize the table holding the predictions of the first level
		// classifiers for each label for every instance of the training set
		firstlevelpredictions = new double[train.numInstances()][numLabels];
		for (int xxx1 = 0; xxx1 < train.numInstances(); xxx1++)
			Arrays.fill(firstlevelpredictions[xxx1], -1.0);

		for (int labelIndex = 0; labelIndex < numLabels; labelIndex++) {
			// for each label
			debug("BR: transforming training set for label " + labelIndex);
			// transform the dataset according to the BR method
			// and attach indexes in order to keep track of the original
			// positions
			firstlevelData[labelIndex] = attachIndexes(transformation
					.transformInstances(train, labelIndex));
			// prepare the transformed dataset for stratified x-fold cv
			Random random = new Random(1);
			firstlevelData[labelIndex].randomize(random);
			// if (firstlevelData[labelIndex].classAttribute().isNominal()) {
			firstlevelData[labelIndex].stratify(numFolds);
			// }
			// create a filtered meta classifier, used to ignore
			// the index attribute in the build process
			// perform stratified x-fold cv and get predictions for class l for
			// every instance
			for (int j = 0; j < numFolds; j++) {

				Instances subtrain = firstlevelData[labelIndex].trainCV(
						numFolds, j, random);
				// Build base classifier (one classifier in our case)

				FilteredClassifier fil = new FilteredClassifier();
				fil.setClassifier(firstlevelEnsemble[j]);
				Remove remove = new Remove();
				remove.setAttributeIndices("first");
				remove.setInputFormat(subtrain);
				fil.setFilter(remove);

				fil.buildClassifier(subtrain);

				// Classify test instance
				Instances subtest = firstlevelData[labelIndex].testCV(numFolds,
						j);
				for (int i = 0; i < subtest.numInstances(); i++) {
					double distribution[] = new double[2];
					distribution = fil.distributionForInstance(subtest
							.instance(i));
					// Ensure correct predictions both for class values {0,1}
					// and {1,0}
					Attribute classAttribute = firstlevelData[labelIndex]
							.classAttribute();
					// System.out.println(subtest.instance(i).value(0));
					firstlevelpredictions[(int) subtest.instance(i).value(0)][labelIndex] = distribution[classAttribute
							.indexOfValue("1")];
				}
			}
		}
		// System.out.println(Arrays.deepToString(firstlevelpredictions));

		// now we can detach the indexes from the first level datasets
		for (int i = 0; i < numLabels; i++) {
			firstlevelData[i] = detachIndexes(firstlevelData[i]);
		}
		// now we are ready to build the l meta-level datasets

		// TO DO: here we should apply the pruning of uncorrelated labels.
		// We can also leave the meta-level datasets intact
		// and use a filtered classifier to ignore the
		// uncorrelated attributes
		for (int i = 0; i < numLabels; i++) {
			metalevelData[i] = metaFormat(firstlevelData[i]);
			for (int l = 0; l < train.numInstances(); l++) { // add the meta
				// instances
				double[] values = new double[metalevelData[i].numAttributes()];
				int k = 0;
				for (k = 0; k < numLabels; k++) {
					values[k] = firstlevelpredictions[l][k];
				}

				values[k] = train.instance(l).value(
						train.numAttributes() - numLabels + i);
				// values[k] = firstlevelData[i].instance(l).classValue();
				Instance metaInstance = new Instance(1, values);
				metaInstance.setDataset(metalevelData[i]);

				metalevelData[i].add(metaInstance);
			}
		}

		// After that we can build the metalevel ensemble of classifiers
		// and rebuilt all the base classifiers on the full training data
		// TO DO: we can apply a filtered classifier to prune uncorrelated
		// labels here.

		for (int i = 0; i < numLabels; i++) {
			metalevelEnsemble[i].buildClassifier(metalevelData[i]);
			firstlevelEnsemble[i].buildClassifier(firstlevelData[i]);
		}
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

		for (int k = 0; k < firstlevelEnsemble.length; k++) {
			String name = firstlevelData[k].classAttribute().toString();
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
			newInstance.setDataset(firstlevelData[labelIndex]);

			double distribution[] = new double[2];
			try {
				distribution = firstlevelEnsemble[labelIndex]
						.distributionForInstance(newInstance);
			} catch (Exception e) {
				System.out.println(e);
				return null;
			}

			// Ensure correct predictions both for class values {0,1} and {1,0}
			Attribute classAttribute = firstlevelData[labelIndex]
					.classAttribute();
			// bipartition[labelIndex] =
			// (classAttribute.value(maxIndex).equals("1")) ? true : false;

			// The confidence of the label being equal to 1
			confidences[labelIndex] = distribution[classAttribute
					.indexOfValue("1")];
		}

		/* creation of the meta-instance with the appropriate values */
		double[] classes = new double[numLabels];
		// for (int i = 0; i < numLabels; i++) {
		// classes[i] =
		// instance.value(instance.numAttributes()-numLabels +i);
		// }
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
			newmetaInstance.setDataset(metalevelData[labelIndex]);

			double distribution[] = new double[2];
			try {
				distribution = metalevelEnsemble[labelIndex]
						.distributionForInstance(newmetaInstance);
			} catch (Exception e) {
				System.out.println(e);
				return null;
			}
			int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

			// Ensure correct predictions both for class values {0,1} and {1,0}
			Attribute classAttribute = metalevelData[labelIndex]
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
