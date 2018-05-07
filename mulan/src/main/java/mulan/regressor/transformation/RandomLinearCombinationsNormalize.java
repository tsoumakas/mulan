package mulan.regressor.transformation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.matrix.LinearRegression;
import weka.core.matrix.Matrix;
import weka.filters.unsupervised.attribute.Remove;

/**
 * <p>
 * Implementation of the multi-target regression method presented in
 * <em>Grigorios Tsoumakas, Eleftherios Spyromitros-Xioufis, Aikaterini Vrekou, Ioannis Vlahavas. 2014. Multi-Target Regression via Random Linear Target
 * Combinations. <a href="http://arxiv.org/abs/1404.5065">arXiv e-prints</a></em>
 * </p>
 * This method extends {@link RandomLinearCombinations}, performing normalization of the target variables internally and
 * applying the reverse transformation during prediction.
 * 
 * @author Grigorios Tsoumakas
 * @author Aikaterini Vrekou
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.03.31
 */
public class RandomLinearCombinationsNormalize extends TransformationBasedMultiTargetRegressor {

	/** The ensemble models */
	private FilteredClassifier[] models;
	private double[][] coefficients;

	/** Matrix containing the coefficients for regression */
	private Matrix coefficientsMatrix;

	private int numCombinations;
	private int nonZero;
	private Random generator;

	/** for speeding up evaluation by training once and testing many */
	private int numModels;

	private static final long serialVersionUID = 1L;

	/** store min and max values of target variables in training set */
	private double[] maxTargetValues;
	private double[] minTargetValues;

	public RandomLinearCombinationsNormalize(int numCombinations, long aSeed, Classifier baseRegressor, int nonZero) {
		super(baseRegressor);
		this.numCombinations = numCombinations;
		this.nonZero = nonZero;
		numModels = numCombinations;
		generator = new Random(aSeed);
	}

	/**
	 * This function is used to speed-up experiments with different number of models, which we built only once, but test
	 * multiple times.
	 * 
	 */
	public double getPredictionOfModel(Instance instance, int model) {
		try {
			return models[model].classifyInstance(instance);
		} catch (Exception ex) {
			Logger.getLogger(RandomLinearCombinationsNormalize.class.getName()).log(Level.SEVERE, null, ex);
			return 0;
		}
	}

	@Override
	protected void buildInternal(MultiLabelInstances train) throws Exception {

		models = new FilteredClassifier[numCombinations];
		coefficients = new double[numCombinations][numLabels];
		maxTargetValues = new double[numLabels];
		minTargetValues = new double[numLabels];

		int[] picked = new int[numLabels]; // stores the times each target was
											// picked

		for (int i = 0; i < numCombinations; i++) {
			models[i] = new FilteredClassifier();
			models[i].setClassifier(AbstractClassifier.makeCopy(baseRegressor));

			int counter = 0;
			ArrayList<Integer> nonZeroTargets = new ArrayList<>();
			do {
				int addedTarget = generator.nextInt(numLabels); // range [0,
																// numLabels)
				if (!nonZeroTargets.contains(addedTarget)) {
					// find the minimum times a target was picked
					int min = numCombinations;
					for (int k = 0; k < numLabels; k++) {
						if (picked[k] < min) {
							min = picked[k];
						}
					}
					// pick a target from the ones with the minimum times picked
					if (picked[addedTarget] == min) {
						nonZeroTargets.add(addedTarget);
						counter++;
						picked[addedTarget]++;
					}
				}
			} while (counter != nonZero);

			for (int k = 0; k < nonZero; k++) {
				int target = nonZeroTargets.get(k);
				double factor = generator.nextDouble();
				coefficients[i][target] = factor;
			}

			// for (double[] coeff : coefficients) {
			// System.out.println(Arrays.toString(coeff));
			// }

			Instances originalData = train.getDataSet();
			Instances changedData = new Instances(originalData);
			// first apply normalization on the data and store normalization ranges
			for (int j = 0; j < numLabels; j++) {
				double[] values = changedData.attributeToDoubleArray(labelIndices[j]);
				// determine max and min values
				maxTargetValues[j] = values[Utils.maxIndex(values)];
				minTargetValues[j] = values[Utils.minIndex(values)];

				// normalize the attribute using these values
				for (int k = 0; k < changedData.numInstances(); k++) {
					double originalValue = changedData.instance(k).value(labelIndices[j]);
					double normalizedValue = (originalValue - minTargetValues[j])
							/ (maxTargetValues[j] - minTargetValues[j]);
					changedData.instance(k).setValue(labelIndices[j], normalizedValue);
				}
			}

			int totalAtts = originalData.numAttributes();
			int predictiveAtts = totalAtts - numLabels;
			Iterator<Instance> trainIt = changedData.iterator();

			// for each instance put the new y value to the 1st target attribute
			while (trainIt.hasNext()) {
				Instance instance = trainIt.next();
				double y = 0;
				for (int r = predictiveAtts; r < totalAtts; r++) {
					double num = instance.value(r);
					y = y + num * coefficients[i][r - predictiveAtts];
				}
				instance.setValue(predictiveAtts, y);
			}
			changedData.setClassIndex(predictiveAtts);

			// Remove all target attributes except the 1st that has the new
			// values
			String[] options = new String[2];
			options[0] = "-R";
			options[1] = (predictiveAtts + 2) + "-" + totalAtts;
			Remove remove = new Remove();
			remove.setOptions(options);
			remove.setInputFormat(changedData);
			models[i].setFilter(remove);

			debug("Building model " + (i + 1) + "/" + numCombinations + " :" + (i + 1));
			models[i].buildClassifier(changedData);
			debug("Built model " + (i + 1) + "/" + numCombinations + " :" + (i + 1));
		}

		coefficientsMatrix = new Matrix(coefficients);
	}

	/**
	 * Sets the number of models to use during prediction
	 * 
	 * @param numModels
	 *            the number of models to use
	 */
	public void setNumModels(int numModels) {
		if (numModels < 1 || numModels > numCombinations) {
			throw new IllegalArgumentException("Num models should be in [1..numCombinations]");
		}
		this.numModels = numModels;
		coefficientsMatrix = new Matrix(Arrays.copyOfRange(coefficients, 0, numModels));
	}

	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException {

		Instances dataset = instance.dataset();
		dataset.setClassIndex(dataset.numAttributes() - 1);

		Matrix meta = new Matrix(numModels, 1);
		for (int i = 0; i < numModels; i++) {
			meta.set(i, 0, models[i].classifyInstance(instance));
		}
		LinearRegression lr = new LinearRegression(coefficientsMatrix, meta, 0);

		double[] predictions = lr.getCoefficients();
		// translate back to original range
		for (int i = 0; i < numLabels; i++) {
			predictions[i] = minTargetValues[i] + (maxTargetValues[i] - minTargetValues[i]) * predictions[i];
		}

		MultiLabelOutput mlo = new MultiLabelOutput(lr.getCoefficients(), true);
		return mlo;
	}

	@Override
	protected String getModelForTarget(int targetIndex) {
		// TODO Auto-generated method stub
		return null;
	}

}
