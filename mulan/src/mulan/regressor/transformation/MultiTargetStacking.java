package mulan.regressor.transformation;

import java.util.ArrayList;
import java.util.HashSet;

import mulan.classifier.MultiLabelOutput;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import mulan.transformations.regression.SingleTargetTransformation;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;
import weka.filters.unsupervised.attribute.AddID;
import weka.filters.unsupervised.attribute.Remove;

/**
 * This class implements the Multi-Target Stacking (MTS) method. 4 alternative methods to obtain the values of
 * the meta features are implemented.<br>
 * For more information, see:<br>
 * <em>E. Spyromitros-Xioufis, G. Tsoumakas, W. Groves, I. Vlahavas. 2014. Multi-label Classification Methods for
 * Multi-target Regression. <a href="http://arxiv.org/abs/1211.6581">arXiv e-prints</a></em>.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.04.01
 */
public class MultiTargetStacking extends TransformationBasedMultiTargetRegressor {

    private static final long serialVersionUID = 1L;

    /**
     * The 4 alternative methods to obtain the values of the meta features.
     */
    public enum metaType {
        /**
         * Using internal k fold cross-validation.
         */
        CV,
        /**
         * Using the full training set.
         */
        TRAIN,
        /**
         * Using the true target values.
         */
        TRUE,
        /**
         * Using a random sample.
         */
        SAMPLE
    }

    /**
     * The method used to obtain the values of the meta features. TRAIN is used by default.
     */
    private metaType meta = metaType.TRAIN;

    /**
     * The number of folds to use in internal k fold cross-validation. 3 folds are used by default.
     */
    private int numFolds = 3;

    /**
     * Whether to include the original features attributes in the second stage training sets. True by default.
     */
    private boolean includeFeatures = true;

    /** The regressors of the first stage. */
    private Classifier[] firstStageRegressors;
    /** The regressors of the second stage. */
    private Classifier[] secondStageRegressors;

    /** The type of the regressor used in the second stage. */
    private Classifier secondStageBaseRegressor;

    /**
     * The values of the meta features (obtained using one of the available methods). The first dimension's
     * size is equalt to the number of training examples and the second is the number of targets.
     */
    private double[][] metaFeatures;
    /** The augmented datasets used to train the second stage regressors. */
    private Instances[] secondStageTrainsets;
    /** This transformation object is used for performing the transformation at the first stage. */
    private SingleTargetTransformation stt;

    /**
     * When the base regressor is capable of attribute selection this ArrayList holds the indices of the
     * target variables that were selected in each target's model.
     */
    protected ArrayList<Integer>[] selectedTargetIndices;
    /**
     * When the base regressor is capable of attribute selection this ArrayList holds the indices of the
     * normal feature variables that were selected in each target's model.
     */
    protected ArrayList<Integer>[] selectedFeatureIndices;

    /**
     * Creates a new instance with the given base regressor at both stages.
     * 
     * @param baseRegressor the base regression algorithm that will be used in both stages
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public MultiTargetStacking(Classifier baseRegressor) throws Exception {
        super(baseRegressor);
        this.secondStageBaseRegressor = baseRegressor;
    }

    /**
     * Creates a new instance with a different base regressor at each stage.
     * 
     * @param firstStageBaseRegressor the base regression algorithm that will be used in the first stage
     * @param secondStageBaseRegressor the base regression algorithm that will be used in the second stage
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public MultiTargetStacking(Classifier firstStageBaseRegressor,
            Classifier secondStageBaseRegressor) throws Exception {
        super(firstStageBaseRegressor);
        this.secondStageBaseRegressor = secondStageBaseRegressor;
    }

    @Override
    protected void buildInternal(MultiLabelInstances mlTrainSet) throws Exception {
        secondStageTrainsets = new Instances[numLabels];
        firstStageRegressors = AbstractClassifier.makeCopies(baseRegressor, numLabels);
        secondStageRegressors = AbstractClassifier.makeCopies(secondStageBaseRegressor, numLabels);
        metaFeatures = new double[mlTrainSet.getDataSet().numInstances()][numLabels];
        stt = new SingleTargetTransformation(mlTrainSet);
        selectedTargetIndices = new ArrayList[numLabels];
        selectedFeatureIndices = new ArrayList[numLabels];
        // any changes are applied to a copy of the original dataset
        Instances trainset = new Instances(mlTrainSet.getDataSet());
        buildFirstStage(trainset);
        buildSecondStage(trainset);
    }

    /**
     * Builds the first stage regressors and populates the {@link #metaFeatures} field.
     * 
     * @param trainSet
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    private void buildFirstStage(Instances trainSet) throws Exception {

        for (int targetIndex = 0; targetIndex < numLabels; targetIndex++) {
            // Transform the training set using the ST transformation
            Instances firstStageTrainSet = stt.transformInstances(targetIndex);
            debug("Building base regressor on full training set for target: " + targetIndex);
            firstStageRegressors[targetIndex].buildClassifier(firstStageTrainSet);

            if (meta == metaType.CV) {
                // attach an index attribute in order to keep track of the original
                // positions of the examples before the cross-validation
                AddID filter = new AddID();
                filter.setInputFormat(firstStageTrainSet);
                firstStageTrainSet = Filter.useFilter(firstStageTrainSet, filter);
                HashSet<Integer> indices = new HashSet<Integer>();
                for (int foldIndex = 0; foldIndex < numFolds; foldIndex++) {
                    // debug("Label=" + labelIndex + ", Fold=" + j);
                    Instances foldKTrainSet = firstStageTrainSet.trainCV(numFolds, foldIndex);
                    Instances foldKTestSet = firstStageTrainSet.testCV(numFolds, foldIndex);
                    // create a filtered meta classifier, used to ignore the index attribute in the
                    // build process
                    FilteredClassifier fil = new FilteredClassifier();
                    fil.setClassifier(AbstractClassifier.makeCopy(baseRegressor));
                    Remove remove = new Remove();
                    remove.setAttributeIndices("first");
                    remove.setInputFormat(foldKTrainSet);
                    fil.setFilter(remove);
                    fil.buildClassifier(foldKTrainSet);

                    // Make prediction for each test instance
                    for (int i = 0; i < foldKTestSet.numInstances(); i++) {
                        double score = fil.classifyInstance(foldKTestSet.instance(i));
                        // get index of the instance which was just classified
                        int index = (int) foldKTestSet.instance(i).value(0);
                        if (!indices.add(index)) {
                            System.out.println("Something went wrong: index" + index
                                    + " was already predicted!");
                        }
                        // The index starts from 1
                        metaFeatures[index - 1][targetIndex] = score;
                    }
                }
                if (indices.size() != trainSet.numInstances()) {
                    System.out.println("Something went wrong: indices size is " + indices.size()
                            + " instead of " + trainSet.numInstances());
                }
            } else if (meta == metaType.TRAIN) {
                // Make prediction for each in the training set instance
                for (int i = 0; i < firstStageTrainSet.numInstances(); i++) {
                    double score = firstStageRegressors[targetIndex]
                            .classifyInstance(firstStageTrainSet.instance(i));
                    metaFeatures[i][targetIndex] = score;
                }
            } else if (meta == metaType.TRUE) {
                for (int i = 0; i < firstStageTrainSet.numInstances(); i++) {
                    metaFeatures[i][targetIndex] = firstStageTrainSet.instance(i).classValue();
                }
            } else if (meta == metaType.SAMPLE) {
                // build a bagged first stageRegressor
                Resample resample = new Resample();
                resample.setRandomSeed(targetIndex); // the same seed could be used
                resample.setInputFormat(firstStageTrainSet);
                Instances sampledFirstStageTrainSet = Filter
                        .useFilter(firstStageTrainSet, resample);
                Classifier sampledfirstStageRegressor = AbstractClassifier.makeCopy(baseRegressor);
                sampledfirstStageRegressor.buildClassifier(sampledFirstStageTrainSet);

                // Make prediction for each instance in the training set
                for (int i = 0; i < firstStageTrainSet.numInstances(); i++) {
                    double score = sampledfirstStageRegressor.classifyInstance(firstStageTrainSet
                            .instance(i));
                    metaFeatures[i][targetIndex] = score;
                }
            }
        }
    }

    /**
     * Builds the second stage regressors.
     * 
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    private void buildSecondStage(Instances trainSet) throws Exception {
        debug("Building the second stage regressors.");
        for (int targetIndex = 0; targetIndex < numLabels; targetIndex++) {
            selectedTargetIndices[targetIndex] = new ArrayList<Integer>();
            selectedFeatureIndices[targetIndex] = new ArrayList<Integer>();
            // creating the second stage training set for each target
            ArrayList<Attribute> attributes = new ArrayList<Attribute>();
            if (includeFeatures) {
                // create an ArrayList with numAttributes-1 size.
                for (int j = 0; j < trainSet.numAttributes(); j++) {
                    // we do not include the values of this target in the meta features!
                    if (j != labelIndices[targetIndex]) {
                        attributes.add(trainSet.attribute(j));
                    }
                }
            } else {
                // create an ArrayList with numLabels size
                for (int j = 0; j < numLabels; j++) {
                    attributes.add(trainSet.attribute(labelIndices[j]));
                }
            }
            String targetName = trainSet.attribute(labelIndices[targetIndex]).name();
            attributes.add(trainSet.attribute(labelIndices[targetIndex])
                    .copy(targetName + "_final"));

            secondStageTrainsets[targetIndex] = new Instances(
                    "Second stage training set for target: " + targetName, attributes, 0);
            secondStageTrainsets[targetIndex].setClassIndex(secondStageTrainsets[targetIndex]
                    .numAttributes() - 1);

            // add instances
            for (int i = 0; i < trainSet.numInstances(); i++) {
                double[] values = new double[secondStageTrainsets[targetIndex].numAttributes()];
                if (includeFeatures) {
                    // copy the original feature values
                    for (int featureIndex = 0; featureIndex < featureIndices.length; featureIndex++) {
                        values[featureIndex] = trainSet.instance(i).value(
                                featureIndices[featureIndex]);
                    }
                    // copy the meta feature values as additional features
                    int index = 0;
                    for (int metaFeatureIndex = 0; metaFeatureIndex < numLabels; metaFeatureIndex++) {
                        if (metaFeatureIndex != targetIndex) {
                            values[trainSet.numAttributes() - numLabels + index] = metaFeatures[i][metaFeatureIndex];
                            index++;
                        }
                    }
                } else {
                    for (int metaFeatureIndex = 0; metaFeatureIndex < numLabels; metaFeatureIndex++) {
                        values[metaFeatureIndex] = metaFeatures[i][metaFeatureIndex];
                    }
                }

                values[values.length - 1] = trainSet.instance(i).value(labelIndices[targetIndex]);
                Instance metaInstance = DataUtils.createInstance(trainSet.instance(i), 1, values);
                metaInstance.setDataset(secondStageTrainsets[targetIndex]);
                secondStageTrainsets[targetIndex].add(metaInstance);
            }
            // debug("Building classifier for meta training set " + targetIndex);
            secondStageRegressors[targetIndex].buildClassifier(secondStageTrainsets[targetIndex]);
            secondStageTrainsets[targetIndex].delete();

            String output = secondStageRegressors[targetIndex].toString();
            // if this is a classifier that performs attribute selection (i.e.
            // AttributeSelectedClassifier or InfoTheoreticFeatureSelectionClassifier)
            if (output.contains("Selected attributes: ")) {
                // gather and output information about which feature and which target attributes
                // were selected
                String selectedString = output.split("Selected attributes: ")[1].split(" :")[0];
                String[] selectedIndicesString = selectedString.split(",");
                for (int j = 0; j < selectedIndicesString.length; j++) {
                    int selectedIndex = Integer.parseInt(selectedIndicesString[j]) - 1;
                    boolean isTarget = false;
                    for (int k = 0; k < numLabels; k++) {
                        String nameOfKthTarget = trainSet.attribute(labelIndices[k]).name();
                        String nameOfSelectedAttribute = secondStageTrainsets[targetIndex]
                                .attribute(selectedIndex).name();
                        if (nameOfKthTarget.equals(nameOfSelectedAttribute)) {
                            selectedTargetIndices[targetIndex].add(labelIndices[k]);
                            isTarget = true;
                            break;
                        }
                    }
                    if (!isTarget) {
                        selectedFeatureIndices[targetIndex].add(selectedIndex);
                    }
                }

                System.err.println("# selected feature attributes for target " + targetIndex + ": "
                        + selectedFeatureIndices[targetIndex].size());
                System.err.println(selectedFeatureIndices[targetIndex].toString());
                System.err.println("# selected target attributes for target " + targetIndex + ": "
                        + selectedTargetIndices[targetIndex].size());
                System.err.println(selectedTargetIndices[targetIndex].toString());
                System.err.flush();
            }

        }
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        // the values predicted by the second stage regressors
        double[] finalPredictions = new double[numLabels];
        // the values predicted by the first stage regressors
        double[] firstStagePredictions = new double[numLabels];

        // getting prediction for each target
        for (int targetIndex = 0; targetIndex < numLabels; targetIndex++) {
            Instance transformedInstance = stt.transformInstance(instance, targetIndex);
            firstStagePredictions[targetIndex] = firstStageRegressors[targetIndex]
                    .classifyInstance(transformedInstance);
        }

        // creation of the meta-instance with the appropriate values
        double[] values = new double[numLabels + 1];

        if (includeFeatures) {
            values = new double[instance.numAttributes() + 1];
            // Copy the original features
            for (int m = 0; m < featureIndices.length; m++) {
                values[m] = instance.value(featureIndices[m]);
            }
            // Copy the predictions for other targets as additional features
            for (int m = 0; m < firstStagePredictions.length; m++) {
                values[instance.numAttributes() - numLabels + m] = firstStagePredictions[m];
            }
        } else {
            for (int m = 0; m < firstStagePredictions.length; m++) {
                values[m] = firstStagePredictions[m];
            }
        }

        // application of the second stage models to the secondStageInstance
        Instance secondStageInstance;
        for (int targetIndex = 0; targetIndex < numLabels; targetIndex++) {
            if (includeFeatures) {
                double[] newValues = new double[instance.numAttributes()];
                int index = 0;
                for (int j = 0; j < values.length; j++) {
                    if (j != labelIndices[targetIndex]) {
                        newValues[index] = values[j];
                        index++;
                    }
                }
                secondStageInstance = DataUtils.createInstance(instance, 1, newValues);
            } else {
                secondStageInstance = DataUtils.createInstance(instance, 1, values);
            }
            secondStageInstance.setDataset(secondStageTrainsets[targetIndex]);

            finalPredictions[targetIndex] = secondStageRegressors[targetIndex]
                    .classifyInstance(secondStageInstance);

        }

        MultiLabelOutput mlo = new MultiLabelOutput(finalPredictions, true);
        return mlo;
    }

    @Override
    protected String getModelForTarget(int targetIndex) {
        try {
            secondStageRegressors[targetIndex].getClass().getMethod("toString", (Class<?>[]) null);
        } catch (NoSuchMethodException e) {
            return "A string representation for this base algorithm is not provided!";
        }
        return secondStageRegressors[targetIndex].toString();
    }

    /**
     * Sets the value of {@link #includeFeatures}.
     * 
     * @param includeAttrs the setter value
     */
    public void setIncludeAttrs(boolean includeAttrs) {
        this.includeFeatures = includeAttrs;
    }

    /**
     * Sets the value of {@link #meta}.
     * 
     * @param meta the setter value
     */
    public void setMeta(metaType meta) {
        this.meta = meta;
    }

    public void setNumFolds(int numFolds) {
        this.numFolds = numFolds;
    }

    public ArrayList<Integer>[] getSelectedTargetIndices() {
        return selectedTargetIndices;
    }

    public ArrayList<Integer>[] getSelectedFeatureIndices() {
        return selectedFeatureIndices;
    }

}
