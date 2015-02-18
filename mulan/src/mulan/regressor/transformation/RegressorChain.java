package mulan.regressor.transformation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;

import mulan.classifier.MultiLabelOutput;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import mulan.transformations.regression.ChainTransformation;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddID;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.Resample;

/**
 * This class implements the Regressor Chain (RC) method. 4 alternative methods to obtain the values of the
 * meta features are implemented.<br>
 * For more information, see:<br>
 * <em>E. Spyromitros-Xioufis, G. Tsoumakas, W. Groves, I. Vlahavas. 2014. Multi-label Classification Methods for
 * Multi-target Regression. <a href="http://arxiv.org/abs/1211.6581">arXiv e-prints</a></em>.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.04.01
 */
public class RegressorChain extends TransformationBasedMultiTargetRegressor {

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
     * The method used to obtain the values of the meta features. TRUE is used by default.
     */
    private metaType meta = metaType.TRUE;

    /**
     * A permutation of the target indices. E.g. If there are 3 targets with indices 14, 15, 16, a valid chain
     * is 15, 14, 16.
     */
    private int[] chain;

    /**
     * The seed to use for random number generation in order to create a random chain (other than the default
     * one which consists of the targets chained in the order they appear in the arff file).
     */
    private int chainSeed = 0;

    /**
     * The number of folds to use in internal k fold cross-validation. 3 folds are used by default.
     */
    private int numFolds = 3;

    /**
     * The training data of each regressor of the chain. After training the actual data are deleted and only
     * the header information is held which is needed during prediction.
     */
    private Instances[] chainRegressorsTrainSets;

    /** The regressors of the chain. */
    private Classifier[] chainRegressors;

    /**
     * The values of the meta features (obtained using one of the available methods). The first dimension's
     * size is equal to the number of training examples and the second is equal to the number of targets minus
     * 1 (we do need meta features for the last target of the chain).
     */
    private double[][] metaFeatures;

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
     * Creates a new instance with the given base regressor. If {@link #chainSeed} == 0, the default chain is
     * used. Otherwise, a random chain is created using the given seed.
     * 
     * @param baseRegressor the base regression algorithm that will be used
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public RegressorChain(Classifier baseRegressor) throws Exception {
        super(baseRegressor);
    }

    /**
     * Creates a new instance with the given base regressor and chain ordering.
     * 
     * @param baseRegressor the base regression algorithm that will be used
     * @param aChain a chain ordering
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public RegressorChain(Classifier baseRegressor, int[] aChain) throws Exception {
        super(baseRegressor);
        chain = aChain;
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainSet) throws Exception {
        // =============================== INITIALIZATION START ===============================
        chainRegressorsTrainSets = new Instances[numLabels];
        metaFeatures = new double[trainSet.getNumInstances()][numLabels - 1];
        chainRegressors = new Classifier[numLabels];
        selectedTargetIndices = new ArrayList[numLabels];
        selectedFeatureIndices = new ArrayList[numLabels];
        // =============================== INITIALIZATION END =================================

        // =============================== CHAIN CREATION START ===============================
        // if no chain has been defined, create the default chain
        if (chain == null) {
            chain = new int[numLabels];
            for (int j = 0; j < numLabels; j++) {
                chain[j] = labelIndices[j];
            }
        }

        if (chainSeed != 0) { // a random chain will be created by shuffling the existing chain
            Random rand = new Random(chainSeed);
            ArrayList<Integer> chainAsList = new ArrayList<Integer>(numLabels);
            for (int j = 0; j < numLabels; j++) {
                chainAsList.add(chain[j]);
            }
            Collections.shuffle(chainAsList, rand);
            for (int j = 0; j < numLabels; j++) {
                chain[j] = chainAsList.get(j);
            }
        }
        debug("Using chain: " + Arrays.toString(chain));
        // =============================== CHAIN CREATION END =================================

        for (int targetIndex = 0; targetIndex < numLabels; targetIndex++) {
            selectedTargetIndices[targetIndex] = new ArrayList<Integer>();
            selectedFeatureIndices[targetIndex] = new ArrayList<Integer>();
            chainRegressors[targetIndex] = AbstractClassifier.makeCopy(baseRegressor);

            // ======================= TRAINING SET CREATION START============================
            // create a copy of the training set and transform it according to the CC transformation
            // the copy is probably not needed
            Instances trainCopy = new Instances(trainSet.getDataSet());
            chainRegressorsTrainSets[targetIndex] = ChainTransformation.transformInstances(
                    trainCopy, chain, targetIndex + 1);

            // if it is not the first target in the chain and if we are not using the true values
            // (as in original RC), use the values stored in metaFeatures to update the training set
            if (targetIndex > 0 && meta != metaType.TRUE) {
                // replace the true values of the targets with the predictions made by the
                // previous regressors in the chain
                for (int j = 0; j < targetIndex; j++) {
                    // get the name of the target for which predictions have been made in the
                    // previous iteration
                    String targetName = chainRegressorsTrainSets[targetIndex - 1 - j]
                            .classAttribute().name();
                    // get the index of the attribute in the new dataset
                    int indexInNewDataset = chainRegressorsTrainSets[targetIndex].attribute(
                            targetName).index();

                    for (int i = 0; i < chainRegressorsTrainSets[targetIndex].numInstances(); i++) {
                        // get the predicted value of that attribute
                        double predictedValue = metaFeatures[i][targetIndex - 1 - j];
                        // replace the true value with the prediction
                        chainRegressorsTrainSets[targetIndex].instance(i).setValue(
                                indexInNewDataset, predictedValue);
                    }
                }
            }
            // ========================== TRAINING SET CREATION ENDED ===========================

            // =========================== REGRESSOR TRAINING START =============================
            debug("RC bulding model " + (targetIndex + 1) + "/" + numLabels + " (for target "
                    + chainRegressorsTrainSets[targetIndex].classAttribute().name() + ")");
            chainRegressors[targetIndex].buildClassifier(chainRegressorsTrainSets[targetIndex]);
            // =========================== REGRESSOR TRAINING ENDED =============================

            String output = chainRegressors[targetIndex].toString();
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
                        String nameOfKthTarget = trainSet.getDataSet().attribute(labelIndices[k])
                                .name();
                        String nameOfSelectedAttribute = chainRegressorsTrainSets[targetIndex]
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

            // ============================ META FEATURE CREATION START ============================
            // we do not need a meta feature for the last target in the chain
            if (targetIndex < numLabels - 1) {

                if (meta == metaType.CV) {
                    // attach an index attribute in order to keep track of the original
                    // positions of the examples before the internal cross-validation
                    AddID filter = new AddID();
                    filter.setInputFormat(chainRegressorsTrainSets[targetIndex]);
                    chainRegressorsTrainSets[targetIndex] = Filter.useFilter(
                            chainRegressorsTrainSets[targetIndex], filter);

                    // debug("Performing internal cv to get predictions (for target "
                    // + chainRegressorsTrainSets[labelIndex].classAttribute().name() + ")");
                    // perform k-fold cross-validation and save the predictions which
                    // will be used by the next regressor in the chain in order to build its model
                    HashSet<Integer> indices = new HashSet<Integer>();
                    for (int foldIndex = 0; foldIndex < numFolds; foldIndex++) {
                        // debug("Label=" + labelIndex + ", Fold=" + foldIndex);
                        // create the training and test set for the current fold
                        Instances foldKTrainset = chainRegressorsTrainSets[targetIndex].trainCV(
                                numFolds, foldIndex);
                        Instances foldKTestset = chainRegressorsTrainSets[targetIndex].testCV(
                                numFolds, foldIndex);
                        // create a filtered meta classifier, used to ignore
                        // the ID attribute in the build process
                        FilteredClassifier fil = new FilteredClassifier();
                        fil.setClassifier(AbstractClassifier.makeCopy(baseRegressor));
                        Remove remove = new Remove();
                        remove.setAttributeIndices("first");
                        remove.setInputFormat(foldKTrainset);
                        fil.setFilter(remove);
                        fil.buildClassifier(foldKTrainset);

                        // Make prediction for each test instance
                        for (int i = 0; i < foldKTestset.numInstances(); i++) {
                            double score = fil.classifyInstance(foldKTestset.instance(i));
                            // get index of the instance which was just classified
                            int index = (int) foldKTestset.instance(i).value(0);
                            if (!indices.add(index)) {
                                System.out.println("Something went wrong: index" + index
                                        + " was already predicted!");
                            }
                            // The index starts from 1
                            metaFeatures[index - 1][targetIndex] = score;
                        }
                    }
                    if (indices.size() != trainSet.getNumInstances()) {
                        System.out.println("Something went wrong: indices size is "
                                + indices.size() + " instead of " + trainSet.getNumInstances());
                    }
                    // now we can detach the indices from this target's training set
                    Remove remove = new Remove();
                    remove.setAttributeIndices("first");
                    remove.setInputFormat(chainRegressorsTrainSets[targetIndex]);
                    chainRegressorsTrainSets[targetIndex] = Filter.useFilter(
                            chainRegressorsTrainSets[targetIndex], remove);
                } else if (meta == metaType.TRAIN) {
                    // Make prediction for each in the training set instance
                    for (int i = 0; i < chainRegressorsTrainSets[targetIndex].numInstances(); i++) {
                        double score = chainRegressors[targetIndex]
                                .classifyInstance(chainRegressorsTrainSets[targetIndex].instance(i));
                        metaFeatures[i][targetIndex] = score;
                    }
                } else if (meta == metaType.TRUE) {
                    for (int i = 0; i < chainRegressorsTrainSets[targetIndex].numInstances(); i++) {
                        metaFeatures[i][targetIndex] = chainRegressorsTrainSets[targetIndex]
                                .instance(i).classValue();
                    }
                } else if (meta == metaType.SAMPLE) {
                    // build a bagged first stageRegressor
                    Resample resample = new Resample();
                    resample.setRandomSeed(targetIndex); // the same seed could be used
                    resample.setNoReplacement(false); // setting this to true could be tried
                    resample.setInputFormat(chainRegressorsTrainSets[targetIndex]);
                    Instances sampledChainRegressorTrainSet = Filter.useFilter(
                            chainRegressorsTrainSets[targetIndex], resample);
                    Classifier sampledChainRegressor = AbstractClassifier.makeCopy(baseRegressor);
                    sampledChainRegressor.buildClassifier(sampledChainRegressorTrainSet);

                    // Make prediction for each in the training set instance
                    for (int i = 0; i < chainRegressorsTrainSets[targetIndex].numInstances(); i++) {
                        double score = sampledChainRegressor
                                .classifyInstance(chainRegressorsTrainSets[targetIndex].instance(i));
                        metaFeatures[i][targetIndex] = score;
                    }
                }
                // these data are no more needed so they are deleted to save some memory
                chainRegressorsTrainSets[targetIndex].delete();
            }
            // ============================ META FEATURE CREATION ENDED ============================
        }
    }

    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        double[] scores = new double[numLabels];

        Instance copyOfInstance = DataUtils.createInstance(instance, instance.weight(),
                instance.toDoubleArray());
        copyOfInstance.setDataset(instance.dataset());

        for (int counter = 0; counter < numLabels; counter++) {
            Instance temp = ChainTransformation.transformInstance(copyOfInstance, chain,
                    counter + 1);
            temp.setDataset(chainRegressorsTrainSets[counter]);

            double score = chainRegressors[counter].classifyInstance(temp);

            // find the appropriate position for that score in the scores array
            // i.e. which is the corresponding target
            int pos = 0;
            for (int i = 0; i < numLabels; i++) {
                if (chain[counter] == labelIndices[i]) {
                    pos = i;
                }
            }
            scores[pos] = score;
            copyOfInstance.setValue(chain[counter], score);
        }

        MultiLabelOutput mlo = new MultiLabelOutput(scores, true);
        return mlo;
    }

    @Override
    protected String getModelForTarget(int targetIndex) {
        // find the position of this target in the chain
        int posInChain = -1;
        for (int i = 0; i < numLabels; i++) {
            if (chain[i] == labelIndices[targetIndex]) {
                posInChain = i;
            }
        }
        try {
            chainRegressors[posInChain].getClass().getMethod("toString", (Class<?>[]) null);
        } catch (NoSuchMethodException e) {
            return "A string representation for this base algorithm is not provided!";
        }
        return chainRegressors[posInChain].toString();
    }

    public void setMeta(metaType meta) {
        this.meta = meta;
    }

    public void setNumFolds(int numFolds) {
        this.numFolds = numFolds;
    }

    public void setChainSeed(int chainSeed) {
        this.chainSeed = chainSeed;
    }

    public ArrayList<Integer>[] getSelectedTargetIndices() {
        return selectedTargetIndices;
    }

    public ArrayList<Integer>[] getSelectedFeatureIndices() {
        return selectedFeatureIndices;
    }

}
