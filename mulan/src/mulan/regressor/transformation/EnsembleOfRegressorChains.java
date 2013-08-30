package mulan.regressor.transformation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.regressor.transformation.RegressorChainCorrected.metaType;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

/**
 * This class implements the Ensemble of Regressor Chains (ERC) method.<br/>
 * For more information, see:<br/>
 * E. Spyromitros-Xioufis, W. Groves, G. Tsoumakas, I. Vlahavas (2012). Multi-label Classification
 * Methods for Multi-target Regression. <a href="http://arxiv.org/abs/1211.6581">ArXiv e-prints</a>.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2013.07.28
 */
public class EnsembleOfRegressorChains extends TransformationBasedMultiTargetRegressor {

    private static final long serialVersionUID = 1L;

    /**
     * The number of RC models to be created.
     */
    private int numOfModels;

    /**
     * Stores the RC models.
     */
    private RegressorChainCorrected[] ensemble;

    /** The seed to use in random number generators. Default = 1. **/
    private int seed = 1;

    /**
     * Three types of sampling.
     */
    public enum SamplingMethod {
        None, WithReplacement, WithoutReplacement,
    };

    /**
     * The method used to obtain the values of the meta features. CV is used by default.
     */
    private metaType meta = RegressorChainCorrected.metaType.CV;

    /**
     * The type of sampling to be used. None is used by default.
     */
    private SamplingMethod sampling = SamplingMethod.None;

    /**
     * The size of each sample (as a percentage of the training set size) when sampling with
     * replacement is performed. Default is 100.
     */
    private double sampleWithReplacementPercent = 100;

    /**
     * The size of each sample (as a percentage of the training set size) when sampling without
     * replacement is performed. Default is 67.
     */
    private double sampleWithoutReplacementPercent = 67;

    /**
     * The number of folds to use in RegressorChainCorrected when CV is selected for obtaining the
     * values of the meta-features.
     */
    private int numFolds = 3;

    /**
     * Constructor.
     * 
     * @param baseRegressor the base regression algorithm that will be used
     * @param numOfModels the number of models in the ensemble
     * @param samplingMethod the sampling method
     * @throws Exception
     */
    public EnsembleOfRegressorChains(Classifier baseRegressor, int numOfModels,
            SamplingMethod sampling) throws Exception {
        super(baseRegressor);
        this.numOfModels = numOfModels;
        this.sampling = sampling;
        ensemble = new RegressorChainCorrected[numOfModels];
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        // calculate the number of models
        int numDistinctChains = 1;
        for (int i = 1; i <= numLabels; i++) {
            numDistinctChains *= i;
            if (numDistinctChains > numOfModels) {
                numDistinctChains = numOfModels;
                break;
            }
        }
        numOfModels = numDistinctChains;

        // will hold the distinct chains created so far
        HashSet<String> distinctChains = new HashSet<String>(numOfModels);

        // this random number generator will be used for taking random samples
        // and creating random chains
        Random rand = new Random(seed);

        for (int i = 0; i < numOfModels; i++) {
            debug("ERC Building Model:" + (i + 1) + "/" + numOfModels);
            MultiLabelInstances sampledTrainingSet = null;
            if (sampling != SamplingMethod.None) {
                // initialize a Resample filter using a different seed each time
                Resample rsmp = new Resample();
                rsmp.setRandomSeed(rand.nextInt());
                if (sampling == SamplingMethod.WithoutReplacement) {
                    rsmp.setNoReplacement(true);
                    rsmp.setSampleSizePercent(sampleWithoutReplacementPercent);
                } else {
                    rsmp.setNoReplacement(false);
                    rsmp.setSampleSizePercent(sampleWithReplacementPercent);
                }
                rsmp.setInputFormat(trainingSet.getDataSet());
                Instances sampled = Filter.useFilter(trainingSet.getDataSet(), rsmp);
                sampledTrainingSet = new MultiLabelInstances(sampled,
                        trainingSet.getLabelsMetaData());
            }

            // create a distinct chain
            int[] chain = new int[numLabels];
            while (true) {
                for (int j = 0; j < numLabels; j++) { // the default chain
                    chain[j] = labelIndices[j];
                }
                ArrayList<Integer> chainAsList = new ArrayList<Integer>(numLabels);
                for (int j = 0; j < numLabels; j++) {
                    chainAsList.add(chain[j]);
                }
                Collections.shuffle(chainAsList, rand);
                for (int j = 0; j < numLabels; j++) {
                    chain[j] = chainAsList.get(j);
                }
                String chainString = chainAsList.toString();
                if (distinctChains.add(chainString)) {
                    // the chain is not in the set so we can break the loop
                    break;
                }
            }

            ensemble[i] = new RegressorChainCorrected(baseRegressor, chain);
            ensemble[i].setNumFolds(numFolds);
            ensemble[i].setMeta(meta);
            ensemble[i].setDebug(getDebug());
            if (sampling == SamplingMethod.None) {
                ensemble[i].build(trainingSet);
            } else {
                ensemble[i].build(sampledTrainingSet);
            }

        }

    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception,
            InvalidDataException {
        double[] scores = new double[numLabels];

        for (int i = 0; i < numOfModels; i++) {
            MultiLabelOutput ensembleMLO = ensemble[i].makePrediction(instance);
            double[] score = ensembleMLO.getPvalues();
            for (int j = 0; j < numLabels; j++) {
                scores[j] += score[j];
            }
        }

        for (int j = 0; j < numLabels; j++) {
            scores[j] /= numOfModels;
        }

        MultiLabelOutput mlo = new MultiLabelOutput(scores, true);
        return mlo;
    }

    @Override
    protected String getModelForTarget(int target) {
        StringBuffer output = new StringBuffer();
        for (int i = 0; i < numOfModels; i++) {
            output.append("Ensemble of Regressor Chains: " + (i + 1) + "\n");
            output.append(ensemble[i].getModel() + "\n");
        }
        return output.toString();
    }

    public void setSampleWithReplacementPercent(int sampleWithReplacementPercent) {
        this.sampleWithReplacementPercent = sampleWithReplacementPercent;
    }

    public void setSampleWithoutReplacementPercent(double sampleWithoutReplacementPercent) {
        this.sampleWithoutReplacementPercent = sampleWithoutReplacementPercent;
    }

    public void setNumFolds(int numFolds) {
        this.numFolds = numFolds;
    }

    public void setMeta(metaType meta) {
        this.meta = meta;
    }

    public void setSeed(int seed) {
        this.seed = seed;
    }

}
