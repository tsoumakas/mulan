package mulan.regressor.transformation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.regressor.transformation.RegressorChain.metaType;
import weka.classifiers.Classifier;
import weka.classifiers.trees.REPTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

/**
 * This class implements the Ensemble of Regressor Chains (ERC) method.<br>
 * For more information, see:<br>
 * <em>E. Spyromitros-Xioufis, G. Tsoumakas, W. Groves, I. Vlahavas. 2014. Multi-label Classification Methods for
 * Multi-target Regression. <a href="http://arxiv.org/abs/1211.6581">arXiv e-prints</a></em>.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.04.01
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
    private RegressorChain[] ensemble;

    /** The seed to use in random number generators. Default = 1. **/
    private int seed = 1;

    /**
     * Three types of sampling.
     */
    public enum SamplingMethod {
        None, WithReplacement, WithoutReplacement,
    };

    /**
     * The method used to obtain the values of the meta features. TRUE is used by default.
     */
    private metaType meta = RegressorChain.metaType.TRUE;

    /**
     * The type of sampling to be used. None is used by default.
     */
    private SamplingMethod sampling = SamplingMethod.None;

    /**
     * The size of each sample (as a percentage of the training set size) when sampling with replacement is
     * performed. Default is 100.
     */
    private double sampleWithReplacementPercent = 100;

    /**
     * The size of each sample (as a percentage of the training set size) when sampling without replacement is
     * performed. Default is 67.
     */
    private double sampleWithoutReplacementPercent = 67;

    /**
     * The number of folds to use in RegressorChainCorrected when CV is selected for obtaining the values of
     * the meta-features.
     */
    private int numFolds = 3;

    /**
     * Default constructor.
     * 
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public EnsembleOfRegressorChains() throws Exception {
        this(new REPTree(), 10, SamplingMethod.WithReplacement);
    }

    /**
     * Constructor.
     * 
     * @param baseRegressor the base regression algorithm that will be used
     * @param numOfModels the number of models in the ensemble
     * @param sampling the sampling method
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public EnsembleOfRegressorChains(Classifier baseRegressor, int numOfModels,
            SamplingMethod sampling) throws Exception {
        super(baseRegressor);
        this.numOfModels = numOfModels;
        this.sampling = sampling;
        ensemble = new RegressorChain[numOfModels];
    }

    @Override
    protected void buildInternal(MultiLabelInstances mlTrainSet) throws Exception {
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
                Instances trainSet = new Instances(mlTrainSet.getDataSet());
                rsmp.setInputFormat(trainSet);
                Instances sampled = Filter.useFilter(trainSet, rsmp);
                sampledTrainingSet = new MultiLabelInstances(sampled,
                        mlTrainSet.getLabelsMetaData());
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

            ensemble[i] = new RegressorChain(baseRegressor, chain);
            ensemble[i].setNumFolds(numFolds);
            ensemble[i].setMeta(meta);
            ensemble[i].setDebug(getDebug());
            if (sampling == SamplingMethod.None) {
                ensemble[i].build(mlTrainSet);
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
    protected String getModelForTarget(int targetIndex) {
        StringBuffer output = new StringBuffer();
        for (int i = 0; i < numOfModels; i++) {
            output.append("Ensemble member: " + (i + 1) + "\n");
            output.append(ensemble[i].getModelForTarget(targetIndex) + "\n");
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
