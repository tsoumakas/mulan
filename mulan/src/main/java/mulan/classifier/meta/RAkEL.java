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
package mulan.classifier.meta;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 <!-- globalinfo-start -->
 * Class implementing a generalized version of the RAkEL (RAndom k-labELsets) algorithm. For more information, see<br>
 * <br>
 * Grigorios Tsoumakas, Ioannis Katakis, Ioannis Vlahavas (2011). Random k-Labelsets for Multi-Label Classification. IEEE Transactions on Knowledge and Data Engineering. 23(7):1079-1089.
 * <br>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Tsoumakas2011,
 *    author = {Grigorios Tsoumakas and Ioannis Katakis and Ioannis Vlahavas},
 *    journal = {IEEE Transactions on Knowledge and Data Engineering},
 *    number = {7},
 *    pages = {1079-1089},
 *    title = {Random k-Labelsets for Multi-Label Classification},
 *    volume = {23},
 *    year = {2011}
 * }
 * </pre>
 * <br>
 <!-- technical-bibtex-end -->
 *
 * @author Grigorios Tsoumakas
 * @version 2012.1.27
 */
@SuppressWarnings("serial")
public class RAkEL extends MultiLabelMetaLearner {

    /**
     * Seed for replication of random experiments
     */
    private int seed = 0;
    /**
     * Random number generator
     */
    private Random rnd;
    /**
     * If true then the confidence of the base classifier to the decisions...
     */
    //private boolean useConfidences = true;
    double[][] sumVotesIncremental; /*
     * comment
     */

    double[][] lengthVotesIncremental;
    double[] sumVotes;
    double[] lengthVotes;
    int numOfModels;
    double threshold = 0.5;
    int sizeOfSubset = 3;
    int[][] classIndicesPerSubset;
    int[][] absoluteIndicesToRemove;
    MultiLabelLearner[] subsetClassifiers;
    private Remove[] remove;
    HashSet<String> combinations;

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.ARTICLE);
        result.setValue(Field.AUTHOR, "Grigorios Tsoumakas and Ioannis Katakis and Ioannis Vlahavas");
        result.setValue(Field.TITLE, "Random k-Labelsets for Multi-Label Classification");
        result.setValue(Field.JOURNAL, "IEEE Transactions on Knowledge and Data Engineering");
        result.setValue(Field.PAGES, "1079-1089");
        result.setValue(Field.VOLUME, "23");
        result.setValue(Field.NUMBER, "7");
        result.setValue(Field.YEAR, "2011");
        return result;
    }

    /**
     * Default constructor
     */
    public RAkEL() {
        this(new BinaryRelevance(new J48()));
    }

    /**
     * Creates an instance based on a given multi-label learner
     * 
     * @param baseLearner the multi-label learner
     */
    public RAkEL(MultiLabelLearner baseLearner) {
        super(baseLearner);
    }

    /**
     * Creates an instance given a specific multi-label learner, number of 
     * models and size of subsets
     * 
     * @param baseLearner a multi-label learner
     * @param models a number of models
     * @param subset a size of subsets
     */
    public RAkEL(MultiLabelLearner baseLearner, int models, int subset) {
        super(baseLearner);
        sizeOfSubset = subset;
        numOfModels = models;
    }

    /**
     * Creates an instance given a specific multi-label learner, number of 
     * models, size of subsets and threshold
     * 
     * @param baseLearner a multi-label learner
     * @param models a number of models
     * @param subset a size of subsets
     * @param threshold a threshold
     */
    public RAkEL(MultiLabelLearner baseLearner, int models, int subset, double threshold) {
        super(baseLearner);
        sizeOfSubset = subset;
        numOfModels = models;
        this.threshold = threshold;
    }

    /**
     * Sets the seed for random number generation
     * 
     * @param x the seed
     */
    public void setSeed(int x) {
        seed = x;
    }

    /**
     * Sets the size of the subsets
     * 
     * @param size the size of the subsets
     */
    public void setSizeOfSubset(int size) {
        sizeOfSubset = size;
        classIndicesPerSubset = new int[numOfModels][sizeOfSubset];
    }

    /**
     * Returns the size of the subsets
     * 
     * @return the size of the subsets
     */
    public int getSizeOfSubset() {
        return sizeOfSubset;
    }

    /**
     * Sets the number of models
     * 
     * @param models number of models
     */
    public void setNumModels(int models) {
        numOfModels = models;
    }

    /**
     * Returns the number of models
     * 
     * @return number of models
     */
    public int getNumModels() {
        return numOfModels;
    }

    /**
     * The binomial function
     * 
     * @param n Binomial coefficient index
     * @param m Binomial coefficient index
     * @return The result of the binomial function
     */
    public static int binomial(int n, int m) {
        int[] b = new int[n + 1];
        b[0] = 1;
        for (int i = 1; i <= n; i++) {
            b[i] = 1;
            for (int j = i - 1; j > 0; --j) {
                b[j] += b[j - 1];
            }
        }
        return b[m];
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingData) throws Exception {
        rnd = new Random(seed);

        // need a structure to hold different combinations
        combinations = new HashSet<String>();
        //MultiLabelInstances mlDataSet = trainData.clone();

        // check whether sizeOfSubset is larger or equal compared to number of labels
        if (sizeOfSubset >= numLabels) {
            throw new IllegalArgumentException("Size of subsets should be less than the number of labels");
        }

        // default number of models = twice the number of labels
        if (numOfModels == 0) {
            numOfModels = Math.min(2 * numLabels, binomial(numLabels, sizeOfSubset));
        }
        classIndicesPerSubset = new int[numOfModels][sizeOfSubset];
        absoluteIndicesToRemove = new int[numOfModels][sizeOfSubset];
        subsetClassifiers = new MultiLabelLearner[numOfModels];
        remove = new Remove[numOfModels];

        for (int i = 0; i < numOfModels; i++) {
            updateClassifier(trainingData, i);
        }
    }

    private void updateClassifier(MultiLabelInstances mlTrainData, int model) throws Exception {
        //todo: check if the following is unnecessary (was used for cvparam)
        if (combinations == null) {
            combinations = new HashSet<String>();
        }

        Instances trainData = mlTrainData.getDataSet();
        // select a random subset of classes not seen before
        // todo: select according to inverse distribution of current selection
        boolean[] selected;
        do {
            selected = new boolean[numLabels];
            for (int j = 0; j < sizeOfSubset; j++) {
                int randomLabel;
                randomLabel = rnd.nextInt(numLabels);
                while (selected[randomLabel] != false) {
                    randomLabel = rnd.nextInt(numLabels);
                }
                selected[randomLabel] = true;
                //System.out.println("label: " + randomLabel);
                classIndicesPerSubset[model][j] = randomLabel;
            }
            Arrays.sort(classIndicesPerSubset[model]);
        } while (combinations.add(Arrays.toString(classIndicesPerSubset[model])) == false);
        debug("Building model " + (model + 1) + "/" + numOfModels + ", subset: " + Arrays.toString(classIndicesPerSubset[model]));
        // remove the unselected labels
        absoluteIndicesToRemove[model] = new int[numLabels - sizeOfSubset];
        int k = 0;
        for (int j = 0; j < numLabels; j++) {
            if (selected[j] == false) {
                absoluteIndicesToRemove[model][k] = labelIndices[j];
                k++;
            }
        }
        remove[model] = new Remove();
        remove[model].setAttributeIndicesArray(absoluteIndicesToRemove[model]);
        remove[model].setInputFormat(trainData);
        remove[model].setInvertSelection(false);
        Instances trainSubset = Filter.useFilter(trainData, remove[model]);

        // build a MultiLabelLearner for the selected label subset;
        subsetClassifiers[model] = getBaseLearner().makeCopy();
        subsetClassifiers[model].build(mlTrainData.reintegrateModifiedDataSet(trainSubset));
    }

    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        double[] sumConf = new double[numLabels];
        sumVotes = new double[numLabels];
        lengthVotes = new double[numLabels];

        // gather votes
        for (int i = 0; i < numOfModels; i++) {
            remove[i].input(instance);
            remove[i].batchFinished();
            Instance newInstance = remove[i].output();
            MultiLabelOutput subsetMLO = subsetClassifiers[i].makePrediction(newInstance);
            for (int j = 0; j < sizeOfSubset; j++) {
                sumConf[classIndicesPerSubset[i][j]] += subsetMLO.getConfidences()[j];
                sumVotes[classIndicesPerSubset[i][j]] += subsetMLO.getBipartition()[j] ? 1 : 0;
                lengthVotes[classIndicesPerSubset[i][j]]++;
            }
        }

        double[] confidence1 = new double[numLabels];
        double[] confidence2 = new double[numLabels];
        boolean[] bipartition = new boolean[numLabels];
        for (int i = 0; i < numLabels; i++) {
            if (lengthVotes[i] != 0) {
                confidence1[i] = sumVotes[i] / lengthVotes[i];
                confidence2[i] = sumConf[i] / lengthVotes[i];
            } else {
                confidence1[i] = 0;
                confidence2[i] = 0;
            }
            if (confidence1[i] >= threshold) {
                bipartition[i] = true;
            } else {
                bipartition[i] = false;
            }
        }

        // todo: optionally use confidence2 for ranking measures
        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidence1);
        return mlo;
    }

    /**
     * Returns a string describing classifier
     *
     * @return a description suitable for displaying
     */
    public String globalInfo() {
        return "Class implementing a generalized version of the RAkEL "
                + "(RAndom k-labELsets) algorithm. For more information, see\n\n"
                + getTechnicalInformation().toString();
    }
}