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

/*
 *    RAkEL.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.classifier.meta;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * <!-- globalinfo-start -->
 *
 * <pre>
 * Class implementing a generalized version of the RAkEL (RAndom k-labELsets) algorithm.
 * </pre>
 *
 * For more information:
 *
 * <pre>
 * Tsoumakas, G, Vlahavas, I. (2007) Random k-Labelsets: An Ensemble Method
 * for Multilabel Classification", Proc. 18th European Conference on Machine
 * Learning (ECML 2007), pp. 406-417, Warsaw, Poland, 17-21 September 2007.
 * </pre>
 *
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 *
 * <pre>
 * &#064;inproceedings{tsoumakas+vlahavas:2007,
 *    author =    {Tsoumakas, G. and Vlahavas, I.},
 *    title =     {Random k-Labelsets: An Ensemble Method for Multilabel Classification},
 *    booktitle = {Proceedings of the 18th European Conference on Machine Learning (ECML 2007)},
 *    year =      {2007},
 *    pages =     {406--417},
 *    address =   {Warsaw, Poland},
 *    month =     {September 17-21},
 * }
 * </pre>
 *
 * <p/> <!-- technical-bibtex-end -->
 *
 * @author Grigorios Tsoumakas
 * @version $Revision: 0.04 $
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
    double[][] sumVotesIncremental; /* comment */

    double[][] lengthVotesIncremental;
    double[] sumVotes;
    double[] lengthVotes;
    int numOfModels;
    double threshold = 0.5;
    int sizeOfSubset = 3;
    int[][] classIndicesPerSubset;
    int[][] absoluteIndicesToRemove;
    MultiLabelLearner[] subsetClassifiers;
    protected Remove[] remove;
    HashSet<String> combinations;

    /**
     * Returns an instance of a TechnicalInformation object, containing
     * detailed information about the technical background of this class,
     * e.g., paper reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "Grigorios Tsoumakas, Ioannis Vlahavas");
        result.setValue(Field.TITLE, "Random k-Labelsets: An Ensemble Method for Multilabel Classification");
        result.setValue(Field.BOOKTITLE, "Proc. 18th European Conference on Machine Learning (ECML 2007)");
        result.setValue(Field.PAGES, "406 - 417");
        result.setValue(Field.LOCATION, "Warsaw, Poland");
        result.setValue(Field.MONTH, "17-21 September");
        result.setValue(Field.YEAR, "2007");

        return result;
    }

    public RAkEL(int models, int subset) throws Exception {
        sizeOfSubset = subset;
        setNumModels(models);
    }

    public RAkEL(MultiLabelLearner baseLearner) {
        super(baseLearner);
    }

    public RAkEL(MultiLabelLearner baseLearner, int models, int subset) {
        super(baseLearner);
        sizeOfSubset = subset;
        setNumModels(models);
    }

    public RAkEL(MultiLabelLearner baseLearner, int models, int subset, double threshold) {
        super(baseLearner);
        sizeOfSubset = subset;
        setNumModels(models);
        this.threshold = threshold;
    }

    public void setSeed(int x) {
        seed = x;
    }

    public void setSizeOfSubset(int size) {
        sizeOfSubset = size;
        classIndicesPerSubset = new int[numOfModels][sizeOfSubset];
    }

    public int getSizeOfSubset() {
        return sizeOfSubset;
    }

    public void setNumModels(int models) {
        numOfModels = models;
    }

    public int getNumModels() {
        return numOfModels;
    }

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
}
