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

import java.util.ArrayList;
import java.util.Collections;
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
 *
 <!-- globalinfo-start -->
 * Class implementing a generalized version of the RAkEL-d (RAndom k-labELsets) algorithm with disjoint labelsets. For more information, see<br>
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
 * @author Ioannis Katakis
 * @author Grigorios Tsoumakas
 * @version 2012.07.16
 */
@SuppressWarnings("serial")
public class RAkELd extends MultiLabelMetaLearner {

    /**
     * Seed for replication of random experiments
     */
    private int seed = 0;
    /**
     * Random number generator
     */
    private Random rnd;
    int numOfModels;
    int sizeOfSubset = 3; //TODO: If numLabels<=3 then ...
    //[R_d
    ArrayList<Integer>[] classIndicesPerSubset_d;
    ArrayList<Integer>[] absoluteIndicesToRemove;
    ArrayList<Integer> listOfLabels;
    //R_d]
    MultiLabelLearner[] subsetClassifiers;
    private Remove[] remove;

    /**
     * Returns an instance of a TechnicalInformation object, containing
     * detailed information about the technical background of this class,
     * e.g., paper reference or book this class is based on.
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
    public RAkELd() {
        this(new BinaryRelevance(new J48()));
    }

    /**
     * Construct a new instance based on the given multi-label learner
     * 
     * @param baseLearner a multi-label learner
     */
    public RAkELd(MultiLabelLearner baseLearner) {
        super(baseLearner);
        rnd = new Random();
    }

    /**
     * Constructs a new instance based on the given multi-label learner and 
     * size of subset
     * 
     * @param baseLearner the multi-label learner
     * @param subset the size of the subset
     */
    public RAkELd(MultiLabelLearner baseLearner, int subset) {
        super(baseLearner);
        rnd = new Random();
        sizeOfSubset = subset;
        //Todo: Check if subset <= numLabels, if not throw exception
    }

    /**
     * Sets the seed for random number generation
     * 
     * @param x the seed
     */
    public void setSeed(int x) {
        seed = x;
        rnd = new Random(seed);
    }

    /**
     * Sets the size of the subsets
     * @param size size of subsets
     */
    public void setSizeOfSubset(int size) {
        sizeOfSubset = size;
    }

    /**
     * Returns the size of the subsets
     * @return the size of the subsets
     */
    public int getSizeOfSubset() {
        return sizeOfSubset;
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingData) throws Exception {
        //[R_d
        if (numLabels % sizeOfSubset == 0 || numLabels % sizeOfSubset == 1) {
            numOfModels = numLabels / sizeOfSubset;
        } else {
            numOfModels = numLabels / sizeOfSubset + 1;
        }
        classIndicesPerSubset_d = new ArrayList[numOfModels];
        for (int i = 0; i < numOfModels; i++) {
            classIndicesPerSubset_d[i] = new ArrayList<>();
        }

        //<new way>
        absoluteIndicesToRemove = new ArrayList[numOfModels]; //This could be a local variable
        for (int i = 0; i < numOfModels; i++) {
            absoluteIndicesToRemove[i] = new ArrayList<>();
        }
        //</new way>
        //R_d]

        subsetClassifiers = new MultiLabelLearner[numOfModels];
        remove = new Remove[numOfModels];

        //[R_d
        listOfLabels = new ArrayList<>();
        for (int c = 0; c < numLabels; c++) {
            listOfLabels.add(c); //add all labels _(relative)_ indices to an arraylist
        }        //R_d]

        for (int i = 0; i < numOfModels; i++) {
            updateClassifier(trainingData, i);
        }
    }

    /**
     * Updates the current ensemble by training a specific classifier
     * 
     * @param mlTrainData the training data
     * @param model the model to train
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public void updateClassifier(MultiLabelInstances mlTrainData, int model) throws Exception {
        Instances trainData = mlTrainData.getDataSet();

        //[R_d]
        if (model == numOfModels - 1) {
            classIndicesPerSubset_d[model].addAll(listOfLabels);
        } else {
            int randomLabelIndex;  // select labels for model i
            for (int j = 0; j < sizeOfSubset; j++) {
                int randomLabel;
                randomLabelIndex = Math.abs(rnd.nextInt() % listOfLabels.size());
                randomLabel = listOfLabels.get(randomLabelIndex);
                listOfLabels.remove(randomLabelIndex); //remove selected labels from the list
                classIndicesPerSubset_d[model].add(randomLabel);
            }
        }
        //Probably not necessary but ensures that Rakel_d at subset=k=numLabels
        //will output the same results as LP
        Collections.sort(classIndicesPerSubset_d[model]);
        //[/R_d]

        debug("Building model " + (model + 1) + "/" + numOfModels + ", subset: " + classIndicesPerSubset_d[model].toString());

        // remove the unselected labels
        //<new way>
        for (int j = 0; j < numLabels; j++) {
            if (!classIndicesPerSubset_d[model].contains(j)) {
                absoluteIndicesToRemove[model].add(labelIndices[j]);
            }
        }

        int[] indicesRemoveArray = new int[absoluteIndicesToRemove[model].size()]; //copy into an array
        for (int j = 0; j < indicesRemoveArray.length; j++) {
            indicesRemoveArray[j] = absoluteIndicesToRemove[model].get(j);
        }
        remove[model] = new Remove();
        remove[model].setInvertSelection(false);
        remove[model].setAttributeIndicesArray(indicesRemoveArray);
        //</new Way>

        remove[model].setInputFormat(trainData);
        Instances trainSubset = Filter.useFilter(trainData, remove[model]);
        // build a MultiLabelLearner for the selected label subset;
        subsetClassifiers[model] = getBaseLearner().makeCopy();
        subsetClassifiers[model].build(mlTrainData.reintegrateModifiedDataSet(trainSubset));
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        double[] confidences = new double[numLabels];
        boolean[] labels = new boolean[numLabels];

        // gather votes
        for (int i = 0; i < numOfModels; i++) {
            remove[i].input(instance);
            remove[i].batchFinished();
            Instance newInstance = remove[i].output();

            MultiLabelOutput subsetMLO = subsetClassifiers[i].makePrediction(newInstance);

            boolean[] localPredictions = subsetMLO.getBipartition();
            double[] localConfidences = subsetMLO.getConfidences();

            for (int j = 0; j < classIndicesPerSubset_d[i].size(); j++) {
                labels[classIndicesPerSubset_d[i].get(j)] = localPredictions[j];
                confidences[classIndicesPerSubset_d[i].get(j)] = localConfidences[j];
            }
        }

        MultiLabelOutput mlo = new MultiLabelOutput(labels, confidences);
        return mlo;
    }

    /**
     * Returns a string describing classifier
     * @return a description suitable for displaying 
     */
    public String globalInfo() {

        return "Class implementing a generalized version of the RAkEL-d "
                + "(RAndom k-labELsets) algorithm with disjoint labelsets. "
                + "For more information, see\n\n"
                + getTechnicalInformation().toString();
    }

}