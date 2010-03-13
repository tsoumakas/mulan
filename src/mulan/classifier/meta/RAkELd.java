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
 *    RAkELd.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.classifier.meta;

//[R_d
import java.util.ArrayList;
import java.util.Collections;
//R_d]
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
 * Class implementing a generalized version of the RAkEL_d (RAndom k Disjoint labELsets) algorithm.
 * </pre>
 *
 * For more information:
 *
 * <pre>
 * Publication Info
 * 
 * </pre>
 *
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 *
 * <pre>
 * bibtex info
 * </pre>
 *
 * <p/> <!-- technical-bibtex-end -->
 *
 * @author 
 * @version $Revision: 0.04 $
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
    protected Remove[] remove;

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
        result.setValue(Field.AUTHOR, "");
        result.setValue(Field.TITLE, "");
        result.setValue(Field.JOURNAL, "");
        result.setValue(Field.PAGES, "");
        result.setValue(Field.VOLUME, "");
        result.setValue(Field.NUMBER, "");
        result.setValue(Field.YEAR, "");

        return result;
    }

    public RAkELd() throws Exception {
        rnd = new Random();
    }

    public RAkELd(MultiLabelLearner baseLearner) {
        super(baseLearner);
        rnd = new Random();
    }

    public RAkELd(MultiLabelLearner baseLearner, int subset) {
        super(baseLearner);
        rnd = new Random();
        sizeOfSubset = subset;
        //Todo: Check if subset <= numLabels, if not throw exception
    }

    public void setSeed(int x) {
        seed = x;
        rnd = new Random(seed);
    }

    public void setSizeOfSubset(int size) {
        sizeOfSubset = size;
    }

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
            classIndicesPerSubset_d[i] = new ArrayList<Integer>();
        }

        //<new way>
        absoluteIndicesToRemove = new ArrayList[numOfModels]; //This could be a local variable
        for (int i = 0; i < numOfModels; i++) {
            absoluteIndicesToRemove[i] = new ArrayList<Integer>();
        }
        //</new way>
        //R_d]

        subsetClassifiers = new MultiLabelLearner[numOfModels];
        remove = new Remove[numOfModels];

        //[R_d
        listOfLabels = new ArrayList<Integer>();
        for (int c = 0; c < numLabels; c++) {
            listOfLabels.add(c); //add all labels _(relative)_ indices to an arraylist
        }        //R_d]

        for (int i = 0; i < numOfModels; i++) {
            updateClassifier(trainingData, i);
        }
    }

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
}


       


