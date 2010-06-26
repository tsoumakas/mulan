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
 *    EnsembleOfClassifierChains.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.classifier.transformation;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 * 
 * <!-- globalinfo-start -->
 * <!-- globalinfo-end -->
 * 
 * <!-- technical-bibtex-start -->
 * <!-- technical-bibtex-end -->
 *
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 * @author Konstantinos Sechidis (sechidis@csd.auth.gr)
 */
public class EnsembleOfClassifierChains extends TransformationBasedMultiLabelLearner {

    /**
     * The aThreshold to use for obtaining bipartitions
     */
    protected double threshold;
    /**
     * The number of classifier chain models
     */
    protected int numOfModels;
    /**
     * An array of ClassifierChains models
     */
    protected ClassifierChains[] ensemble;
    /**
     * Random number generator
     */
    protected Random rand;

    /**
     * Creates a new object
     *
     * @param classifier the base classifier for each ClassifierChains model
     * @param aNumOfModels the number of models
     * @param aThreshold the aThreshold to obtain bipartitions
     */
    public EnsembleOfClassifierChains(Classifier classifier, int aNumOfModels, double aThreshold) {
        super(classifier);
        numOfModels = aNumOfModels;
        threshold = aThreshold;
        ensemble = new ClassifierChains[aNumOfModels];
        rand = new Random(1);
        for (int i = 0; i < numOfModels; i++) {
            ensemble[i] = new ClassifierChains(classifier);
        }
    }

    /**
     * Returns a string describing classifier.
     * @return a description suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {
        return "Class implementing the Classifier Chains for Multi-label Classification algorithm." + "\n\n" + "For more information, see\n\n" + getTechnicalInformation().toString();
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "Read, Jesse and Pfahringer, Bernhard and Holmes, Geoff and Frank, Eibe");
        result.setValue(Field.TITLE, "Classifier Chains for Multi-label Classification");
        result.setValue(Field.VOLUME, "Proceedings of ECML/PKDD 2009");
        result.setValue(Field.YEAR, "2009");
        result.setValue(Field.PAGES, "254--269");
        result.setValue(Field.ADDRESS, "Bled, Slovenia");

        return result;
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet)
            throws Exception {

        double percentage = 67;
        Instances dataSet = new Instances(trainingSet.getDataSet());

        for (int i = 0; i < numOfModels; i++) {
            debug("ECC Building Model:" + (i + 1) + "/" + numOfModels);
            dataSet.randomize(rand);
            RemovePercentage rmvp = new RemovePercentage();
            rmvp.setInvertSelection(true);
            rmvp.setPercentage(percentage);
            rmvp.setInputFormat(dataSet);
            Instances trainDataSet = Filter.useFilter(dataSet, rmvp);
            MultiLabelInstances train = new MultiLabelInstances(trainDataSet, trainingSet.getLabelsMetaData());

            int[] newLabelIndices = Arrays.copyOf(labelIndices, labelIndices.length);

            // Shuffle the elements in the array
            for (int j = 0; j < newLabelIndices.length; j++) {
                int randomPosition = rand.nextInt(newLabelIndices.length);
                int temp = newLabelIndices[j];
                newLabelIndices[j] = newLabelIndices[randomPosition];
                newLabelIndices[randomPosition] = temp;
            }
            System.out.println(Arrays.toString(newLabelIndices));


            // MAYBE WE SHOULD CHECK NOT TO PRODUCE THE SAME VECTOR FOR THE INDICES
            // BUT IN THE PAPER IT DID NOT MENTION SOMETHING LIKE THAT
            // IT JUST SIMPLE SAY A RANDOM CHAIN ORDERING OF L

            ensemble[i].setRandom(true);
            ensemble[i].setNewLabelIndices(newLabelIndices);
            ensemble[i].build(train);
        }

    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance)
            throws Exception, InvalidDataException {


        int[] sumVotes = new int[numLabels];

        for (int j = 0; j < numLabels; j++) {
            sumVotes[j] = 0;
        }
        for (int i = 0; i < numOfModels; i++) {
            MultiLabelOutput ensembleMLO = ensemble[i].makePrediction(instance);
            boolean[] bip = ensembleMLO.getBipartition();

            for (int j = 0; j < sumVotes.length; j++) {
                sumVotes[j] += bip[j] == true ? 1 : 0;
            }
        }
        double[] confidence = new double[numLabels];

        for (int j = 0; j < sumVotes.length; j++) {
            confidence[j] = (double) sumVotes[j] / (double) numOfModels;
        }

        MultiLabelOutput mlo = new MultiLabelOutput(confidence, threshold);
        return mlo;
    }
}
