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
 *    RegressorChain.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.regressor.transformation;

import java.util.*;

import mulan.classifier.MultiLabelOutput;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Class implementing the Regressor Chain (RC) algorithm.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 11.27.2012
 */
public class RegressorChain extends TransformationBasedMultiTargetRegressor {

    private static final long serialVersionUID = 1L;

    /**
     * The seed to use for random number generation in order to create a random chain (other than
     * the default)
     */
    private int chainSeed = 0;

    public void setChainSeed(int chainSeed) {
        this.chainSeed = chainSeed;
    }

    /**
     * This array should contain a random permutation of the label indices.
     */
    private int[] chain;

    /**
     * The ensemble of regression chain models. These are Weka FilteredClassifier objects, where the
     * filter corresponds to removing all targets apart from the one that serves as a target for the
     * corresponding model.
     */
    protected FilteredClassifier[] ensemble;

    /**
     * Creates a new instance
     * 
     * @param classifier the base regression algorithm that will be used for training each model
     * @param aChain
     */
    public RegressorChain(Classifier classifier, int[] aChain) {
        super(classifier);
        chain = aChain;
    }

    /**
     * Creates a new instance
     * 
     * @param classifier the base-level classification algorithm that will be used for training each
     *            of the binary models
     */
    public RegressorChain(Classifier classifier) {
        super(classifier);
    }

    protected void buildInternal(MultiLabelInstances train) throws Exception {
        // if no chain has been defined, create the default chain
        if (chain == null) {
            chain = new int[numLabels];
            for (int j = 0; j < numLabels; j++) {
                chain[j] = labelIndices[j];
            }
        }

        if (chainSeed != 0) { // we should create a random chain
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
        debug(Arrays.toString(chain));

        ensemble = new FilteredClassifier[numLabels];
        Instances trainDataset = train.getDataSet();

        for (int i = 0; i < numLabels; i++) {
            ensemble[i] = new FilteredClassifier();
            ensemble[i].setClassifier(AbstractClassifier.makeCopy(baseRegressor));

            // Indices of attributes to remove.
            // First removes numLabels attributes, then numLabels - 1 attributes and so on.
            // The loop starts from the last attribute.
            int[] indicesToRemove = new int[numLabels - 1 - i];
            for (int counter1 = 0; counter1 < numLabels - i - 1; counter1++) {
                indicesToRemove[counter1] = chain[numLabels - 1 - counter1];
            }

            Remove remove = new Remove();
            remove.setAttributeIndicesArray(indicesToRemove);
            remove.setInvertSelection(false);
            remove.setInputFormat(trainDataset);
            ensemble[i].setFilter(remove);

            trainDataset.setClassIndex(chain[i]);
            debug("Bulding model " + (i + 1) + "/" + numLabels);
            ensemble[i].setDebug(true);
            ensemble[i].buildClassifier(trainDataset);
        }
    }

    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        double[] scores = new double[numLabels];

        // create a new temporary instance so that the passed instance is not altered
        Instances dataset = instance.dataset();
        Instance tempInstance = DataUtils.createInstance(instance, instance.weight(), instance
                .toDoubleArray());

        for (int counter = 0; counter < numLabels; counter++) {
            dataset.setClassIndex(chain[counter]);
            tempInstance.setDataset(dataset);

            if (Double.isNaN(ensemble[counter].classifyInstance(tempInstance))) {
                System.out.println("Some problem occured");
            }
            // find the appropriate position for that score in the scores array
            // i.e. which is the corresponding target
            int pos = 0;
            for (int i = 0; i < numLabels; i++) {
                if (chain[counter] == labelIndices[i]) {
                    pos = i;
                    break;
                }
            }
            scores[pos] = ensemble[counter].classifyInstance(tempInstance);
            tempInstance.setValue(chain[counter], scores[pos]);
        }

        MultiLabelOutput mlo = new MultiLabelOutput(scores, true);
        return mlo;
    }

    @Override
    protected String getModelForTarget(int target) {
        Classifier model = ensemble[target].getClassifier();
        try {
            model.getClass().getMethod("toString", (Class<?>[]) null);
        } catch (NoSuchMethodException e) {
            return "A string representation for this base algorithm is not provided!";
        }
        return model.toString();
    }

}
