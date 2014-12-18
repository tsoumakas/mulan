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
package mulan.regressor.transformation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

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
 * This class implements the Regressor Chain (RC) method.<br>
 * <br>
 * For more information, see:<br>
 * <em>E. Spyromitros-Xioufis, G. Tsoumakas, W. Groves, I. Vlahavas. 2014. Multi-label Classification Methods for
 * Multi-target Regression. <a href="http://arxiv.org/abs/1211.6581">arXiv e-prints</a></em>.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.04.01
 */
public class RegressorChainSimple extends TransformationBasedMultiTargetRegressor {

    private static final long serialVersionUID = 1L;

    /**
     * The seed to use for random number generation in order to create a random chain (other than
     * the default one which consists of the targets chained in the order they appear in the arff
     * file).
     */
    private int chainSeed = 0;

    /**
     * A permutation of the target indices. E.g. If there are 3 targets with indices 14,15 and 16, a
     * valid chain is 15,14,16.
     */
    private int[] chain;

    /**
     * The regressors of the chain.
     */
    protected FilteredClassifier[] chainRegressors;

    /**
     * Creates a new instance with the given base regressor. If {@link #chainSeed} == 0, the default
     * chain is used. Otherwise, a random chain is created using the given seed.
     * 
     * @param regressor the base regression algorithm that will be used
     */
    public RegressorChainSimple(Classifier regressor) {
        super(regressor);
    }

    /**
     * Creates a new instance with the given base regressor and chain ordering.
     * 
     * @param regressor the base regression algorithm that will be used
     * @param aChain a chain ordering
     */
    public RegressorChainSimple(Classifier regressor, int[] aChain) {
        super(regressor);
        chain = aChain;
    }

    protected void buildInternal(MultiLabelInstances train) throws Exception {
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

        chainRegressors = new FilteredClassifier[numLabels];
        Instances trainDataset = train.getDataSet();

        for (int i = 0; i < numLabels; i++) {
            chainRegressors[i] = new FilteredClassifier();
            chainRegressors[i].setClassifier(AbstractClassifier.makeCopy(baseRegressor));

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
            chainRegressors[i].setFilter(remove);

            trainDataset.setClassIndex(chain[i]);
            debug("Bulding model " + (i + 1) + "/" + numLabels);
            chainRegressors[i].setDebug(true);
            chainRegressors[i].buildClassifier(trainDataset);
        }
    }

    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        double[] scores = new double[numLabels];

        // create a new temporary instance so that the passed instance is not altered
        Instances dataset = instance.dataset();
        Instance tempInstance = DataUtils.createInstance(instance, instance.weight(),
                instance.toDoubleArray());

        for (int counter = 0; counter < numLabels; counter++) {
            dataset.setClassIndex(chain[counter]);
            tempInstance.setDataset(dataset);
            // find the appropriate position for that score in the scores array
            // i.e. which is the corresponding target
            int pos = 0;
            for (int i = 0; i < numLabels; i++) {
                if (chain[counter] == labelIndices[i]) {
                    pos = i;
                    break;
                }
            }
            scores[pos] = chainRegressors[counter].classifyInstance(tempInstance);
            tempInstance.setValue(chain[counter], scores[pos]);
        }

        MultiLabelOutput mlo = new MultiLabelOutput(scores, true);
        return mlo;
    }

    @Override
    protected String getModelForTarget(int targetIndex) {
        try {
            chainRegressors[targetIndex].getClassifier().getClass()
                    .getMethod("toString", (Class<?>[]) null);
        } catch (NoSuchMethodException e) {
            return "A string representation for this base algorithm is not provided!";
        }
        return chainRegressors[targetIndex].toString();
    }

    public void setChainSeed(int chainSeed) {
        this.chainSeed = chainSeed;
    }

}
