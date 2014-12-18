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
package mulan.classifier.transformation;

import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.transformations.BinaryRelevanceTransformation;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * <p>Algorithm that builds one binary model per label.</p>
 *
 * @author Robert Friberg
 * @author Grigorios Tsoumakas
 * @version 2012.03.14
 */
public class BinaryRelevance extends TransformationBasedMultiLabelLearner {

    /**
     * The ensemble of binary relevance models. These are Weka Classifier
     * objects.
     */
    protected Classifier[] ensemble;
    /**
     * The correspondence between ensemble models and labels
     */
    private String[] correspondence;
    private BinaryRelevanceTransformation brt;

    /**
     * Creates a new instance
     *
     * @param classifier the base-level classification algorithm that will be
     * used for training each of the binary models
     */
    public BinaryRelevance(Classifier classifier) {
        super(classifier);
    }

    protected void buildInternal(MultiLabelInstances train) throws Exception {
        ensemble = new Classifier[numLabels];

        correspondence = new String[numLabels];
        for (int i = 0; i < numLabels; i++) {
            correspondence[i] = train.getDataSet().attribute(labelIndices[i]).name();
        }

        debug("preparing shell");
        brt = new BinaryRelevanceTransformation(train);

        for (int i = 0; i < numLabels; i++) {
            ensemble[i] = AbstractClassifier.makeCopy(baseClassifier);
            Instances shell = brt.transformInstances(i);
            debug("Bulding model " + (i + 1) + "/" + numLabels);
            ensemble[i].buildClassifier(shell);
        }
    }

    protected MultiLabelOutput makePredictionInternal(Instance instance) {
        boolean[] bipartition = new boolean[numLabels];
        double[] confidences = new double[numLabels];


        for (int counter = 0; counter < numLabels; counter++) {
            Instance transformedInstance = brt.transformInstance(instance, counter);
            double distribution[];
            try {
                distribution = ensemble[counter].distributionForInstance(transformedInstance);
            } catch (Exception e) {
                System.out.println(e);
                return null;
            }
            int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

            // Ensure correct predictions both for class values {0,1} and {1,0}
            bipartition[counter] = (maxIndex == 1) ? true : false;

            // The confidence of the label being equal to 1
            confidences[counter] = distribution[1];
        }

        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
        return mlo;
    }

    /**
     * Returns the model which corresponds to the label with labelName
     *
     * @param labelName The label name of the model to be returned
     * @return the corresponding model or null if the labelIndex is wrong
     */
    public Classifier getModel(String labelName) {
        for (int i = 0; i < numLabels; i++) {
            if (correspondence[i].equals(labelName)) {
                return ensemble[i];
            }
        }
        return null;
    }
}