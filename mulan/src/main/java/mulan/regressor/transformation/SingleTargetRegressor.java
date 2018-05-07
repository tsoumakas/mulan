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

import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

/**
 * This class implements the baseline Single-Target (ST) method for multi-target regression that
 * learns a separate regression model for each target.<br>
 * <br>
 * For more information, see:<br>
 * <em>E. Spyromitros-Xioufis, G. Tsoumakas, W. Groves, I. Vlahavas. 2014. Multi-label Classification Methods for
 * Multi-target Regression. <a href="http://arxiv.org/abs/1211.6581">arXiv e-prints</a></em>.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.04.01
 */
public class SingleTargetRegressor extends TransformationBasedMultiTargetRegressor {

    private static final long serialVersionUID = 1L;
    /**
     * The ensemble of ST models. These are Weka FilteredClassifier objects, where the filter
     * corresponds to removing all targets apart from the one that serves as a target for the
     * corresponding model.
     */
    protected FilteredClassifier[] stRegressors;

    /**
     * Constructor.
     * 
     * @param regressor the base regression algorithm that will be used
     */
    public SingleTargetRegressor(Classifier regressor) {
        super(regressor);
    }

    protected void buildInternal(MultiLabelInstances mlTrainSet) throws Exception {
        stRegressors = new FilteredClassifier[numLabels];
        // any changes are applied to a copy of the original dataset
        Instances trainSet = new Instances(mlTrainSet.getDataSet());
        for (int i = 0; i < numLabels; i++) {
            stRegressors[i] = new FilteredClassifier();
            stRegressors[i].setClassifier(AbstractClassifier.makeCopy(baseRegressor));

            // Indices of attributes to remove. All labelIndices except for the current index
            int[] indicesToRemove = new int[numLabels - 1];
            int counter2 = 0;
            for (int counter1 = 0; counter1 < numLabels; counter1++) {
                if (labelIndices[counter1] != labelIndices[i]) {
                    indicesToRemove[counter2] = labelIndices[counter1];
                    counter2++;
                }
            }

            Remove remove = new Remove();
            remove.setAttributeIndicesArray(indicesToRemove);
            remove.setInvertSelection(false);
            remove.setInputFormat(trainSet);
            stRegressors[i].setFilter(remove);

            trainSet.setClassIndex(labelIndices[i]);
            debug("Bulding model " + (i + 1) + "/" + numLabels);
            stRegressors[i].buildClassifier(trainSet);
        }
    }

    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        double[] scores = new double[numLabels];
        Instances dataset = instance.dataset();

        for (int counter = 0; counter < numLabels; counter++) {
            dataset.setClassIndex(labelIndices[counter]);
            instance.setDataset(dataset);
            scores[counter] = stRegressors[counter].classifyInstance(instance);
        }

        MultiLabelOutput mlo = new MultiLabelOutput(scores, true);

        return mlo;
    }

    @Override
    public String getModelForTarget(int target) {
        try {
            stRegressors[target].getClassifier().getClass()
                    .getMethod("toString", (Class<?>[]) null);
        } catch (NoSuchMethodException e) {
            return "A string representation for this base algorithm is not provided!";
        }
        return stRegressors[target].getClassifier().toString();
    }

}
