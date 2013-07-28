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
 * This class implements the baseline regression approach that learns a separate regression model
 * for each target.<br/>
 * <br/>
 * For more information, see:<br/>
 * E. Spyromitros-Xioufis, W. Groves, G. Tsoumakas, I. Vlahavas (2012). Multi-label Classification
 * Methods for Multi-target Regression. <a href="http://arxiv.org/abs/1211.6581">ArXiv e-prints</a>.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2013.07.28
 */
public class SingleTargetRegressor extends TransformationBasedMultiTargetRegressor {

    private static final long serialVersionUID = 1L;
    /**
     * The ensemble of STRegression models. These are Weka FilteredClassifier objects, where the
     * filter corresponds to removing all targets apart from the one that serves as a target for the
     * corresponding model.
     */
    protected FilteredClassifier[] ensemble;

    /**
     * Creates a new instance
     * 
     * @param regressor the base-level regression algorithm that will be used for training each of
     *            the single-target models
     */
    public SingleTargetRegressor(Classifier regressor) {
        super(regressor);
    }

    protected void buildInternal(MultiLabelInstances train) throws Exception {
        ensemble = new FilteredClassifier[numLabels];
        Instances trainingData = train.getDataSet();
        for (int i = 0; i < numLabels; i++) {
            ensemble[i] = new FilteredClassifier();
            ensemble[i].setClassifier(AbstractClassifier.makeCopy(baseRegressor));

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
            remove.setInputFormat(trainingData);
            ensemble[i].setFilter(remove);

            trainingData.setClassIndex(labelIndices[i]);
            debug("Bulding model " + (i + 1) + "/" + numLabels);
            ensemble[i].buildClassifier(trainingData);
        }
    }

    /**
     * Makes a prediction by calling all the ST models iteratively.
     */
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        double[] scores = new double[numLabels];
        Instances dataset = instance.dataset();

        for (int counter = 0; counter < numLabels; counter++) {
            dataset.setClassIndex(labelIndices[counter]);
            instance.setDataset(dataset);
            scores[counter] = ensemble[counter].classifyInstance(instance);
        }

        MultiLabelOutput mlo = new MultiLabelOutput(scores, true);

        return mlo;
    }

    @Override
    public String getModelForTarget(int target) {
        Classifier model = ensemble[target].getClassifier();
        try {
            model.getClass().getMethod("toString", (Class<?>[]) null);
        } catch (NoSuchMethodException e) {
            return "A string representation for this base algorithm is not provided!";
        }
        return model.toString();
    }

}
