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
import mulan.transformations.IncludeLabelsTransformation;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * A multilabel classifier based on the include labels transformation. The
 * multiple label attributes are mapped to two attributes: <ul> <li>a) one
 * nominal attribute containing the class</li> <li>b) one binary attribute
 * containing whether it is true.</li> </ul> 
 *
 * @author Robert Friberg
 * @author Grigorios Tsoumakas
 * @version 2012.02.27
 */
public class IncludeLabelsClassifier extends TransformationBasedMultiLabelLearner {

    /**
     * The transformation used by the classifier
     */
    private IncludeLabelsTransformation pt6Trans;
    /**
     * A dataset with the format needed by the base classifier. It is
     * potentially expensive copying datasets with many attributes, so it is
     * used for building the classifier and then it's mlData are discarded and
     * it is reused during prediction.
     */
    protected Instances transformed;

    /**
     * Constructor that initializes a new learner with the given base classifier
     *
     * @param classifier the base classifier used to initialize the learner
     */
    public IncludeLabelsClassifier(Classifier classifier) {
        super(classifier);
    }

    @Override
    public void buildInternal(MultiLabelInstances mlData) throws Exception {
        //Do the transformation
        //and generate the classifier
        pt6Trans = new IncludeLabelsTransformation();
        debug("Transforming the dataset");
        transformed = pt6Trans.transformInstances(mlData);
        debug("Building the base-level classifier");
        baseClassifier.buildClassifier(transformed);
        transformed.delete();
    }

    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        double[] confidences = new double[numLabels];
        boolean[] bipartition = new boolean[numLabels];

        Instance newInstance = pt6Trans.transformInstance(instance);
        //calculate confidences
        //debug(instance.toString());
        for (int i = 0; i < numLabels; i++) {
            newInstance.setDataset(transformed);
            newInstance.setValue(newInstance.numAttributes() - 2, instance.dataset().attribute(labelIndices[i]).name());
            //debug(newInstance.toString());
            double[] temp = baseClassifier.distributionForInstance(newInstance);
            //debug(temp.toString());
            confidences[i] = temp[transformed.classAttribute().indexOfValue("1")];
            //debug("" + confidences[i]);
            bipartition[i] = temp[transformed.classAttribute().indexOfValue("1")] >= temp[transformed.classAttribute().indexOfValue("0")] ? true : false;
            //debug("" + bipartition[i]);
        }

        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
        return mlo;
    }
}