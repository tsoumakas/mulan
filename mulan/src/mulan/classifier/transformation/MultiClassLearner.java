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
import mulan.transformations.RemoveAllLabels;
import mulan.transformations.multiclass.MultiClassTransformation;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * <p>Class enabling a number of simple transformations of a multi-label task to
 * a multi-class task (labels become class values).</p>
 * <p>For more information,
 * see <ul><li><em>Boutell, M.; Luo, J.; Shen, X.; Brown, C. (2004) Learning
 * multi-label scene classification. Pattern Recognition 37:1757-1771</em></li>
 * <li><em>Chen, W.; Yan, J.; Zhang, B.; Chen, Z.; Yang, Q. (2007) Document
 * transformation for multi-label feature selection in text categorization. In:
 * Proc. 7th IEEE International Conference on Data Mining, Los Alamitos, CA,
 * USA, IEEE Computer Society, pp. 451-456</em></li></ul>
 *
 * @author Stavros Mpakirtzoglou
 * @author Grigorios Tsoumakas
 * @version 2012.02.27
 */
public class MultiClassLearner extends TransformationBasedMultiLabelLearner {

    private Instances header;
    private MultiClassTransformation transformation;

    /**
     * Initializes learner
     *
     * @param baseClassifier the base single-label classification algorithm
     * @param dt the {@link MultiClassTransformation} to use
     */
    public MultiClassLearner(Classifier baseClassifier, MultiClassTransformation dt) {
        super(baseClassifier);
        transformation = dt;
    }

    protected void buildInternal(MultiLabelInstances train) throws Exception {
        debug("Transforming the training set");
        Instances meta = transformation.transformInstances(train);
        baseClassifier.buildClassifier(meta);
        header = new Instances(meta, 0);
    }

    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        //delete labels
        instance = RemoveAllLabels.transformInstance(instance, labelIndices);
        instance.setDataset(null);
        instance.insertAttributeAt(instance.numAttributes());
        instance.setDataset(header);

        double[] distribution = baseClassifier.distributionForInstance(instance);

        MultiLabelOutput mlo = new MultiLabelOutput(MultiLabelOutput.ranksFromValues(distribution));
        return mlo;
    }
}