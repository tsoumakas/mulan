package mulan.classifier.transformation;

import mulan.classifier.MultiLabelOutput;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import mulan.transformations.multiclass.MultiClassTransformation;
import mulan.transformations.RemoveAllLabels;

/**
 *
 * @author Stavros
 */
public class MultiClassLearner extends TransformationBasedMultiLabelLearner {

    private Instances header;
    private MultiClassTransformation transformation;

	public MultiClassLearner(Classifier baseClassifier, int numLabels, MultiClassTransformation dt)
	{
		super(baseClassifier, numLabels);
        transformation = dt;
    }

    @Override
    public void build(Instances train) throws Exception {
        Instances meta = transformation.transformInstances(train);
        baseClassifier.buildClassifier(meta);
        header = new Instances(meta, 0);
    }

    public MultiLabelOutput makePrediction(Instance instance) throws Exception {
        //delete labels
        RemoveAllLabels rmLabels = new RemoveAllLabels();
        instance = rmLabels.transformInstance(instance, numLabels);
        instance.insertAttributeAt(instance.numAttributes());
        instance.setDataset(header);

        double[] distribution = baseClassifier.distributionForInstance(instance);

        MultiLabelOutput mlo = new MultiLabelOutput(MultiLabelOutput.ranksFromValues(distribution));
		return mlo;
    }

}
