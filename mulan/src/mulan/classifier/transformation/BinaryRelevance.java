package mulan.classifier.transformation;


import mulan.classifier.MultiLabelOutput;
import mulan.core.data.MultiLabelInstances;
import mulan.transformations.BinaryRelevanceTransformation;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class that implements a binary relevance classifier <p>
 *
 * @author Robert Friberg
 * @author Grigorios Tsoumakas 
 * @version $Revision: 0.04 $
 */
public class BinaryRelevance extends TransformationBasedMultiLabelLearner 
{
	protected Instances[] metadataTest;
	protected Classifier[] ensemble;
    protected BinaryRelevanceTransformation transformation;

	public BinaryRelevance(Classifier classifier)
			throws Exception
	{
		super(classifier);
        transformation = new BinaryRelevanceTransformation(numLabels);
		metadataTest = new Instances[numLabels];
		debug("BR: making classifier copies");
		ensemble = Classifier.makeCopies(classifier, numLabels);
	}


	protected void buildInternal(MultiLabelInstances train) throws Exception
	{
		Instances dataSet = train.getDataSet();  
		for (int labelIndex=0; labelIndex<numLabels; labelIndex++)
		{
			debug("BR: transforming training set for label " + labelIndex);
			Instances subTrain = transformation.transformInstances(dataSet, labelIndex);
			debug("BR: building base classifier for label " + labelIndex);
			ensemble[labelIndex].buildClassifier(subTrain);
			subTrain.delete();
			metadataTest[labelIndex] = subTrain;
		}
	}

    public MultiLabelOutput makePrediction(Instance instance) throws Exception {
        boolean[] bipartition = new boolean[numLabels];
		double[] confidences = new double[numLabels];

		for (int labelIndex=0; labelIndex<numLabels; labelIndex++)
		{
			Instance newInstance = transformation.transformInstance(instance, labelIndex);
			newInstance.setDataset(metadataTest[labelIndex]);

            double distribution[] = new double[2];
            try {
                distribution = ensemble[labelIndex].distributionForInstance(newInstance);
            } catch (Exception e) {
                System.out.println(e);
                return null;
            }
            int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

			// Ensure correct predictions both for class values {0,1} and {1,0}
			Attribute classAttribute = metadataTest[labelIndex].classAttribute();
			bipartition[labelIndex] = (classAttribute.value(maxIndex).equals("1")) ? true : false;

			// The confidence of the label being equal to 1
			confidences[labelIndex] = distribution[classAttribute.indexOfValue("1")];
		}

        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
		return mlo;
    }
          
}

