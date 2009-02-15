package mulan.classifier;


import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Class that implements a binary relevance classifier <p>
 *
 * @author Robert Friberg
 * @author Grigorios Tsoumakas 
 * @version $Revision: 0.04 $
 */
public class BinaryRelevance extends TransformationBasedMultiLabelLearner implements MultiLabelLearner
{

	protected Instances[] metadataTest;
	protected Classifier[] ensemble;

	public BinaryRelevance(Classifier classifier, int numLabels)
			throws Exception
	{
		super(classifier,numLabels);
		metadataTest = new Instances[numLabels];
		debug("BR: making classifier copies");
		ensemble = Classifier.makeCopies(classifier, numLabels);
	}


	public void build(Instances train) throws Exception
	{
		debug("BR: calling super constructor");
		
		for (int i = 0; i < numLabels; i++)
		{
			debug("BR: transforming training set for label " + i);
			Instances subTrain = transform(train, i);
			debug("BR: building base classifier for label " + i);
			ensemble[i].buildClassifier(subTrain);
			subTrain.delete();
			metadataTest[i] = subTrain;
		}
	}

	private Instance transformInstance(Instance instance, int label)
	{
		Instance newInstance = new Instance(instance.numAttributes());
		newInstance = (Instance) instance.copy();
		newInstance.setDataset(null);
		int numPredictors = instance.numAttributes() - numLabels;
		int skipLabel = 0;
		for (int i = 0; i < numLabels; i++)
		{
			if (i == label)
			{
				skipLabel++;
				continue;
			}
			newInstance.deleteAttributeAt(numPredictors + skipLabel);
		}
		return newInstance;
	}

	/**
	 * Remove all label attributes except label i
	 */
	private Instances transform(Instances train, int i) throws Exception
	{
		// Indices of attributes to remove
		int indices[] = new int[numLabels - 1];

		int k = 0;
		for (int j = 0; j < numLabels; j++)
			if (j != i)
			{
				indices[k] = train.numAttributes() - numLabels + j;
				k++;
			}

		Remove remove = new Remove();
		remove.setAttributeIndicesArray(indices);
		remove.setInputFormat(train);
		remove.setInvertSelection(true);
		Instances result = Filter.useFilter(train, remove);
		result.setClassIndex(result.numAttributes() - 1);
		return result;
	}

    public BipartitionAndRanking predictAndRank(Instance instance) {
        Boolean[] predictions = new Boolean[numLabels];
		double[] confidences = new double[numLabels];

		for (int i=0; i<numLabels; i++)
		{
			Instance newInstance = transformInstance(instance, i);
			newInstance.setDataset(metadataTest[i]);

            double distribution[] = new double[2];
            try {
                distribution = ensemble[i].distributionForInstance(newInstance);
            } catch (Exception e) {
                System.out.println(e);
                return null;
            }
            int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

			// Ensure correct predictions both for class values {0,1} and {1,0}
			Attribute classAttribute = metadataTest[i].classAttribute();
			predictions[i] = (classAttribute.value(maxIndex).equals("1")) ? true : false;

			// The confidence of the label being equal to 1
			confidences[i] = distribution[classAttribute.indexOfValue("1")];
		}
        Ranking ranking = new Ranking(confidences);
        Bipartition bipartition = new Bipartition(predictions);
        BipartitionAndRanking result = new BipartitionAndRanking(bipartition, ranking);
		return result;
    }

    public MultiLabelOutput makePrediction(Instance instance) throws Exception {
        boolean[] bipartition = new boolean[numLabels];
		double[] confidences = new double[numLabels];

		for (int i=0; i<numLabels; i++)
		{
			Instance newInstance = transformInstance(instance, i);
			newInstance.setDataset(metadataTest[i]);

            double distribution[] = new double[2];
            try {
                distribution = ensemble[i].distributionForInstance(newInstance);
            } catch (Exception e) {
                System.out.println(e);
                return null;
            }
            int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

			// Ensure correct predictions both for class values {0,1} and {1,0}
			Attribute classAttribute = metadataTest[i].classAttribute();
			bipartition[i] = (classAttribute.value(maxIndex).equals("1")) ? true : false;

			// The confidence of the label being equal to 1
			confidences[i] = distribution[classAttribute.indexOfValue("1")];
		}

        MultiLabelOutput mlo = new MultiLabelOutput();
        mlo.setBipartition(bipartition);
        mlo.setConfidencesAndRanking(confidences);
		return mlo;
    }
          
}

