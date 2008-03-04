package mulan.classifier;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.core.Utils;

@SuppressWarnings("serial")
/**
 * Class that implements a binary relevance classifier <p>
 *
 * @author Robert Friberg
 * @author Grigorios Tsoumakas 
 * @version $Revision: 0.02 $ 
 */
public class BinaryRelevanceClassifier extends AbstractMultiLabelClassifier
{

	protected Instances[] metadataTest;
	protected Classifier[] ensemble;

	public BinaryRelevanceClassifier(Classifier classifier, int numLabels)
			throws Exception
	{
		setNumLabels(numLabels);
		dbg("BR: making classifier copies");
		ensemble = makeCopies(classifier, numLabels);
	}

	public void setNumLabels(int numLabels)
	{
		super.setNumLabels(numLabels);
		metadataTest = new Instances[numLabels];

	}

	public BinaryRelevanceClassifier()
	{
	}

	public void buildClassifier(Instances train) throws Exception
	{
		dbg("BR: calling super constructor");
		super.buildClassifier(train);
		
		// Added to support zero argument constructor
		if (ensemble == null)
		{
			dbg("BR: making classifier copies");
			ensemble = makeCopies(getBaseClassifier(), numLabels);
		}
			

		for (int i = 0; i < numLabels; i++)
		{
			dbg("BR: transforming training set for label " + i);
			Instances subTrain = transform(train, i);
			dbg("BR: building base classifier for label " + i);
			ensemble[i].buildClassifier(subTrain);
			subTrain.delete();
			metadataTest[i] = subTrain;
		}
	}

	private Instance transformInstance(Instance instance, int label)
			throws Exception
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

	protected Prediction makePrediction(Instance instance) throws Exception
	{
                double predictions[] = new double[numLabels];
		double confidences[] = new double[numLabels];

		for (int i = 0; i < numLabels; i++)
		{
			Instance newInstance = transformInstance(instance, i);			
			newInstance.setDataset(metadataTest[i]);

			double[] distribution = ensemble[i]
					.distributionForInstance(newInstance);
			int maxIndex = Utils.maxIndex(distribution);

			// Ensure correct predictions both for class values {0,1} and {1,0}
			Attribute classAttribute = metadataTest[i].classAttribute();				
			predictions[i] = Double.parseDouble(classAttribute.value(maxIndex));

			// The confidence of the label being equal to 1
			confidences[i] = distribution[classAttribute.indexOfValue("1")];
		}
		Prediction result = new Prediction(predictions, confidences);
		return result;
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
}

