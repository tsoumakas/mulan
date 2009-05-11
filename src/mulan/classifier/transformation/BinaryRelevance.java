package mulan.classifier.transformation;


import mulan.classifier.MultiLabelOutput;
import mulan.core.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Class that implements a binary relevance classifier <p>
 *
 * @author Robert Friberg
 * @author Grigorios Tsoumakas 
 * @version $Revision: 0.05 $
 */
public class BinaryRelevance extends TransformationBasedMultiLabelLearner 
{
	protected FilteredClassifier[] ensemble;
    
	public BinaryRelevance(Classifier classifier) throws Exception
    {
		super(classifier);
	}

	protected void buildInternal(MultiLabelInstances train) throws Exception
	{
        numLabels = train.getNumLabels();
		ensemble = new FilteredClassifier[numLabels];
        Instances trainingData = train.getDataSet();
        for (int i=0; i<numLabels; i++) {
            ensemble[i] = new FilteredClassifier();
            ensemble[i].setClassifier(Classifier.makeCopy(baseClassifier));

            // Indices of attributes to remove
            int[] indicesToRemove = new int[numLabels-1];
            int counter2=0;
            for (int counter1=0; counter1<numLabels; counter1++)
                if (labelIndices[counter1] != labelIndices[i])
                {
                    indicesToRemove[counter2] = labelIndices[counter1];
                    counter2++;
                }

            Remove remove = new Remove();
            remove.setAttributeIndicesArray(indicesToRemove);
            remove.setInputFormat(trainingData);
            remove.setInvertSelection(false);
            ensemble[i].setFilter(remove);

            trainingData.setClassIndex(labelIndices[i]);
            ensemble[i].buildClassifier(trainingData);
        }
	}

    public MultiLabelOutput makePrediction(Instance instance) throws Exception
    {
        boolean[] bipartition = new boolean[numLabels];
		double[] confidences = new double[numLabels];

		for (int counter=0; counter<numLabels; counter++)
		{
            double distribution[] = new double[2];
            try {
                distribution = ensemble[counter].distributionForInstance(instance);
            } catch (Exception e) {
                System.out.println(e);
                return null;
            }
            int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

			// Ensure correct predictions both for class values {0,1} and {1,0}
			Attribute classAttribute = ensemble[counter].getFilter().getOutputFormat().classAttribute();
			bipartition[counter] = (classAttribute.value(maxIndex).equals("1")) ? true : false;

			// The confidence of the label being equal to 1
			confidences[counter] = distribution[classAttribute.indexOfValue("1")];
		}

        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
		return mlo;
    }
          
}

