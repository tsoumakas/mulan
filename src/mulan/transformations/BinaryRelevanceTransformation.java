/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mulan.transformations;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author greg
 */
public class BinaryRelevanceTransformation {
	int numOfLabels;

    public BinaryRelevanceTransformation(int num) {
        numOfLabels = num;
    }

	/**
	 * Remove all label attributes except labelToKeep
	 */
    public Instance transformInstance(Instance instance, int labelToKeep)
	{
		Instance newInstance = new Instance(instance.numAttributes());
		newInstance = (Instance) instance.copy();
		newInstance.setDataset(null);
		int numPredictors = instance.numAttributes() - numOfLabels;
		int skipLabel = 0;
		for (int labelIndex=0; labelIndex<numOfLabels; labelIndex++)
		{
			if (labelIndex == labelToKeep)
			{
				skipLabel++;
				continue;
			}
			newInstance.deleteAttributeAt(numPredictors + skipLabel);
		}
		return newInstance;
	}

	/**
	 * Remove all label attributes except labelToKeep
	 */
	public Instances transformInstances(Instances train, int labelToKeep) throws Exception
	{
		// Indices of attributes to remove
		int indices[] = new int[numOfLabels-1];

		int k=0;
		for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++)
			if (labelIndex != labelToKeep)
			{
				indices[k] = train.numAttributes() - numOfLabels + labelIndex;
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
