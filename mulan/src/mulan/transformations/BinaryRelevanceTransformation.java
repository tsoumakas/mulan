/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mulan.transformations;

import java.io.Serializable;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author greg
 */
public class BinaryRelevanceTransformation implements Serializable {
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


	/**
	 * Remove all label attributes except that at indexOfLabelToKeep
	 */
	public static Instances transformInstances(Instances train, int[] labelIndices, int indexToKeep) throws Exception
	{
        int numLabels = labelIndices.length;
        
        train.setClassIndex(indexToKeep);


		// Indices of attributes to remove
		int[] indicesToRemove = new int[numLabels-1];
        int counter2=0;
        for (int counter1=0; counter1<numLabels; counter1++)
            if (labelIndices[counter1] != indexToKeep)
            {
                indicesToRemove[counter2] = labelIndices[counter1];
                counter2++;
            }

		Remove remove = new Remove();
		remove.setAttributeIndicesArray(indicesToRemove);
		remove.setInputFormat(train);
		remove.setInvertSelection(true);
		Instances result = Filter.useFilter(train, remove);
		return result;
	}


	/**
	 * Remove all label attributes except label at position indexToKeep
	 */
    public static Instance transformInstance(Instance instance, int[] labelIndices, int indexToKeep)
	{
        double[] values = instance.toDoubleArray();
		double[] transformedValues = new double[values.length
				- labelIndices.length + 1];

		int counterTransformed = 0;
		int counterLabelIndices = 0;
		boolean isLabel = false;
		
		for (int i = 0; i < values.length; i++) {
			if (counterLabelIndices < labelIndices.length) {
				for (int j = 0; j < labelIndices.length; j++) {
					if (i == labelIndices[j] && i != indexToKeep) {
						isLabel = true;
						break;
					} 
				}
				
			} 
			if(!isLabel){
				transformedValues[counterTransformed] = instance.value(i);
				counterTransformed++;
			}
			isLabel = false;
		}

		Instance transformedInstance = new Instance(1, transformedValues);
		return transformedInstance;
	}


}
