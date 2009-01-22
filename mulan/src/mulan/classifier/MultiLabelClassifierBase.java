package mulan.classifier;

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

import java.util.List;

import weka.core.Instance;


/**
 * Common base class for all multi-label classifiers.
 *
 * @author Robert Friberg
 * @author Jozef Vilcek
 * @author Grigorios Tsoumakas
 * @version $Revision: 0.3 $ 
*/
public abstract class MultiLabelClassifierBase extends MultiLabelLearnerBase implements MultiLabelClassifier {
	 
 /*  TODO: Subset mapping stuff - decide if this will be reused somehow or discard
	public enum SubsetMappingMethod {
		NONE,
		GREEDY,
		PROBABILISTIC
	}

	protected SubsetMapper subsetMapper;
	protected HybridSubsetMapper hybridMapper;
	private SubsetMappingMethod subsetMappingMethod;
	private int subsetDistanceThreshold = -1;
 */
	
	/**
	 * Creates a {@link MultiLabelClassifierBase} instance.
	 * 
	 * @param numLabels the number of labels the classifier should use
	 */
	public MultiLabelClassifierBase(final int numLabels) {
		super(numLabels);
	}
	
	public final List<Boolean> predict(Instance instance) throws Exception
	{
            List<Boolean> original = makePrediction(instance);
		
/*          TODO: Subset mapping stuff - decide if this will be reused somehow or discard
            if (subsetMappingMethod == SubsetMappingMethod.GREEDY)
            {
                    return subsetMapper.nearestSubset(instance, original.predictedLabels);
            }
            else if (subsetMappingMethod == SubsetMappingMethod.PROBABILISTIC)
            {
                    return hybridMapper.nearestSubset(instance, original.predictedLabels);
            }
            else return original;
*/	
            return original;
	}
	
	
	/**
	 * Internal method for making prediction for passed {@link Instance}. 
	 * The method is called from {@link MultiLabelClassifier#predict(Instance)}.
	 * 
	 * @param instance the instance for which prediction is made
	 * @return the labels bipartition prediction for the instance
	 * @throws Exception if prediction was not successful
	 */
	protected abstract List<Boolean> makePrediction(Instance instance) throws Exception;

}
