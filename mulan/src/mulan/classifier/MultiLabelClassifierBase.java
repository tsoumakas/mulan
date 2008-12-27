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

import java.io.Serializable;
import java.util.Date;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializedObject;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;


/**
 * Common base class for all multi-label classifiers.
 *
 * @author Robert Friberg
 * @author Jozef Vilcek
 * @author Grigorios Tsoumakas
 * @version $Revision: 0.3 $ 
*/
public abstract class MultiLabelClassifierBase 
implements TechnicalInformationHandler, MultiLabelClassifier, Serializable {

	/**
	 * The number of labels the classifier should handle. 
	 * The label attributes are stored at the end of {@link Instances} data.
	 */
	protected final int numLabels;

	/** Whether the classifier is run in debug mode. */
	protected boolean isDebug = false;
	
        
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
		this.numLabels = numLabels;
	}
	
	/**
	 * {@inheritDoc}
	 */
	public int getNumLabels()
	{
		return numLabels;
	}
	
	/**
	 * {@inheritDoc}
	 */
	public abstract TechnicalInformation getTechnicalInformation();
	
	/**
	 * {@inheritDoc}
	 */
	public abstract void buildClassifier(Instances instances) throws Exception;

	public final Prediction predict(Instance instance) throws Exception
	{
            Prediction original = makePrediction(instance);
		
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
	 * @return the prediction for the instance
	 * @throws Exception if prediction was not successful
	 */
	protected abstract Prediction makePrediction(Instance instance) throws Exception;


	/**
	 * Set debugging mode.
	 * 
	 * @param debug
	 *            true if debug output should be printed
	 */
	public void setDebug(boolean debug) {
		isDebug = debug;
	}
		
	/**
	 * Get whether debugging is turned on.
	 *
	 * @return true if debugging output is on
	 */
	public boolean getDebug() {
		return isDebug;
	}
	
	/**
	 * Writes the debug message string to the console output.
	 * @param msg the message
	 */
	protected void debug(String msg)
	{
		if (!getDebug()) return;
			System.err.println("" + new Date() + ": " + msg);
	}
	
	
	/**
	 * Creates a deep copy of the given classifier using serialization.
	 *
	 * @param model the classifier to copy
	 * @return a deep copy of the classifier
	 * @exception Exception if an error occurs
	 */
	public static MultiLabelClassifier makeCopy(MultiLabelClassifier model) 
	throws Exception {      
	    return (MultiLabelClassifier) new SerializedObject(model).getObject();
	}
	
}
