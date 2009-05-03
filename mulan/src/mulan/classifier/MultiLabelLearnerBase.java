package mulan.classifier;

import java.io.Serializable;
import java.util.Date;

import mulan.core.data.MultiLabelInstances;

import weka.core.Instances;
import weka.core.SerializedObject;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;


/**
 * Common base class for all multi-label learner types.
 *
 * @author Robert Friberg
 * @author Jozef Vilcek
 * @author Grigorios Tsoumakas
 * @version $Revision: 0.3 $ 
*/
public abstract class MultiLabelLearnerBase 
	implements TechnicalInformationHandler, MultiLabelLearner, Serializable {

	/**
	 * The number of labels the classifier should handle. 
	 * The label attributes are stored at the end of {@link Instances} data.
	 */
	protected int numLabels;

	/** Whether the classifier is run in debug mode. */
	protected boolean isDebug = false;
		
//	/**
//	 * Creates a {@link MultiLabelLearnerBase} instance.
//	 * 
//	 * @param numLabels the number of labels the classifier should use
//	 */
//	public MultiLabelLearnerBase(final int numLabels) {
//		this.numLabels = numLabels;
//	}
	
	public int getNumLabels()
	{
		return numLabels;
	}
	
	public abstract TechnicalInformation getTechnicalInformation();
	
	public final void build(MultiLabelInstances dataSet) throws Exception{
		if(dataSet == null){
			throw new IllegalArgumentException("The dataSet is null.");
		}
		numLabels = dataSet.getNumLabels();
		dataSet.reorderLabels();
		buildInternal(dataSet);
	}
	
	protected abstract void buildInternal(MultiLabelInstances dataSet) throws Exception;
		
	

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
	
	public MultiLabelLearner makeCopy() throws Exception {      
	    return (MultiLabelLearner) new SerializedObject(this).getObject();
	}
	
}
