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
 * @version $Revision: 0.4 $
*/
public abstract class MultiLabelLearnerBase 
	implements TechnicalInformationHandler, MultiLabelLearner, Serializable {

	/**
	 * The number of labels the classifier should handle. 
	 * The label attributes are stored at the end of {@link Instances} data.
	 */
	protected int numLabels;

	/**
	 * An array containing the indexes of the label attributes within the
     * Instances object of the training data in increasing order. The same
     * order will be followed in the arrays of predictions given by each learner
     * in the {@link MultiLabelOutput} object.
	 */
	protected int[] labelIndices;

    /** Whether the classifier is run in debug mode. */
	protected boolean isDebug = false;

	public abstract TechnicalInformation getTechnicalInformation();
	
	public final void build(MultiLabelInstances trainingSet) throws Exception
    {
		if (trainingSet == null)
			throw new IllegalArgumentException("The dataSet is null.");
		
		numLabels = trainingSet.getNumLabels();
        labelIndices = trainingSet.getLabelIndices();
 
		buildInternal(trainingSet);
	}
	
	protected abstract void buildInternal(MultiLabelInstances trainingSet) throws Exception;

	/**
	 * Set debugging mode.
	 * 
	 * @param debug
	 *            true if debug output should be printed
	 */
	public void setDebug(boolean debug)
    {
		isDebug = debug;
	}
		
	/**
	 * Get whether debugging is turned on.
	 *
	 * @return true if debugging output is on
	 */
	public boolean getDebug()
    {
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
	
	public MultiLabelLearner makeCopy() throws Exception
    {
	    return (MultiLabelLearner) new SerializedObject(this).getObject();
	}
	
}
