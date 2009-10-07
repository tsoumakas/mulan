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

/*
*    MultiLabelLearnerBase.java
*    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
*
*/

package mulan.classifier;

import java.io.Serializable;
import java.util.Date;

import mulan.data.MultiLabelInstances;

import weka.core.Instances;
import weka.core.SerializedObject;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;


/**
 * Common root base class for all multi-label learner types.
 * Provides default implementation of {@link MultiLabelLearner} interface.
 *
 * @author Robert Friberg
 * @author Jozef Vilcek
 * @author Grigorios Tsoumakas
*/
public abstract class MultiLabelLearnerBase 
	implements TechnicalInformationHandler, MultiLabelLearner, Serializable {

	/**
	 * The number of labels the learner can handle. 
	 * The number of labels are determined form the training data when learner is build.  
	 */
	protected int numLabels;

	/**
	 * An array containing the indexes of the label attributes within the
     * {@link Instances} object of the training data in increasing order. The same
     * order will be followed in the arrays of predictions given by each learner
     * in the {@link MultiLabelOutput} object.
	 */
	protected int[] labelIndices;
	
	/**
	 * An array containing the indexes of the feature attributes within the
     * {@link Instances} object of the training data in increasing order.
	 */
	protected int[] featureIndices;

	private boolean isDebug = false;

	/**
	 * Gets the {@link TechnicalInformation} for the current learner type.
     *
     * @return technical information
     */
	public abstract TechnicalInformation getTechnicalInformation();
	
	public boolean isUpdatable(){
		/** as default learners are assumed not to be updatable */
		return false;
	}
	
	public final void build(MultiLabelInstances trainingSet) throws Exception
    {
		if (trainingSet == null)
			throw new IllegalArgumentException("The dataSet is null.");
		
		numLabels = trainingSet.getNumLabels();
        labelIndices = trainingSet.getLabelIndices();
        featureIndices = trainingSet.getFeatureIndices();
 
		buildInternal(trainingSet);
	}
	
	/**
	 * Learner specific implementation of building the model from {@link MultiLabelInstances}
	 * training data set. This method is called from {@link #build(MultiLabelInstances)} method, 
	 * where behavior common across all learners is applied. 
	 * 
	 * @param trainingSet the training data set
	 * @throws Exception if learner model was not created successfully 
	 */
	protected abstract void buildInternal(MultiLabelInstances trainingSet) throws Exception;

	/**
	 * Set debugging mode.
	 * 
	 * @param debug <code>true</code> if debug output should be printed
	 */
	public void setDebug(boolean debug)
    {
		isDebug = debug;
	}
		
	/**
	 * Get whether debugging is turned on.
	 *
	 * @return <code>true</code> if debugging output is on
	 */
	public boolean getDebug()
    {
		return isDebug;
	}
	
	/**
	 * Writes the debug message string to the console output 
	 * if debug for the learner is enabled.
	 * 
	 * @param msg the debug message
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
