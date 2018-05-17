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
package mulan.classifier;

import java.io.Serializable;
import java.util.Date;
import mulan.core.ArgumentNullException;
import mulan.data.MultiLabelInstances;
import weka.core.*;

/**
 * Common root base class for all multi-label learner types.
 * Provides default implementation of {@link MultiLabelLearner} interface.
 *
 * @author Robert Friberg
 * @author Jozef Vilcek
 * @author Grigorios Tsoumakas
 * @version 2012.11.24
 */
public abstract class MultiLabelLearnerBase
        implements TechnicalInformationHandler, MultiLabelLearner, Serializable {

    private boolean isModelInitialized = false;
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
     * An array containing the names of the label attributes within the
     * {@link Instances} object of the training data in increasing order. The same
     * order will be followed in the arrays of predictions given by each learner
     * in the {@link MultiLabelOutput} object.
     */
    protected String[] labelNames;
    /**
     * An array containing the indexes of the feature attributes within the
     * {@link Instances} object of the training data in increasing order.
     */
    protected int[] featureIndices;
    /** Whether debugging is on/off */
    private boolean isDebug = false;

    public boolean isUpdatable() {
        /** as default learners are assumed not to be updatable */
        return false;
    }

    public final void build(MultiLabelInstances trainingSet) throws Exception {
        if (trainingSet == null) {
            throw new ArgumentNullException("trainingSet");
        }

        isModelInitialized = false;

        numLabels = trainingSet.getNumLabels();
        labelIndices = trainingSet.getLabelIndices();
        labelNames = trainingSet.getLabelNames();
        featureIndices = trainingSet.getFeatureIndices();

        buildInternal(trainingSet);
        isModelInitialized = true;
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
     * Gets whether learner's model is initialized by {@link #build(MultiLabelInstances)}.
     * This is used to check if {@link #makePrediction(weka.core.Instance)} can be processed.
     * @return isModelInitialized returns true if the model has been initialized
     */
    protected boolean isModelInitialized() {
        return isModelInitialized;
    }

    public final MultiLabelOutput makePrediction(Instance instance)
            throws Exception, InvalidDataException, ModelInitializationException {
        if (instance == null) {
            throw new ArgumentNullException("instance");
        }
        if (!isModelInitialized()) {
            throw new ModelInitializationException("The model has not been trained.");
        }

        return makePredictionInternal(instance);
    }

    /**
     * Learner specific implementation for predicting on specified data based on trained model.
     * This method is called from {@link #makePrediction(weka.core.Instance)} which guards for model
     * initialization and apply common handling/behavior.
     *
     * @param instance the data instance to predict on
     * @throws Exception if an error occurs while making the prediction.
     * @throws InvalidDataException if specified instance data is invalid and can not be processed by the learner
     * @return the output of the learner for the given instance
     */
    protected abstract MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException;

    /**
     * Set debugging mode.
     *
     * @param debug <code>true</code> if debug output should be printed
     */
    public void setDebug(boolean debug) {
        isDebug = debug;
    }

    /**
     * Get whether debugging is turned on.
     *
     * @return <code>true</code> if debugging output is on
     */
    public boolean getDebug() {
        return isDebug;
    }

    /**
     * Writes the debug message string to the console output
     * if debug for the learner is enabled.
     *
     * @param msg the debug message
     */
    protected void debug(String msg) {
        if (!getDebug()) {
            return;
        }
        System.err.println("" + new Date() + ": " + msg);
    }

    public MultiLabelLearner makeCopy() throws Exception {
        return (MultiLabelLearner) new SerializedObject(this).getObject();
    }


    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    abstract public TechnicalInformation getTechnicalInformation();    
    
}