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

import mulan.data.MultiLabelInstances;
import weka.core.Instance;

/**
 * Common root interface for all multi-label learner types.
 *  
 * @author Jozef Vilcek
 */
public interface MultiLabelLearner {

    /**
     * Returns value indicating if learner is updatable, so if learner is able to
     * perform on-line learning. The fact if learner is updatable or not influences
     * the behavior of {@link MultiLabelLearner#build(MultiLabelInstances)} method.<br>
     * When <code>false</code> is returned, each call of the
     * {@link MultiLabelLearner#build(MultiLabelInstances)} will initialize the learner from
     * the scratch, removing any potential knowledge built by previously entered training data.
     * When <code>true</code> is returned, then on the first call of the
     * {@link MultiLabelLearner#build(MultiLabelInstances)} the learner is initialized
     * with the passed training data. All other calls contribute to the existing learner's
     * model with new data.<br>
     *
     * @return <code>true</code> if learner is updatable (on-line), <code>false</code> otherwise.
     */
    public boolean isUpdatable();

    /**
     * Builds the learner model from specified {@link MultiLabelInstances} data.
     * Sequential calls to this method either re-build the learners model with new data
     * (off-line learner) or contribute to the existing model with new data (on-line learner).
     * The behavior is determined by the outcome of {@link MultiLabelLearner#isUpdatable()} method.
     *
     * @param instances set of training data, upon which the learner model should be built
     * @throws Exception if learner model was not created successfully
     * @throws InvalidDataException if specified instances data is invalid or not supported by the learner
     * @see MultiLabelLearner#isUpdatable()
     */
    public void build(MultiLabelInstances instances) throws Exception, InvalidDataException;

    /**
     * Creates a deep copy of the given learner using serialization.
     *
     * @return a deep copy of the learner
     * @exception Exception if an error occurs while making copy of the learner.
     */
    public MultiLabelLearner makeCopy() throws Exception;

    /**
     * Returns the prediction of the learner for a given input {@link Instance}.
     *
     * @param instance the input given to the learner in the form of {@link Instance}
     * @return a prediction of the learner in form of {@link MultiLabelOutput}.
     * @throws Exception if an error occurs while making the prediction.
     * @throws InvalidDataException if specified instance data is invalid and can not be processed by the learner
     * @throws ModelInitializationException if method is called before {@link MultiLabelLearner#build(MultiLabelInstances)}
     */
    public MultiLabelOutput makePrediction(Instance instance)
            throws Exception, InvalidDataException, ModelInitializationException;

    /**
     * Sets whether debugging information should be output by the model
     * 
     * @param debug True to show debug information, False not to
     */
    public void setDebug(boolean debug);
}