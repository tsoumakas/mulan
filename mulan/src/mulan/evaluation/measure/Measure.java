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
 *    Measure.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.evaluation.measure;

import mulan.classifier.MultiLabelOutput;

/**
 * Interface for a measure, used to evaluate the performance of a multi-label learner
 * on when performing multi-label learning task. Various measures capture different 
 * characteristics of a learning task performance. 
 * 
 * @author Jozef Vilcek
 */
public interface Measure {

    /**
     * Gets the name of a measure.
     * @return the name of a measure.
     */
    String getName();

    /**
     * Gets the value of a measure. The measure is incrementally cumulated for learner's
     * prediction by each {@link Measure#update(MultiLabelOutput, boolean[])} call. The value
     * returned by the method, returns sum of all update calls divided by the number
     * of calls (average of all measures for all predictions).
     *
     * @return the average measure value computed so far
     */
    double getValue();

    /**
     * Gets an 'ideal' value of a measure. The 'ideal' means, that the value
     * represents the best achievable performance of a learner for an prediction of
     * a multi-label task and associated true labels.
     *
     * @return the ideal value
     */
    double getIdealValue();

    /**
     * Computes the value of a measure for the given prediction and true labels. The immediate value of
     * a measure is	returned and result is added to the cumulated measure value.
     *
     * @param prediction the prediction for which measure has to be computed
     * @param truth the true labels bipartition for given prediction
     * @see Measure#getValue()
     */
    void update(MultiLabelOutput prediction, boolean[] truth);

    /**
     * Creates a deep copy of the given measure using serialization.
     *
     * @return a deep copy of the measure
     * @exception Exception if an error occurs while making copy of the measure.
     */
    public Measure makeCopy() throws Exception;

    /**
     * Resets the cumulated measure value, so the process of computation can be started
     * from beginning (e.g. for a new series of outputs from learning task).
     */
    void reset();
}
