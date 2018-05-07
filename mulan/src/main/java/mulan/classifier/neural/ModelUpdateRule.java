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
 *    ModelUpdateRule.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.classifier.neural;

import java.util.Map;

/**
 * Represents an update rule, which can be used by a learner, to process an input 
 * example in learning phase and perform an update of a model when necessary.
 * 
 * @author Jozef Vilcek
 * @version 2012.02.27
 */
/* REMARK: This might be generalized to the all learners, if distinguish between model and learner will be made in design of the learners  */
public interface ModelUpdateRule {

    /**
     * Process the training example and performs a model update when suitable.
     * The decision when to perform model update is carried by the update rule
     * (e.g. when the model response is not within an acceptable boundaries from
     * the true output for given example).
     *
     * @param example the input example
     * @param params the additional parameters for an update.
     * @return the error measure of the model response for given input pattern
     * and specified true output.
     */
    public double process(DataPair example, Map<String, Object> params);
}